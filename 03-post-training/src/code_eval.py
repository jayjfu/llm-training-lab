import argparse
import ast
import json
import os
import sys
from math import comb
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from vllm import LLM, SamplingParams

from data import CODE_PROMPT
from rewards import extract_code
from sandbox import run_command

BENCHMARKS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def problem_text(dataset, problem):
    if dataset == "humaneval":
        return f"Complete the following function.\n```python\n{problem['prompt'].strip()}\n```"
    return problem["prompt"].strip().strip('"').strip()


def build_prompt(dataset, problem, tokenizer, chat):
    text = CODE_PROMPT + problem_text(dataset, problem)
    if chat:
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return text + "\n\n```python\n"


def strip_script(code):
    # drop top-level demo code (prints, example asserts, __main__ blocks); evalplus' sanitize can drop needed helpers
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree.body = [node for node in tree.body if isinstance(node, DEFINITIONS)]
    return ast.unparse(tree)


def to_solution(dataset, problem, output, chat):
    raw = output if chat else f"```python\n{output}\n```"
    code = strip_script(extract_code(raw))
    if dataset == "humaneval":
        # HumanEval prompts carry imports (e.g. typing.List) that models often drop when rewriting the function
        imports = [line for line in problem["prompt"].splitlines() if line.startswith(("import ", "from "))]
        code = "\n".join(imports + [code])
    return code


def pass_at_k(num_samples, num_correct, k):
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def score(dataset, samples_path, num_samples, timeout):
    cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    command = [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", str(samples_path), "--i_just_wanna_run"]
    result = run_command(command, samples_path.parent, timeout, memory_mb=0, file_mb=1024, extra_env={"XDG_CACHE_HOME": cache_home})
    if result.timed_out or result.returncode != 0:
        raise SystemExit(f"evalplus failed (timed_out={result.timed_out}):\n{result.stderr}")

    eval_results = json.loads(samples_path.with_name(samples_path.stem + "_eval_results.json").read_text())["eval"]
    base_correct = [sum(sample["base_status"] == "pass" for sample in samples) for samples in eval_results.values()]
    plus_correct = [sum(sample["base_status"] == sample["plus_status"] == "pass" for sample in samples) for samples in eval_results.values()]
    metrics = {}
    for k in (1, 10, 100):
        if k <= num_samples:
            metrics[f"pass@{k},none"] = sum(pass_at_k(num_samples, correct, k) for correct in plus_correct) / len(plus_correct)
            metrics[f"pass@{k}_base,none"] = sum(pass_at_k(num_samples, correct, k) for correct in base_correct) / len(base_correct)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=list(BENCHMARKS))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_chat_template", action="store_true", help="plain-text prompt for base models")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    dataset, load_problems = BENCHMARKS[args.task]
    problems = load_problems()
    chat = not args.no_chat_template
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed)
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt(dataset, problem, tokenizer, chat) for problem in problems.values()]
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop=None if chat else ["```"],
    )
    outputs = llm.generate(prompts, sampling_params)

    samples_path = output_dir / "samples.jsonl"
    with open(samples_path, "w") as samples_file:
        for (task_id, problem), output in zip(problems.items(), outputs):
            for sample in output.outputs:
                solution = to_solution(dataset, problem, sample.text, chat)
                samples_file.write(json.dumps({"task_id": task_id, "solution": solution, "raw": sample.text, "finish_reason": sample.finish_reason}) + "\n")

    metrics = score(dataset, samples_path, args.num_samples, args.timeout)
    (output_dir / "results_code.json").write_text(json.dumps({"results": {args.task: metrics}, "config": vars(args)}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
