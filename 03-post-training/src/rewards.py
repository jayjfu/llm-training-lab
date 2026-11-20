import re

from math_verify import parse, verify

from sandbox import run_tests_batch

BOXED = "\\boxed{"
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_boxed(text):
    start = text.rfind(BOXED)
    if start == -1:
        return ""
    begin = start + len(BOXED)
    depth = 0
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            if depth == 0:
                return text[begin:index].strip()
            depth -= 1
    return ""


def is_equivalent(prediction, answer):
    if "".join(prediction.split()) == "".join(answer.split()):
        return True
    return verify(parse(f"${answer}$"), parse(f"${prediction}$"))


def completion_text(completion):
    return completion[-1]["content"] if isinstance(completion, list) else completion


def accuracy_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, answer in zip(completions, ground_truth):
        prediction = extract_boxed(completion_text(completion))
        rewards.append(1.0 if prediction and is_equivalent(prediction, answer) else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    return [0.5 if extract_boxed(completion_text(completion)) else 0.0 for completion in completions]


def extract_code(text):
    blocks = CODE_BLOCK.findall(text)
    blocks = [block for block in blocks if "def " in block] or blocks
    return blocks[-1] if blocks else ""


def test_results(completions, ground_truth):
    codes = [extract_code(completion_text(completion)) for completion in completions]
    return run_tests_batch(codes, ground_truth)


def code_pass_reward(completions, ground_truth, **kwargs):
    return [1.0 if results and all(results) else 0.0 for results in test_results(completions, ground_truth)]


def code_fraction_reward(completions, ground_truth, **kwargs):
    return [sum(results) / len(results) if results else 0.0 for results in test_results(completions, ground_truth)]


def code_format_reward(completions, **kwargs):
    return [0.1 if extract_code(completion_text(completion)) else 0.0 for completion in completions]


REWARDS = {
    "math_accuracy": accuracy_reward,
    "math_format": format_reward,
    "code_pass": code_pass_reward,
    "code_fraction": code_fraction_reward,
    "code_format": code_format_reward,
}
DEFAULT_REWARDS = {"math": ["math_accuracy", "math_format"], "code": ["code_pass"]}


def reward_funcs(config):
    names = config.get("rewards") or DEFAULT_REWARDS[config.get("task", "math")]
    return [REWARDS[name] for name in names]