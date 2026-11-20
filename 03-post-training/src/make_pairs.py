import json

from vllm import LLM, SamplingParams

from common import pairs_path, parse_args
from data import task_dataset
from rewards import reward_funcs


def make_pairs(config):
    pairs_args = config["pairs"]
    output_path = pairs_path(config)
    if output_path.exists():
        print(f"skip: {output_path} exists")
        return

    dataset = task_dataset(config)
    funcs = reward_funcs(config)
    llm = LLM(model=config["model"], dtype="bfloat16", seed=pairs_args["seed"])
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in dataset["prompt"]]
    sampling_params = SamplingParams(
        n=pairs_args["num_generations"],
        temperature=pairs_args["temperature"],
        top_p=pairs_args["top_p"],
        top_k=pairs_args["top_k"] if pairs_args["top_k"] > 0 else -1,
        repetition_penalty=pairs_args["repetition_penalty"],
        max_tokens=pairs_args["max_completion_length"],
        seed=pairs_args["seed"],
    )
    outputs = llm.generate(prompts, sampling_params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    with open(output_path, "w") as pairs_file:
        for example, output in zip(dataset, outputs):
            # drop truncated samples
            samples = [sample.text for sample in output.outputs if sample.finish_reason == "stop"]
            if len(samples) < 2:
                continue
            per_func = [reward_func(samples, ground_truth=[example["ground_truth"]] * len(samples)) for reward_func in funcs]
            rewards = [sum(values) for values in zip(*per_func)]
            best = max(range(len(samples)), key=rewards.__getitem__)
            worst = min(range(len(samples)), key=rewards.__getitem__)
            if rewards[best] == rewards[worst]:
                continue
            pair = {
                "prompt": example["prompt"],
                "chosen": [{"role": "assistant", "content": samples[best]}],
                "rejected": [{"role": "assistant", "content": samples[worst]}],
            }
            pairs_file.write(json.dumps(pair) + "\n")
            num_pairs += 1
    print(f"{num_pairs} pairs from {len(dataset)} prompts -> {output_path}")


if __name__ == "__main__":
    for config in parse_args().configs:
        make_pairs(config)