from datasets import load_dataset

from common import pairs_path
from rewards import extract_boxed

MATH_PROMPT = "Solve the following math competition problem step by step. Put the final answer inside \\boxed{}.\n\n"


def math_dataset(name, levels):
    levels = set(levels)
    raw = load_dataset(name, split="train").filter(lambda example: example["level"] in levels)
    dataset = raw.map(
        lambda example: {
            "prompt": [{"role": "user", "content": MATH_PROMPT + example["problem"]}],
            "ground_truth": extract_boxed(example["solution"]),
        },
        remove_columns=raw.column_names,
    )
    return dataset.filter(lambda example: example["ground_truth"] != "")


def split_turns(batch):
    prompts, completions = [], []
    for messages in batch["messages"]:
        for index, message in enumerate(messages):
            if message["role"] == "assistant" and any(turn["role"] == "user" for turn in messages[:index]):
                prompts.append(messages[:index])
                completions.append([message])
    return {"prompt": prompts, "completion": completions}


def sft_dataset(name, tokenizer, max_length, max_train_samples=None, seed=42, num_proc=None):
    raw = load_dataset(name, split="train")
    if max_train_samples:
        raw = raw.shuffle(seed=seed).select(range(min(max_train_samples, len(raw))))
    dataset = raw.map(split_turns, batched=True, remove_columns=raw.column_names, num_proc=num_proc)

    def fits(example):
        text = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)
        return len(tokenizer(text).input_ids) <= max_length

    return dataset.filter(fits, num_proc=num_proc)


def load_pairs(config):
    path = pairs_path(config)
    if not path.exists():
        raise SystemExit(f"{path} not found; run src/make_pairs.py first")
    return load_dataset("json", data_files=str(path), split="train")