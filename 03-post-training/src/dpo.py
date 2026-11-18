import json

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

from common import completion_logprobs, load_model, make_run_dir, parse_args, save_final
from data import load_pairs


def encode_pair(tokenizer, example):
    prompt_text = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
    encoded = {"prompt_ids": tokenizer(prompt_text, add_special_tokens=False).input_ids}
    for key in ("chosen", "rejected"):
        full_text = tokenizer.apply_chat_template(example["prompt"] + example[key], tokenize=False)
        encoded[f"{key}_ids"] = tokenizer(full_text[len(prompt_text):], add_special_tokens=False).input_ids
    return encoded


def collate(batch, pad_token_id, device):
    # chosen rows, then rejected rows
    rows = [(example["prompt_ids"], example["chosen_ids"]) for example in batch]
    rows += [(example["prompt_ids"], example["rejected_ids"]) for example in batch]
    prompt_length = max(len(prompt_ids) for prompt_ids, _ in rows)
    completion_length = max(len(completion_ids) for _, completion_ids in rows)
    sequences, attention_mask, completion_mask = [], [], []
    for prompt_ids, completion_ids in rows:
        prompt_pad = prompt_length - len(prompt_ids)
        completion_pad = completion_length - len(completion_ids)
        sequences.append([pad_token_id] * prompt_pad + prompt_ids + completion_ids + [pad_token_id] * completion_pad)
        attention_mask.append([0] * prompt_pad + [1] * (len(prompt_ids) + len(completion_ids)) + [0] * completion_pad)
        completion_mask.append([1] * len(completion_ids) + [0] * completion_pad)
    return (
        torch.tensor(sequences, device=device),
        torch.tensor(attention_mask, device=device),
        torch.tensor(completion_mask, dtype=torch.float32, device=device),
    )


def sequence_logps(model, sequences, attention_mask, completion_mask):
    token_logps = completion_logprobs(model, sequences, attention_mask, completion_mask.shape[1])
    return (token_logps * completion_mask).sum(dim=1)


def dpo_loss(policy_logps, ref_logps, beta):
    chosen_rewards, rejected_rewards = (beta * (policy_logps - ref_logps)).chunk(2)
    margins = chosen_rewards - rejected_rewards
    return -functional.logsigmoid(margins).mean(), chosen_rewards.detach(), rejected_rewards.detach()


def train(config):
    train_args = config["train"]
    torch.manual_seed(train_args["seed"])
    run_dir = make_run_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = load_model(config["model"], torch.float32).to(device)
    model.gradient_checkpointing_enable()
    ref_model = load_model(config["model"], torch.bfloat16).to(device).eval().requires_grad_(False)

    pairs = load_pairs(config)
    dataset = pairs.map(lambda example: encode_pair(tokenizer, example), remove_columns=pairs.column_names)
    dataloader = DataLoader(
        dataset, batch_size=train_args["pairs_per_step"], shuffle=True, drop_last=True,
        collate_fn=lambda batch: collate(batch, pad_token_id, device),
    )
    total_steps = len(dataloader) * train_args["epochs"]
    if train_args["max_steps"] > 0:
        total_steps = min(total_steps, train_args["max_steps"])

    optimizer = AdamW(model.parameters(), lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"])
    warmup_steps = train_args["warmup_steps"]
    if warmup_steps < 1:
        warmup_steps = int(warmup_steps * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    step = 0
    progress = tqdm(total=total_steps)
    model.train()
    with open(run_dir / "metrics.jsonl", "w") as metrics_file:
        while step < total_steps:
            for sequences, attention_mask, completion_mask in dataloader:
                if step >= total_steps:
                    break
                with torch.no_grad():
                    ref_logps = sequence_logps(ref_model, sequences, attention_mask, completion_mask)
                policy_logps = sequence_logps(model, sequences, attention_mask, completion_mask)
                loss, chosen_rewards, rejected_rewards = dpo_loss(policy_logps, ref_logps, train_args["beta"])

                loss.backward()
                clip_grad_norm_(model.parameters(), train_args["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                metrics = {
                    "step": step,
                    "loss": loss.item(),
                    "chosen_reward": chosen_rewards.mean().item(),
                    "rejected_reward": rejected_rewards.mean().item(),
                    "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
                    "margin": (chosen_rewards - rejected_rewards).mean().item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                metrics_file.write(json.dumps(metrics) + "\n")
                progress.update(1)
                progress.set_postfix({key: f"{value:.4f}" for key, value in metrics.items() if key in ("loss", "reward_accuracy", "margin")})

    save_final(model, AutoProcessor.from_pretrained(config["model"]), run_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        raise SystemExit("--resume is not supported by the from-scratch DPO yet")
    for config in args.configs:
        train(config)