import json

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

from common import completion_logprobs, load_model, make_run_dir, parse_args, save_final
from data import math_dataset
from rewards import REWARD_FUNCS


def compute_rewards(completions, ground_truth):
    per_func = [reward_func(completions, ground_truth) for reward_func in REWARD_FUNCS]
    return torch.tensor([sum(values) for values in zip(*per_func)], dtype=torch.float32)


def compute_advantages(rewards, num_generations, eps=1e-4):
    grouped = rewards.view(-1, num_generations)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True) + eps)
    return advantages.view(-1)


def stop_token_ids(model, tokenizer):
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [] if eos_ids is None else [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    return list(dict.fromkeys([tokenizer.eos_token_id, *eos_ids]))


@torch.no_grad()
def rollout(model, tokenizer, prompts, eos_ids, train_args):
    texts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    model.eval()
    with torch.autocast(model.device.type, dtype=torch.bfloat16):
        sequences = model.generate(
            **inputs,
            do_sample=True,
            temperature=train_args["temperature"],
            top_p=train_args["top_p"],
            top_k=train_args["top_k"],
            repetition_penalty=train_args["repetition_penalty"],
            eos_token_id=eos_ids,
            max_new_tokens=train_args["max_completion_length"],
            num_return_sequences=train_args["num_generations"],
            pad_token_id=tokenizer.pad_token_id,
        )
    model.train()

    completion_ids = sequences[:, inputs.input_ids.shape[1]:]
    is_eos = torch.isin(completion_ids, torch.tensor(eos_ids, device=completion_ids.device))
    eos_index = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1), completion_ids.shape[1])
    positions = torch.arange(completion_ids.shape[1], device=sequences.device)
    completion_mask = (positions.unsqueeze(0) <= eos_index.unsqueeze(1)).long()
    prompt_mask = inputs.attention_mask.repeat_interleave(train_args["num_generations"], dim=0)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return sequences, attention_mask, completion_mask.float(), completions


def grpo_loss(policy_logps, old_logps, ref_logps, advantages, completion_mask, beta, epsilon):
    ratio = torch.exp(policy_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))

    log_ratio_ref = ref_logps - policy_logps
    token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    token_loss = -(surrogate - beta * token_kl)

    def sequence_mean(values):
        return (values * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)

    loss = sequence_mean(token_loss).mean()
    return loss, sequence_mean(token_kl).mean().item(), sequence_mean(ratio).mean().item()


def train(config):
    train_args = config["train"]
    torch.manual_seed(train_args["seed"])
    run_dir = make_run_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(config["model"], torch.float32).to(device)
    model.gradient_checkpointing_enable()
    ref_model = load_model(config["model"], torch.bfloat16).to(device).eval().requires_grad_(False)

    eos_ids = stop_token_ids(model, tokenizer)
    dataset = math_dataset(config["dataset"], config["levels"])
    dataloader = DataLoader(
        dataset, batch_size=train_args["prompts_per_step"], shuffle=True, drop_last=True, collate_fn=lambda examples: examples
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
    with open(run_dir / "metrics.jsonl", "w") as metrics_file:
        while step < total_steps:
            for batch in dataloader:
                if step >= total_steps:
                    break
                sequences, attention_mask, completion_mask, completions = rollout(model, tokenizer, [example["prompt"] for example in batch], eos_ids, train_args)
                ground_truth = [answer for answer in [example["ground_truth"] for example in batch] for _ in range(train_args["num_generations"])]
                rewards = compute_rewards(completions, ground_truth)
                advantages = compute_advantages(rewards, train_args["num_generations"]).to(device)

                completion_length = completion_mask.shape[1]
                with torch.no_grad():
                    old_logps = completion_logprobs(model, sequences, attention_mask, completion_length)
                    ref_logps = completion_logprobs(ref_model, sequences, attention_mask, completion_length)

                for _ in range(train_args["num_iterations"]):
                    policy_logps = completion_logprobs(model, sequences, attention_mask, completion_length)
                    loss, mean_kl, mean_ratio = grpo_loss(
                        policy_logps, old_logps, ref_logps, advantages, completion_mask,
                        train_args["beta"], train_args["epsilon"],
                    )
                    loss.backward()
                    clip_grad_norm_(model.parameters(), train_args["max_grad_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                step += 1

                metrics = {
                    "step": step,
                    "loss": loss.item(),
                    "reward": rewards.mean().item(),
                    "kl": mean_kl,
                    "ratio": mean_ratio,
                    "completion_length": completion_mask.sum(dim=1).mean().item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                metrics_file.write(json.dumps(metrics) + "\n")
                progress.update(1)
                progress.set_postfix({key: f"{value:.4f}" for key, value in metrics.items() if key in ("loss", "reward", "kl")})

    save_final(model, AutoProcessor.from_pretrained(config["model"]), run_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        raise SystemExit("--resume is not supported by the from-scratch GRPO yet")
    for config in args.configs:
        train(config)