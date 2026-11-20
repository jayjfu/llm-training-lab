import json
import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

from common import completion_logprobs, load_model, make_run_dir, parse_args, save_final
from data import task_dataset
from rewards import reward_funcs, test_results


def compute_rewards(funcs, completions, ground_truth):
    per_func = {reward_func.__name__: reward_func(completions, ground_truth=ground_truth) for reward_func in funcs}
    totals = torch.tensor([sum(values) for values in zip(*per_func.values())], dtype=torch.float32)
    return totals, per_func


def compute_advantages(rewards, num_generations, scale_rewards=True, eps=1e-4):
    grouped = rewards.view(-1, num_generations)
    advantages = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_rewards:
        advantages = advantages / (grouped.std(dim=1, keepdim=True) + eps)
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
    truncated = (~is_eos.any(dim=1)).tolist()
    return sequences, attention_mask, completion_mask.float(), completions, truncated, None


def make_vllm(config, train_args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    return LLM(
        model=config["model"],
        dtype="bfloat16",
        seed=train_args["seed"],
        gpu_memory_utilization=train_args.get("vllm_gpu_memory_utilization", 0.12),
        max_model_len=train_args.get("vllm_max_model_len", 4096),
        max_num_seqs=train_args["prompts_per_step"] * train_args["num_generations"],
        language_model_only=True,  # text only
    )


@torch.no_grad()
def vllm_rollout(llm, tokenizer, prompts, eos_ids, train_args, device):
    from vllm import SamplingParams

    texts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
    prompt_ids = [tokenizer(text, add_special_tokens=False).input_ids for text in texts]
    sampling_params = SamplingParams(
        n=train_args["num_generations"],
        temperature=train_args["temperature"],
        top_p=train_args["top_p"],
        top_k=train_args["top_k"],
        repetition_penalty=train_args["repetition_penalty"],
        max_tokens=train_args["max_completion_length"],
        stop_token_ids=eos_ids,
        logprobs=0,
    )
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params, use_tqdm=False)
    samples = [(ids, sample) for ids, output in zip(prompt_ids, outputs) for sample in output.outputs]

    prompt_width = max(len(ids) for ids in prompt_ids)
    completion_width = max(1, max(len(sample.token_ids) for _, sample in samples))
    sequences = torch.full((len(samples), prompt_width + completion_width), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(sequences)
    completion_mask = torch.zeros(len(samples), completion_width)
    vllm_logps = torch.zeros(len(samples), completion_width)
    for row, (ids, sample) in enumerate(samples):
        tokens = list(sample.token_ids)
        sequences[row, prompt_width - len(ids):prompt_width] = torch.tensor(ids)
        sequences[row, prompt_width:prompt_width + len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, prompt_width - len(ids):prompt_width + len(tokens)] = 1
        completion_mask[row, :len(tokens)] = 1
        vllm_logps[row, :len(tokens)] = torch.tensor([logprobs[token].logprob for logprobs, token in zip(sample.logprobs, tokens)])

    completions = tokenizer.batch_decode([sample.token_ids for _, sample in samples], skip_special_tokens=True)
    truncated = [sample.finish_reason == "length" for _, sample in samples]
    return sequences.to(device), attention_mask.to(device), completion_mask.to(device), completions, truncated, vllm_logps.to(device)


@torch.no_grad()
def sync_vllm(llm, model):
    weights = model.state_dict()
    llm.apply_model(lambda vllm_model: vllm_model.load_weights(weights.items()))
    llm.reset_prefix_cache()


def grpo_loss(policy_logps, old_logps, ref_logps, advantages, completion_mask, train_args, batch_sequences):
    beta = train_args["beta"]
    epsilon_low = train_args["epsilon"]
    epsilon_high = train_args.get("epsilon_high", epsilon_low)
    ratio = torch.exp(policy_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
    surrogate = torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))

    log_ratio_ref = ref_logps - policy_logps
    token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    token_loss = -(surrogate - beta * token_kl)

    def sequence_mean(values):
        return (values * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)

    loss = sequence_mean(token_loss).sum() / batch_sequences
    return loss, sequence_mean(token_kl).sum().item(), sequence_mean(ratio).sum().item()


def chunked_logprobs(model, sequences, attention_mask, completion_length, chunks):
    return torch.cat([completion_logprobs(model, sequences[chunk], attention_mask[chunk], completion_length) for chunk in chunks])


def train(config):
    train_args = config["train"]
    torch.manual_seed(train_args["seed"])
    run_dir = make_run_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(config["model"])
    model = load_model(config["model"], torch.float32).to(device)
    model.gradient_checkpointing_enable()
    ref_model = load_model(config["model"], torch.bfloat16).to(device).eval().requires_grad_(False)
    llm = make_vllm(config, train_args) if train_args.get("use_vllm") else None

    eos_ids = stop_token_ids(model, tokenizer)
    dataset = task_dataset(config)
    funcs = reward_funcs(config)
    dataloader = DataLoader(
        dataset, batch_size=train_args["prompts_per_step"], shuffle=True, drop_last=True, collate_fn=lambda examples: examples
    )
    log_rollouts_every = train_args.get("log_rollouts_every", 0)
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
    with open(run_dir / "metrics.jsonl", "w") as metrics_file, open(run_dir / "rollouts.jsonl", "w") as rollouts_file:
        while step < total_steps:
            for batch in dataloader:
                if step >= total_steps:
                    break
                prompts = [example["prompt"] for example in batch]
                if llm is None:
                    rollout_outputs = rollout(model, tokenizer, prompts, eos_ids, train_args)
                else:
                    rollout_outputs = vllm_rollout(llm, tokenizer, prompts, eos_ids, train_args, device)
                sequences, attention_mask, completion_mask, completions, truncated, vllm_logps = rollout_outputs
                ground_truth = [answer for answer in [example["ground_truth"] for example in batch] for _ in range(train_args["num_generations"])]
                rewards, per_func = compute_rewards(funcs, completions, ground_truth)
                advantages = compute_advantages(rewards, train_args["num_generations"], train_args.get("scale_rewards", True)).to(device)

                completion_length = completion_mask.shape[1]
                batch_sequences = len(sequences)
                micro_batch_size = train_args.get("micro_batch_size") or batch_sequences
                chunks = [slice(start, start + micro_batch_size) for start in range(0, batch_sequences, micro_batch_size)]
                with torch.no_grad():
                    old_logps = chunked_logprobs(model, sequences, attention_mask, completion_length, chunks)
                    ref_logps = chunked_logprobs(ref_model, sequences, attention_mask, completion_length, chunks)

                for _ in range(train_args["num_iterations"]):
                    loss, mean_kl, mean_ratio = 0.0, 0.0, 0.0
                    for chunk in chunks:
                        policy_logps = completion_logprobs(model, sequences[chunk], attention_mask[chunk], completion_length)
                        chunk_loss, kl_sum, ratio_sum = grpo_loss(
                            policy_logps, old_logps[chunk], ref_logps[chunk], advantages[chunk], completion_mask[chunk],
                            train_args, batch_sequences,
                        )
                        chunk_loss.backward()
                        loss += chunk_loss.item()
                        mean_kl += kl_sum / batch_sequences
                        mean_ratio += ratio_sum / batch_sequences
                    clip_grad_norm_(model.parameters(), train_args["max_grad_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                if llm is not None:
                    sync_vllm(llm, model)
                step += 1

                metrics = {
                    "step": step,
                    "loss": loss,
                    "reward": rewards.mean().item(),
                    **{name: sum(values) / len(values) for name, values in per_func.items()},
                    "zero_std_groups": (rewards.view(-1, train_args["num_generations"]).std(dim=1) == 0).float().mean().item(),
                    "truncated": sum(truncated) / len(truncated),
                    "kl": mean_kl,
                    "ratio": mean_ratio,
                    "completion_length": completion_mask.sum(dim=1).mean().item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                if vllm_logps is not None:
                    gap = ((vllm_logps - old_logps).abs() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                    metrics["vllm_logprob_gap"] = gap.item()
                metrics_file.write(json.dumps(metrics) + "\n")
                if log_rollouts_every and step % log_rollouts_every == 0:
                    tests = test_results(completions, ground_truth) if config.get("task") == "code" else None
                    prompt_texts = [example["prompt"][-1]["content"] for example in batch for _ in range(train_args["num_generations"])]
                    for index, completion in enumerate(completions):
                        record = {
                            "step": step,
                            "group": index // train_args["num_generations"],
                            "prompt": prompt_texts[index],
                            "completion": completion,
                            "rewards": {name: values[index] for name, values in per_func.items()},
                            "advantage": advantages[index].item(),
                            "length": int(completion_mask[index].sum().item()),
                            "truncated": truncated[index],
                        }
                        if tests is not None:
                            record["tests"] = list(tests[index])
                        rollouts_file.write(json.dumps(record) + "\n")
                progress.update(1)
                progress.set_postfix({key: f"{value:.4f}" for key, value in metrics.items() if key in ("loss", "reward", "kl")})

    save_final(model, processor, run_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        raise SystemExit("--resume is not supported by the from-scratch GRPO yet")
    for config in args.configs:
        train(config)