import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pretraining.model import GPT, GPTConfig
from pretraining.optim import build_optimizers
from pretraining.schedule import cosine_schedule, wsd_schedule
from pretraining.tokenizer import load_tokenizer
from pretraining.data import prepare_data, DataReader
from pretraining.hellaswag import load_hellaswag, evaluate_hellaswag

ROOT = Path(__file__).resolve().parents[2]  # 01-pretraining
EVAL_SEED = 1234  # same eval batches for every run
SAMPLE_PROMPTS = ["The capital of France is", "In a shocking finding, scientists discovered", "The best way to learn programming is"]


def fixed_batches(path, batch_size, sequence_length, n_batches):
    reader = DataReader(path, batch_size, sequence_length, seed=EVAL_SEED, with_replacement=True)
    return [reader.sample_batch() for _ in range(n_batches)]


def token_bytes_table(tokenizer, model_vocab_size, device):
    table = torch.zeros(model_vocab_size, dtype=torch.float32)  # special and padding tokens stay at 0 bytes
    table[: tokenizer.vocab_size] = torch.tensor(tokenizer.token_bytes(), dtype=torch.float32)
    return table.to(device)


@torch.no_grad()
def evaluate(model, eval_sets, token_bytes, autocast, device):
    model.eval()
    stats = {"max_logit": 0.0}
    for split, batches in eval_sets.items():
        loss_sum, token_count, bpb_nats, byte_count, correct = 0.0, 0, 0.0, 0.0, 0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            with autocast():
                logits, _ = model(x)
            targets = y.view(-1)
            per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, reduction="none")
            byte_counts = token_bytes[targets]
            loss_sum += per_token.sum().item()
            token_count += targets.numel()
            bpb_nats += per_token[byte_counts > 0].sum().item()
            byte_count += byte_counts.sum().item()
            correct += (logits.argmax(-1) == y).sum().item()
            stats["max_logit"] = max(stats["max_logit"], logits.max().item())
        mean_loss = loss_sum / token_count
        stats[f"{split}_loss"] = mean_loss
        stats[f"{split}_ppl"] = math.exp(mean_loss)
        stats[f"{split}_bpb"] = bpb_nats / math.log(2) / max(byte_count, 1.0)
        stats[f"{split}_acc"] = correct / token_count
    model.train()
    return stats


def save_checkpoint(path, model, optimizers, step, train_reader, config, tokenizer_spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "config": asdict(config),
            "tokenizer": tokenizer_spec,
            "itr": step,
            "train_step": train_reader.step,
            "rng": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path, model, optimizers, train_reader, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    for opt, state in zip(optimizers, ckpt["optimizers"]):
        opt.load_state_dict(state)
    torch.set_rng_state(ckpt["rng"].cpu())
    train_reader.set_step(ckpt["train_step"])
    return int(ckpt["itr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--dataset", choices=["fineweb", "fineweb_edu"], default="fineweb")
    # model
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_kv_head", type=int, default=0)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--norm_position", choices=["pre", "post"], default="pre")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--activation", choices=["gelu", "relu2", "swiglu", "geglu"], default="gelu")
    parser.add_argument("--pos_emb", choices=["learned", "rope", "nope"], default="learned")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--untie_embeddings", action="store_true")
    parser.add_argument("--logit_softcap", type=float, default=0.0)
    parser.add_argument("--depth_scaled_init", action="store_true")
    # optimization
    parser.add_argument("--optimizer", choices=["adamw", "muon", "lion"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--z_loss", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--acc_steps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--scheduler", choices=["wsd", "cos"], default="wsd")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--wsd_fract_decay", type=float, default=0.1)
    parser.add_argument("--wsd_final_lr_scale", type=float, default=0.0)
    parser.add_argument("--decay_type", choices=["linear", "cosine", "exp"], default="linear")
    parser.add_argument("--init_div_factor", type=float, default=100.0)
    # eval, logging, io
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--hellaswag", action="store_true")  # run at the end of training
    parser.add_argument("--hellaswag_limit", type=int, default=0)  # 0 = all 10,042
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--with_replacement", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--data", type=str, default=str(ROOT / "data"))
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "output"))
    parser.add_argument("--ckpt_dir", type=str, default=str(ROOT / "ckpts"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    use_bf16 = args.dtype == "bf16" and device.startswith("cuda")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    train_path, val_path, tokenizer_spec = prepare_data(args.dataset, args.data)
    tokenizer = load_tokenizer(tokenizer_spec)
    model_vocab_size = 64 * math.ceil(tokenizer.vocab_size / 64)  # e.g. GPT-2 50257 -> 50304
    train_reader = DataReader(
        train_path, args.batch_size, args.sequence_length,
        seed=args.data_seed, with_replacement=args.with_replacement,
    )
    eval_sets = {
        split: fixed_batches(path, args.batch_size, args.sequence_length, args.eval_batches)
        for split, path in (("train", train_path), ("val", val_path))
    }
    token_bytes = token_bytes_table(tokenizer, model_vocab_size, device)
    print(f"dataset={args.dataset}, train tokens={train_reader.num_tokens:,}, vocab={tokenizer.vocab_size}")

    config = GPTConfig(
        vocab_size=model_vocab_size,
        sequence_length=args.sequence_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        norm=args.norm,
        norm_position=args.norm_position,
        qk_norm=args.qk_norm,
        activation=args.activation,
        pos_emb=args.pos_emb,
        rope_theta=args.rope_theta,
        tie_embeddings=not args.untie_embeddings,
        logit_softcap=args.logit_softcap,
        depth_scaled_init=args.depth_scaled_init,
    )
    model = GPT(config).to(device)
    n_params, n_non_embedding = model.num_params(), model.num_params(non_embedding=True)
    print(f"params={n_params / 1e6:.3f}M, non-embedding={n_non_embedding / 1e6:.3f}M, device={device}, bf16={use_bf16}")

    optimizers = build_optimizers(model, args.optimizer, args.lr, args.muon_lr, args.weight_decay)
    if args.scheduler == "wsd":
        schedule = wsd_schedule(
            args.iterations,
            final_lr_factor=args.wsd_final_lr_scale,
            n_warmup=args.warmup_steps,
            init_div_factor=args.init_div_factor,
            fract_decay=args.wsd_fract_decay,
            decay_type=args.decay_type,
        )
    else:
        schedule = cosine_schedule(args.iterations, args.warmup_steps, min_factor=args.wsd_final_lr_scale or 0.1)

    ckpt_path = Path(args.ckpt_dir) / f"{args.run_name}/latest.pt"
    log_path = Path(args.out_dir) / f"{args.run_name}/log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(Path(args.resume_from), model, optimizers, train_reader, device)
        print(f"resumed from {args.resume_from} at step {start_step}")
    log_file = log_path.open("a" if args.resume_from else "w")
    if not args.resume_from:
        log_file.write(json.dumps({"args": vars(args), "params": n_params, "non_embedding_params": n_non_embedding}) + "\n")

    tokens_per_step = args.batch_size * args.acc_steps * args.sequence_length
    print(f"steps={args.iterations:,}, tokens/step={tokens_per_step:,}, total tokens={args.iterations * tokens_per_step / 1e9:.2f}B")
    model.train()
    t0 = time.time()
    progress = tqdm(
        range(start_step, args.iterations), initial=start_step, total=args.iterations,
        desc=args.run_name, unit="step", dynamic_ncols=True,
    )
    for step in progress:
        lr_factor = schedule(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_factor

        for _ in range(args.acc_steps):
            x, y = train_reader.sample_batch()
            with autocast():
                _, loss = model(x.to(device), y.to(device), z_loss_weight=args.z_loss)
            (loss / args.acc_steps).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
        ).item()
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        done = step + 1
        if done % args.log_interval == 0:
            train_loss = loss.item()
            tok_per_sec = (done - start_step) * tokens_per_step / (time.time() - t0)
            progress.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr_factor:.3f}x", gn=f"{grad_norm:.2f}", tok_s=f"{tok_per_sec:,.0f}")
            log_file.write(json.dumps({"step": done, "loss": train_loss, "lr_factor": lr_factor, "grad_norm": grad_norm}) + "\n")
        if done % args.ckpt_interval == 0 or done == args.iterations:
            save_checkpoint(ckpt_path, model, optimizers, done, train_reader, config, tokenizer_spec)
        if step % args.eval_interval == 0 or done == args.iterations:
            stats = evaluate(model, eval_sets, token_bytes, autocast, device)
            tokens = (done - start_step) * tokens_per_step
            progress.write(
                f"step={step:<5d} | val_loss: {stats['val_loss']:.4f} | val_bpb: {stats['val_bpb']:.4f} | "
                f"train_loss: {stats['train_loss']:.4f} | max_logit: {stats['max_logit']:.1f} | "
                f"lr_factor: {lr_factor:.3f} | gn: {grad_norm:.2f} | tok/s: {tokens / (time.time() - t0):,.0f}"
            )
            log_file.write(json.dumps({"step": done, "tokens": done * tokens_per_step, "elapsed": time.time() - t0, "eval": stats}) + "\n")
            log_file.flush()

    progress.close()
    model.eval()
    for prompt in SAMPLE_PROMPTS:
        prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
        torch.manual_seed(args.seed)
        with autocast():
            output_ids = model.generate(prompt_ids, max_new_tokens=48, temperature=0.8, valid_vocab_size=tokenizer.vocab_size)
        sample = tokenizer.decode(output_ids[0].tolist(), skip_special=True)
        print(f"sample: {sample!r}")
        log_file.write(json.dumps({"sample": sample}) + "\n")
    model.train()
    if args.hellaswag:
        examples = load_hellaswag(args.data)
        examples = examples[: args.hellaswag_limit] if args.hellaswag_limit else examples
        scores = evaluate_hellaswag(model, tokenizer, examples, device, autocast)
        print(f"hellaswag acc: {scores['hellaswag_acc']:.4f} | acc_norm: {scores['hellaswag_acc_norm']:.4f}")
        log_file.write(json.dumps({"step": args.iterations, "eval": scores}) + "\n")
    log_file.close()
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()