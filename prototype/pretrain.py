import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from data import PackedLoader, best_fit_pack, prepare_shakespeare, read_shard, tokenize_docs
from model import GPT, config_from_depth
from optim import build_optimizer, split_params
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


# Learning rate schedule (linear warmup, constant, linear warmdown)
def lr_multiplier(step, total_steps, warmup=40, warmdown_ratio=0.65, final_frac=0.05):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    warmdown_steps = int(total_steps * warmdown_ratio)
    warmdown_start = total_steps - warmdown_steps
    if step < warmdown_start:
        return 1.0
    progress = (step - warmdown_start) / max(warmdown_steps, 1)
    return 1.0 - progress * (1.0 - final_frac)


def token_nbytes_table(tokenizer):
    table = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
    for token_id, raw in tokenizer.vocab.items():
        table[token_id] = len(raw)
    return table


@torch.no_grad()
def evaluate_bpb(model, batches, nbytes, special_ids):
    special = torch.zeros(nbytes.numel(), dtype=torch.bool, device=nbytes.device)
    if special_ids:
        special[list(special_ids)] = True

    nats = torch.zeros((), device=nbytes.device)
    total_bytes = torch.zeros((), device=nbytes.device)
    for inputs, targets in batches:
        logits, _ = model(inputs, targets)
        per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        keep = ~special[targets.reshape(-1)]  # mask out special tokens
        nats += per_token[keep].sum()
        total_bytes += nbytes[targets.reshape(-1)][keep].sum()

    if total_bytes <= 0:
        return float("inf")
    return (nats / math.log(2) / total_bytes).item()


@torch.no_grad()
def sample(model, tokenizer, device, prompt="ROMEO:"):
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=60, temperature=0.8, top_k=40)
    return tokenizer.decode(output_ids[0].tolist(), skip_special=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--matrix_lr", type=float, default=0.02)
    parser.add_argument("--embed_lr", type=float, default=3e-3)
    parser.add_argument("--adamw_only", action="store_true")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default=str(ROOT / "data/tinyshakespeare/tokenizer.json"))
    parser.add_argument("--data", type=str, default=str(ROOT / "data/tinyshakespeare"))
    parser.add_argument("--out", type=str, default=str(ROOT / "ckpts/base.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    tokenizer = BPETokenizer.load(args.tokenizer)
    train_path, val_path = prepare_shakespeare(Path(args.data))
    train_docs = tokenize_docs(read_shard(train_path), tokenizer)
    val_docs = tokenize_docs(read_shard(val_path), tokenizer)
    train_rows, train_stats = best_fit_pack(train_docs, args.seq_len + 1)
    val_rows, val_stats = best_fit_pack(val_docs, args.seq_len + 1)
    print(
        f"train docs={len(train_docs):,}, rows={train_stats.rows:,}, crop={train_stats.crop_ratio:.2f}\n"
        f"val   docs={len(val_docs):,}, rows={val_stats.rows:,}, crop={val_stats.crop_ratio:.2f}"
    )

    train_loader = PackedLoader(train_rows, args.batch_size, device)
    val_loader = PackedLoader(val_rows, min(args.batch_size, len(val_rows)), device)
    nbytes = token_nbytes_table(tokenizer).to(device)
    special_ids = set(tokenizer.special_to_id.values())

    config = config_from_depth(args.depth, vocab_size=tokenizer.vocab_size, sequence_len=args.seq_len, head_dim=64)
    model = GPT(config).to(device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    print(f"depth={args.depth}, params={model.num_params() / 1e6:.2f}M, seq={args.seq_len}, vocab={tokenizer.vocab_size}")
    matrix, adam = split_params(model)
    print(f"matrix params={sum(param.numel() for param in matrix) / 1e6:.3f}M, adamw params={sum(param.numel() for param in adam) / 1e6:.3f}M")

    if args.adamw_only:
        optimizers = [torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)]
        base_lrs = [args.lr]
    else:
        optimizers = list(build_optimizer(model, matrix_lr=args.matrix_lr, embed_lr=args.embed_lr))
        base_lrs = [args.matrix_lr, args.embed_lr]

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr_factor = lr_multiplier(step, args.steps)
        for opt, base_lr in zip(optimizers, base_lrs):
            for group in opt.param_groups:
                group["lr"] = base_lr * lr_factor

        inputs, targets = train_loader.next_batch()
        _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0 or step == args.steps - 1:
            tok_per_sec = (step + 1) * args.batch_size * args.seq_len / (time.time() - t0)
            print(f"step={step:<4d} | loss: {loss.item():.4f} | lr_factor: {lr_factor:.3f} | tok/s: {tok_per_sec:.0f}")

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            val_bpb = evaluate_bpb(model, [val_loader.next_batch() for _ in range(2)], nbytes, special_ids)
            print(f"  val bpb: {val_bpb:.4f}")
            print("  sample:", sample(model, tokenizer, device).replace("\n", " "))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(config), "step": args.steps}, out_path)
    print(f"saved {out_path}, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()