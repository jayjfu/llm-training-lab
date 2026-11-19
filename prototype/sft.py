import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent

IDENTITY = [
    [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am nanogpt."}],
    [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "My name is nanogpt."}],
    [{"role": "user", "content": "Who made you?"}, {"role": "assistant", "content": "I was trained in this repo."}],
    [{"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Hello."}],
]

CHAT = [
    [{"role": "user", "content": "What is 2 plus 2?"}, {"role": "assistant", "content": "4."}],
    [{"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}],
    [{"role": "user", "content": "Is the sun a star?"}, {"role": "assistant", "content": "Yes."}],
    [{"role": "user", "content": "Name a city."}, {"role": "assistant", "content": "Paris."}],
]

MULTIPLE_CHOICE = [
    [{"role": "user", "content": "Capital of France? A) Paris B) Rome"}, {"role": "assistant", "content": "A"}],
    [{"role": "user", "content": "2+2? A) 3 B) 4"}, {"role": "assistant", "content": "B"}],
    [{"role": "user", "content": "Water freezes at? A) 0C B) 100C"}, {"role": "assistant", "content": "A"}],
    [{"role": "user", "content": "Sun is a? A) planet B) star"}, {"role": "assistant", "content": "B"}],
]


# ids: <|bos|><|user_start|>question<|user_end|><|assistant_start|>answer<|assistant_end|>
# mask: 0 on bos, user span, and assistant_start; 1 on answer tokens and <|assistant_end|>
def render(messages, tokenizer):
    ids, mask = [tokenizer.bos_id()], [0]
    for message in messages:
        if message["role"] == "user":
            start, end, supervise = "<|user_start|>", "<|user_end|>", False
        elif message["role"] == "assistant":
            start, end, supervise = "<|assistant_start|>", "<|assistant_end|>", True
        else:
            raise ValueError(f"unknown role {message['role']}")
        body = tokenizer.encode(message["content"])
        ids += [tokenizer.special_to_id[start]] + body + [tokenizer.special_to_id[end]]
        mask += [0] + [int(supervise)] * len(body) + [int(supervise)]
    return ids, mask


def prompt_ids(messages, tokenizer):
    ids, _ = render(messages + [{"role": "assistant", "content": ""}], tokenizer)
    return ids[:-1]  # drop the assistant_end


def padded_rows(conversations, tokenizer, seq_len):
    rows = []
    for ids, mask in (render(messages, tokenizer) for messages in conversations):
        if len(ids) <= seq_len:
            n_pad = seq_len - len(ids)
            rows.append((ids + [tokenizer.bos_id()] * n_pad, mask + [0] * n_pad))
    return rows


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


@torch.no_grad()
def reply(model, tokenizer, prompt, device, max_new_tokens=64):
    ids = prompt_ids([{"role": "user", "content": prompt}], tokenizer)
    input_ids = torch.tensor([ids], device=device)
    end_id = tokenizer.special_to_id["<|assistant_end|>"]
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids[:, -model.config.sequence_len :])
        next_id = int(logits[0, -1].argmax())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == end_id:
            break
    return tokenizer.decode(input_ids[0, len(ids) :].tolist(), skip_special=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tokenizer", type=str, default=str(ROOT / "data/tinyshakespeare/tokenizer.json"))
    parser.add_argument("--base", type=str, default=str(ROOT / "ckpts/base.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "ckpts/sft.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    tokenizer = BPETokenizer.load(args.tokenizer)
    rows = padded_rows(IDENTITY + CHAT + MULTIPLE_CHOICE, tokenizer, args.seq_len)
    print(f"rows={len(rows)}, seq={args.seq_len}")

    model = load_model(args.base, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        ids, mask = rows[step % len(rows)]
        inputs = torch.tensor([ids[:-1]], device=device)
        targets = torch.tensor([ids[1:]], device=device)
        loss_mask = torch.tensor([mask[1:]], dtype=torch.float32, device=device)
        _, loss = model(inputs, targets, loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step={step:<4} | loss: {loss.item():.4f}")

    for question in ["Who are you?", "What is 2 plus 2?", "Capital of France? A) Paris B) Rome"]:
        print(f"Q: {question}")
        print(f"A: {reply(model, tokenizer, question, device)!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(model.config)}, out_path)
    print(f"saved {out_path}, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()