import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from sft import load_model, prompt_ids
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent

PAIRS = [
    ("What is 2 plus 2?", "4"),
    ("What is 5 minus 2?", "3"),
    ("What is 10 divided by 2?", "5"),
]


def sample_reply(model, tokenizer, prompt, device, max_new_tokens=64):
    ids = prompt_ids([{"role": "user", "content": prompt}], tokenizer)
    input_ids = torch.tensor([ids], device=device)
    end_id = tokenizer.special_to_id["<|assistant_end|>"]
    logps = []
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids[:, -model.config.sequence_len :])
        logp = F.log_softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(logp.exp(), 1)
        logps.append(logp.gather(1, next_id))
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == end_id:
            break
    return tokenizer.decode(input_ids[0, len(ids) :].tolist(), skip_special=True), logps


def reward(text, gold):
    return 1.0 if gold.lower() in text.lower() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default=str(ROOT / "data/tinyshakespeare/tokenizer.json"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "ckpts/sft.pt"))
    parser.add_argument("--group_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    hits, total = 0.0, 0
    model.train()
    for step, (question, gold) in enumerate(PAIRS * 4):
        # sample a group
        group = [sample_reply(model, tokenizer, question, device) for _ in range(args.group_size)]
        texts = [text for text, _ in group]
        rewards = torch.tensor([reward(text, gold) for text in texts], device=device)

        # group-relative advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # policy gradient loss over the group
        loss = torch.zeros((), device=device)
        for (_, logps), advantage in zip(group, advantages):
            if logps:
                loss = loss - torch.cat(logps).sum() * advantage
        (loss / args.group_size).backward()
        optimizer.step()
        optimizer.zero_grad()

        hits += rewards.sum().item()
        total += args.group_size
        if step % 3 == 0:
            print(f"step={step:<2} | mean_r: {rewards.mean():.2f} | adv: {advantages.tolist()} | sample: {texts[0]!r}")
    print(f"mean reward: {hits / total:.2f}")


if __name__ == "__main__":
    main()