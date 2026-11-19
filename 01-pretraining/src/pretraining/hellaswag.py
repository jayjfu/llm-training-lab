import argparse
import json
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT, GPTConfig
from tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[2]  # 01-pretraining
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def load_hellaswag(data_dir):
    path = Path(data_dir) / "hellaswag/val.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HELLASWAG_URL, path)
        except OSError:
            path.unlink(missing_ok=True)
            from datasets import load_dataset
            rows = load_dataset("Rowan/hellaswag", split="validation")
            with path.open("w") as file:
                for row in rows:
                    file.write(json.dumps({"ctx": row["ctx"], "endings": row["endings"], "label": int(row["label"])}) + "\n")
    with path.open() as file:
        return [json.loads(line) for line in file]


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, examples, device, autocast=nullcontext):
    model.eval()
    max_len = model.config.sequence_length
    correct, correct_norm = 0, 0
    for example in tqdm(examples, desc="hellaswag", unit="ex", leave=False, dynamic_ncols=True):
        context = tokenizer.encode(example["ctx"])
        endings = [tokenizer.encode(" " + ending) for ending in example["endings"]]
        rows = [(context + ending)[-max_len:] for ending in endings]
        width = max(len(row) for row in rows)
        tokens = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width))
        for index, (row, ending) in enumerate(zip(rows, endings)):
            tokens[index, : len(row)] = torch.tensor(row)
            mask[index, len(row) - len(ending) : len(row)] = 1
        tokens, mask = tokens.to(device), mask.to(device)
        with autocast():
            logits, _ = model(tokens)
        per_token = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1), reduction="none"
        ).view(len(rows), -1)
        ending_mask = mask[:, 1:]
        total_loss = (per_token * ending_mask).sum(dim=1)
        mean_loss = total_loss / ending_mask.sum(dim=1)
        correct += int(total_loss.argmin().item() == example["label"])
        correct_norm += int(mean_loss.argmin().item() == example["label"])
    model.train()
    return {"hellaswag_acc": correct / len(examples), "hellaswag_acc_norm": correct_norm / len(examples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)  # 0 = all 10,042
    parser.add_argument("--data", type=str, default=str(ROOT / "data"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    tokenizer = load_tokenizer(ckpt["tokenizer"])
    examples = load_hellaswag(args.data)
    examples = examples[: args.limit] if args.limit else examples
    use_bf16 = args.device.startswith("cuda")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    print(json.dumps(evaluate_hellaswag(model, tokenizer, examples, args.device, autocast)))


if __name__ == "__main__":
    main()