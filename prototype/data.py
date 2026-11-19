import json
import urllib.request
from dataclasses import dataclass

import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URL, dest)
    return dest


def split_documents(text, min_chars=80):
    blocks = (block.strip() for block in text.split("\n\n"))
    return [block for block in blocks if len(block) >= min_chars]


def write_shards(docs, out_dir, val_frac=0.05):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = int(len(docs) * val_frac)
    paths = {}
    for name, rows in {"train": docs[:-n_val], "val": docs[-n_val:]}.items():
        path = out_dir / f"{name}_000.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for text in rows:
                file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        paths[name] = path
    return paths["train"], paths["val"]


def read_shard(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line)["text"] for line in file if line.strip()]


def prepare_shakespeare(shard_dir):
    text = download_shakespeare(shard_dir / "input.txt").read_text(encoding="utf-8")
    return write_shards(split_documents(text), shard_dir)


@dataclass
class PackStats:
    rows: int = 0
    docs_placed: int = 0
    tokens_cropped: int = 0
    tokens_emitted: int = 0

    @property
    def crop_ratio(self):
        total = self.tokens_emitted + self.tokens_cropped
        return self.tokens_cropped / total if total else 0.0


def tokenize_docs(docs, tokenizer):
    bos = tokenizer.bos_id()
    return [[bos] + tokenizer.encode(text) for text in docs]


# Documents packed using best-fit algorithm to minimize cropping
# When no document fits remaining space, crops a document to fill exactly
def best_fit_pack(docs, seq_len, buffer_size=32):
    stats = PackStats()
    rows = []
    buffer = []
    cursor = 0

    def refill():
        nonlocal cursor
        while len(buffer) < buffer_size and cursor < len(docs):
            buffer.append(docs[cursor])
            cursor += 1

    while True:
        refill()
        if not buffer:
            break
        row = []
        remaining = seq_len
        while remaining > 0 and buffer:
            fits = [index for index, doc in enumerate(buffer) if len(doc) <= remaining]
            if fits:
                doc = buffer.pop(max(fits, key=lambda index: len(buffer[index])))
                row.extend(doc)
                remaining -= len(doc)
            else:
                doc = buffer.pop(min(range(len(buffer)), key=lambda index: len(buffer[index])))
                row.extend(doc[:remaining])
                stats.tokens_cropped += len(doc) - remaining
                remaining = 0
            stats.docs_placed += 1
        if len(row) < seq_len:
            break
        rows.append(row)
        stats.rows += 1
        stats.tokens_emitted += seq_len
    return rows, stats


class PackedLoader:
    def __init__(self, rows, batch_size, device):
        if len(rows) < batch_size:
            raise ValueError(f"need at least {batch_size} packed rows, got {len(rows)}")
        self.rows = rows
        self.batch_size = batch_size
        self.pos = 0
        self.device = device

    def next_batch(self):
        inputs, targets = [], []
        for _ in range(self.batch_size):
            row = self.rows[self.pos % len(self.rows)]
            self.pos += 1
            inputs.append(row[:-1])
            targets.append(row[1:])
        return (
            torch.tensor(inputs, dtype=torch.long, device=self.device),
            torch.tensor(targets, dtype=torch.long, device=self.device),
        )