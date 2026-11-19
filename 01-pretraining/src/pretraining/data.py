import argparse
import os
from pathlib import Path

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]  # 01-pretraining
FINEWEB = {"fineweb": "HuggingFaceFW/fineweb", "fineweb_edu": "HuggingFaceFW/fineweb-edu"}


def text_batches(stream, docs_per_batch):
    batch = []
    for row in stream:
        batch.append(row["text"])
        if len(batch) == docs_per_batch:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_fineweb(data_dir, dataset="fineweb", num_tokens=2_500_000_000, val_tokens=10_000_000,
                    docs_per_batch=1024, num_threads=16):
    out_dir = Path(data_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {split: out_dir / f"{split}.bin" for split in ("train", "val")}
    if all(path.exists() for path in final_paths.values()):
        return final_paths

    enc = tiktoken.get_encoding("gpt2")
    stream = load_dataset(FINEWEB[dataset], name="sample-10BT", split="train", streaming=True)
    tmp_paths = {split: path.with_suffix(".bin.tmp") for split, path in final_paths.items()}
    files = {split: path.open("wb") for split, path in tmp_paths.items()}
    written = {"train": 0, "val": 0}  # val comes first, from docs never used for training
    with tqdm(total=num_tokens + val_tokens, unit="tok", unit_scale=True, desc=f"tokenizing {dataset}") as bar:
        for texts in text_batches(stream, docs_per_batch):
            chunks = {"train": [], "val": []}
            for ids in enc.encode_ordinary_batch(texts, num_threads=num_threads):
                split = "val" if written["val"] < val_tokens else "train"
                chunks[split].append(np.asarray([enc.eot_token] + ids, dtype=np.uint16))
                written[split] += len(ids) + 1
                bar.update(len(ids) + 1)
            for split, arrays in chunks.items():
                if arrays:
                    np.concatenate(arrays).tofile(files[split])
            if written["train"] >= num_tokens:
                break
    for split, file in files.items():
        file.close()
        tmp_paths[split].rename(final_paths[split])
    print(f"train={written['train']:,} tokens, val={written['val']:,} tokens -> {out_dir}")
    return final_paths


def prepare_data(dataset, data_dir):
    out_dir = Path(data_dir) / dataset
    if not (out_dir / "train.bin").exists():
        raise FileNotFoundError(f"{out_dir}/train.bin not found, run: python -m pretraining.data --dataset {dataset}")
    return out_dir / "train.bin", out_dir / "val.bin", "gpt2"


class DataReader:
    """Random fixed-length windows from a uint16 token file. Without replacement: one shuffled pass per epoch."""

    def __init__(self, data_path, batch_size, sequence_length, seed=0, with_replacement=False, keep_in_ram=False):
        self.data_path = str(data_path)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.step = 0
        self.epoch = -1
        self.order = None
        self.epoch_offset = 0
        tokens = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.num_tokens = len(tokens)
        self.tokens = np.array(tokens) if keep_in_ram else None

    def _tokens(self):
        return self.tokens if self.tokens is not None else np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def set_step(self, step):
        self.step = step
        self.epoch = -1

    def _starts(self, step):
        max_start = self.num_tokens - self.sequence_length - 1
        if self.with_replacement:
            return np.random.default_rng(self.seed + step).integers(0, max_start, size=self.batch_size)

        num_windows = max(self.num_tokens // self.sequence_length, 1)
        steps_per_epoch = max(num_windows // self.batch_size, 1)
        epoch = step // steps_per_epoch
        if epoch != self.epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self.order = rng.permutation(num_windows)
            self.epoch_offset = int(rng.integers(0, self.sequence_length))
            self.epoch = epoch
        first = (step % steps_per_epoch) * self.batch_size
        windows = self.order[first : first + self.batch_size]
        if len(windows) < self.batch_size:
            windows = np.concatenate([windows, self.order[: self.batch_size - len(windows)]])
        return np.clip(windows * self.sequence_length + self.epoch_offset, 0, max_start)

    def sample_batch(self):
        tokens = self._tokens()
        starts = self._starts(self.step)
        self.step += 1
        rows = np.stack([np.asarray(tokens[start : start + self.sequence_length + 1], dtype=np.int64) for start in starts])
        return torch.from_numpy(rows[:, :-1]).contiguous(), torch.from_numpy(rows[:, 1:]).contiguous()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(FINEWEB), default="fineweb")
    parser.add_argument("--num_tokens", type=int, default=2_500_000_000)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--data", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()
    prepare_fineweb(args.data, args.dataset, args.num_tokens, args.val_tokens, num_threads=args.num_threads)
    os._exit(0)  # files are closed and renamed; skip teardown, where the half-read HF stream's threads crash