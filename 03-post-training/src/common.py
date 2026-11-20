import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[2]
CKPTS_DIR = ROOT / "03-post-training/ckpts"
DATA_DIR = ROOT / "03-post-training/data"


def merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path):
    config = yaml.safe_load(path.read_text())
    parent = config.pop("defaults", None)
    return merge(read_yaml(path.parent / parent), config) if parent else config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint-* of each run")
    args = parser.parse_args()
    configs = []
    for config_path in map(Path, args.configs):
        if config_path.name.startswith("_"):
            continue
        config = read_yaml(config_path)
        config.setdefault("run_name", config_path.stem)
        if config["model"].startswith("ckpts/"):
            model_path = CKPTS_DIR.parent / config["model"]
            if not model_path.exists():
                raise SystemExit(f"{model_path} not found; train that run first")
            config["model"] = str(model_path)
        configs.append(config)
    args.configs = configs
    return args


def make_run_dir(config):
    run_dir = CKPTS_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return run_dir


def pairs_path(config):
    return DATA_DIR / f"{config['pairs']['name']}.jsonl"


def load_model(name, dtype=torch.float32):
    return AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)


def save_final(model, processor, run_dir):
    final_dir = run_dir / "final"
    model.to(torch.bfloat16).save_pretrained(final_dir)
    processor.save_pretrained(final_dir)


def completion_logprobs(model, sequences, attention_mask, completion_length):
    with torch.autocast(sequences.device.type, dtype=torch.bfloat16):
        logits = model(input_ids=sequences, attention_mask=attention_mask, logits_to_keep=completion_length + 1).logits
    targets = sequences[:, -completion_length:]
    # one row at a time in float32: never holds a float32 copy of the whole [batch, length, vocab] logits
    return torch.stack([
        row_logits[:-1].float().log_softmax(dim=-1).gather(-1, row_targets.unsqueeze(-1)).squeeze(-1)
        for row_logits, row_targets in zip(logits, targets)
    ])