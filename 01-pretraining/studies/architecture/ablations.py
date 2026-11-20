"""One-change ablations at 25M, then confirm the winners at 50M.

python studies/architecture/ablations.py screen --base_lr 1e-3     # all groups, lr x {0.5, 1, 2}
python studies/architecture/ablations.py screen --base_lr 1e-3 --groups optimizer schedule --lr_mults 1 2
python studies/architecture/ablations.py confirm --base_lr 6e-4 --variants muon swiglu --lr_mults 0.5 1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 01-pretraining
SRC_DIR = ROOT / "src"

MODEL_CONFIGS = {  # key = run-name prefix
    "25m": "--n_layer 8 --n_head 8 --n_embd 512 --sequence_length 1024 --batch_size 16 --acc_steps 16 "
         "--iterations 1907 --warmup_steps 100 --eval_interval 200 --hellaswag",
    "50m": "--n_layer 10 --n_head 10 --n_embd 640 --sequence_length 1024 --batch_size 8 --acc_steps 64 "
         "--iterations 1907 --warmup_steps 100 --eval_interval 200 --hellaswag",
}

# group -> variant -> extra flags. "lr_flag" sets which LR the sweep moves, "lr_scale" its center vs base_lr.
VARIANTS = {
    "optimizer": {
        "muon": {"flags": "--optimizer muon", "lr_flag": "--muon_lr", "lr_center": 0.02},
        "lion": {"flags": "--optimizer lion --weight_decay 1.0", "lr_scale": 0.2},
    },
    "schedule": {
        "cosine": {"flags": "--scheduler cos"},
        "wsd_decay20": {"flags": "--wsd_fract_decay 0.2"},
        "wsd_decay30": {"flags": "--wsd_fract_decay 0.3"},
    },
    "norm": {
        "rmsnorm": {"flags": "--norm rmsnorm"},
        "post_norm": {"flags": "--norm_position post"},
        "qk_norm": {"flags": "--qk_norm"},
    },
    "activation": {
        "relu2": {"flags": "--activation relu2"},
        "swiglu": {"flags": "--activation swiglu"},
        "geglu": {"flags": "--activation geglu"},
    },
    "positions": {
        "rope": {"flags": "--pos_emb rope"},
        "rope_theta500k": {"flags": "--pos_emb rope --rope_theta 500000"},
        "nope": {"flags": "--pos_emb nope"},
    },
    "attention": {
        "gqa4": {"flags": "--n_kv_head 2"},
        "mqa": {"flags": "--n_kv_head 1"},
    },
    "embeddings": {
        "untied": {"flags": "--untie_embeddings"},
    },
    "stability": {
        "z_loss": {"flags": "--z_loss 1e-4"},
        "softcap30": {"flags": "--logit_softcap 30"},
        "depth_init": {"flags": "--depth_scaled_init"},
    },
    "shape": {  # 25M only, ~same non-embedding params as 8L x 512
        "shape_4x704": {"flags": "--n_layer 4 --n_head 11 --n_embd 704"},
        "shape_6x576": {"flags": "--n_layer 6 --n_head 9 --n_embd 576"},
        "shape_12x448": {"flags": "--n_layer 12 --n_head 7 --n_embd 448"},
        "shape_16x384": {"flags": "--n_layer 16 --n_head 6 --n_embd 384"},
    },
    "batch": {  # 25M only, same 0.5B tokens
        "batch_131k": {"flags": "--acc_steps 8 --iterations 3814"},
        "batch_524k": {"flags": "--acc_steps 32 --iterations 954"},
    },
}


def lr_name(value):  # 0.0005 -> "5e-4"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def is_done(run_name):
    log_path = ROOT / f"output/{run_name}/log.jsonl"
    if not log_path.exists():
        return False
    with log_path.open() as file:
        return any("hellaswag_acc" in json.loads(line).get("eval", {}) for line in file)


def run(run_name, flags, dry_run):
    if is_done(run_name):
        print(f"skip {run_name} (done)")
        return
    command = [sys.executable, "-m", "pretraining.train", "--run_name", run_name, *flags.split()]
    print(" ".join(command[1:]))
    if not dry_run:
        env = {name: value for name, value in os.environ.items() if name != "MKL_THREADING_LAYER"}  # conda MKL vs libgomp
        subprocess.run(command, cwd=SRC_DIR, check=True, env=env)


def sweep(size_prefix, variant_name, spec, base_lr, lr_mults, dry_run):
    lr_flag = spec.get("lr_flag", "--lr")
    center = spec.get("lr_center", base_lr * spec.get("lr_scale", 1.0))
    for mult in lr_mults:
        lr = center * mult
        extra = f"--lr {base_lr} {lr_flag} {lr}" if lr_flag != "--lr" else f"--lr {lr}"
        run(f"{size_prefix}_{variant_name}_lr{lr_name(lr)}", f"{MODEL_CONFIGS[size_prefix]} {spec['flags']} {extra}", dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["screen", "confirm"])
    parser.add_argument("--base_lr", type=float, required=True)  # best AdamW LR of the baseline at this model size
    parser.add_argument("--groups", nargs="+", default=list(VARIANTS))
    parser.add_argument("--variants", nargs="+", default=[])  # confirm mode: variant names
    parser.add_argument("--lr_mults", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    specs = {name: spec for group in VARIANTS.values() for name, spec in group.items()}
    if args.mode == "screen":
        for group in args.groups:
            for variant_name, spec in VARIANTS[group].items():
                sweep("25m", variant_name, spec, args.base_lr, args.lr_mults, args.dry_run)
    else:
        sweep("50m", "adamw", {"flags": ""}, args.base_lr, args.lr_mults, args.dry_run)
        for variant_name in args.variants:
            sweep("50m", variant_name, specs[variant_name], args.base_lr, args.lr_mults, args.dry_run)


if __name__ == "__main__":
    main()