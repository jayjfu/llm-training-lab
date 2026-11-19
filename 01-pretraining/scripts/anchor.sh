#!/usr/bin/env bash
# Anchor: tier S LR sweep and seed noise floor, then the tier L anchor.
# usage: bash scripts/anchor.sh sweep | seeds <best_lr> | anchor
set -euo pipefail
cd "$(dirname "$0")/../src"

TIER_S="--n_layer 8 --n_head 8 --n_embd 512 --sequence_length 1024 --batch_size 16 --acc_steps 16 \
  --iterations 1907 --warmup_steps 100 --eval_interval 200 --hellaswag"
TIER_L="--n_layer 12 --n_head 12 --n_embd 768 --sequence_length 1024 --batch_size 8 --acc_steps 64 \
  --iterations 4768 --warmup_steps 250 --eval_interval 250 --eval_batches 40 --hellaswag"

case "${1:-}" in
  sweep)
    for lr in 5e-4 1e-3 2e-3; do
      python -m pretraining.train --run_name "s_adamw_lr${lr}" $TIER_S --lr "$lr"
    done
    python -m pretraining.results --runs "s_adamw_lr*" --plot ;;
  seeds)
    for seed in 1 2; do
      python -m pretraining.train --run_name "s_adamw_lr${2}_seed${seed}" $TIER_S --lr "$2" --seed "$seed" --data_seed "$seed"
    done
    python -m pretraining.results --runs "s_adamw_lr${2}*" ;;
  anchor)
    python -m pretraining.train --run_name l_baseline $TIER_L --lr 6e-4
    python -m pretraining.results --runs "l_*" --plot ;;
  *)
    echo "usage: bash scripts/anchor.sh sweep | seeds <best_lr> | anchor"
    exit 1 ;;
esac
python -m pretraining.results --update_results