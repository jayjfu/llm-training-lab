#!/usr/bin/env bash
# Hyperparameter tuning: LR sweep and seed variance at 25M, then the GPT-2 124M baseline.
# usage: bash scripts/hparam_tuning.sh lr_sweep | seeds <best_lr> | baseline
set -euo pipefail
cd "$(dirname "$0")/../src"

MODEL_25M="--n_layer 8 --n_head 8 --n_embd 512 --sequence_length 1024 --batch_size 16 --acc_steps 16 \
  --iterations 1907 --warmup_steps 100 --eval_interval 200 --hellaswag"
MODEL_124M="--n_layer 12 --n_head 12 --n_embd 768 --sequence_length 1024 --batch_size 8 --acc_steps 64 \
  --iterations 4768 --warmup_steps 250 --eval_interval 250 --eval_batches 40 --hellaswag"

case "${1:-}" in
  lr_sweep)
    for lr in 5e-4 1e-3 2e-3; do
      python -m pretraining.train --run_name "25m_adamw_lr${lr}" $MODEL_25M --lr "$lr"
    done
    python -m pretraining.results --runs "25m_adamw_lr*" --plot ;;
  seeds)
    for seed in 1 2; do
      python -m pretraining.train --run_name "25m_adamw_lr${2}_seed${seed}" $MODEL_25M --lr "$2" --seed "$seed" --data_seed "$seed"
    done
    python -m pretraining.results --runs "25m_adamw_lr${2}*" ;;
  baseline)
    python -m pretraining.train --run_name 124m_baseline $MODEL_124M --lr 6e-4
    python -m pretraining.results --runs "124m_*" --plot ;;
  *)
    echo "usage: bash scripts/hparam_tuning.sh lr_sweep | seeds <best_lr> | baseline"
    exit 1 ;;
esac
python -m pretraining.results --update_results