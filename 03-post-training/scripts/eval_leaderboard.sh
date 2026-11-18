#!/bin/bash
# Usage: bash scripts/eval_leaderboard.sh [run_name ...]
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
INSTRUCT_MODEL="Qwen/Qwen3.5-0.8B"
TASKS=(
  leaderboard_ifeval
  leaderboard_gpqa
  leaderboard_mmlu_pro
  leaderboard_musr
  leaderboard_bbh
  leaderboard_math_hard
)

runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
  runs=(base instruct)
  for final_dir in ckpts/*/final; do
    [[ -d "${final_dir}" ]] && runs+=("$(basename "$(dirname "${final_dir}")")")
  done
fi

for run in "${runs[@]}"; do
  case "${run}" in
    base) model_path="${BASE_MODEL}"; apply_chat=false ;;
    instruct) model_path="${INSTRUCT_MODEL}"; apply_chat=true ;;
    *)
      model_path="ckpts/${run}/final"
      apply_chat=true
      grep -q "^apply_chat_template: false" "ckpts/${run}/run.yaml" && apply_chat=false
      ;;
  esac

  for task in "${TASKS[@]}"; do
    out_dir="results/${run}/${task}"
    if find "${out_dir}" -name "results_*.json" 2>/dev/null | grep -q .; then
      echo "skip ${run}/${task}"
      continue
    fi
    cmd=(lm-eval run --model vllm --model_args "pretrained=${model_path},dtype=auto"
         --tasks "${task}" --batch_size auto --output_path "${out_dir}")
    [[ "${apply_chat}" == "true" ]] && cmd+=(--apply_chat_template --fewshot_as_multiturn)
    "${cmd[@]}"
  done
done

python scripts/collect_results.py