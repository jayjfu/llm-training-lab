## Results
Scores are on Open LLM Leaderboard v2, with deltas against the base model. `scripts/collect_results.py` generates this table.

<!-- results:start -->
| Stage | Model | IFEval | MuSR | MATH Lvl-5 | BBH | MMLU-Pro | GPQA | Avg |
|---|---|---|---|---|---|---|---|---|
| **Base** | Qwen/Qwen3.5-0.8B-Base | 32.54 | 38.10 | 7.33 | 40.72 | 24.85 | 29.19 | 28.79 |
| **SFT** | sft_full_lr2e5 | 54.29 (+21.74) | 40.87 (+2.78) | 5.06 (-2.27) | 38.67 (-2.05) | 21.58 (-3.27) | 25.08 (-4.11) | 30.93 (+2.14) |
| **Instruct** | Qwen/Qwen3.5-0.8B | 57.87 (+25.33) | 42.59 (+4.50) | 12.99 (+5.66) | 33.80 (-6.93) | 16.27 (-8.58) | 26.01 (-3.19) | 31.59 (+2.80) |
<!-- results:end -->

Coding benchmarks, pass@1 (greedy):

<!-- coding:start -->
| Stage | Model | HumanEval | HumanEval+ | MBPP | MBPP+ | Avg |
|---|---|---|---|---|---|---|
<!-- coding:end -->