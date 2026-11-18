## Results
Scores are on Open LLM Leaderboard v2, with deltas against the base model. `scripts/collect_results.py` generates this table.

<!-- results:start -->
| Stage | Model | IFEval | MuSR | MATH Lvl-5 | BBH | MMLU-Pro | GPQA | Avg |
|---|---|---|---|---|---|---|---|---|
| **Base** | Qwen/Qwen3.5-0.8B-Base | 25.88 | 38.10 | 7.33 | 40.72 | 24.84 | 29.19 | 27.68 |
| **Instruct** | Qwen/Qwen3.5-0.8B | 53.97 (+28.09) | 42.72 (+4.62) | 12.99 (+5.66) | 33.76 (-6.96) | 16.27 (-8.57) | 26.01 (-3.18) | 30.95 (+3.28) |
<!-- results:end -->