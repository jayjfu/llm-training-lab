
## Results
We compare the base model, the official instruct model and our SFT variants on six leaderboard tasks: IFEval, MuSR, MATH 
Level-5, BBH, MMLU-Pro, and GPQA. 

| Stage        | Model                              | IFEval         | MuSR          | MATH Lvl-5    | BBH           | MMLU-Pro      | GPQA          | Avg           |
|--------------|------------------------------------|----------------|---------------|---------------|---------------|---------------|---------------|---------------|
| **Base**     | Qwen/Qwen3.5-0.8B-Base             | 25.88          | 38.10         | 7.33          | 40.72         | 24.84         | 29.19         | 27.68         |
| **SFT**      | sft_lr2e5_bs32_acc1_pack<br>0<br>0 | 20.52 (-5.36)<br>0<br>0 | 40.61 (+2.51)<br>0<br>0 | 4.31 (-3.02)<br>0<br>0 | 38.43 (-2.29)<br>0<br>0 | 17.79 (-7.05)<br>0<br>0 | 26.34 (-2.85)<br>0<br>0 | 24.67 (-3.01)<br>0<br>0 |
| **Instruct** | Qwen/Qwen3.5-0.8B                  | 53.97 (+28.09) | 42.72 (+4.62) | 12.99 (+5.66) | 33.76 (-6.96) | 16.27 (-8.57) | 26.01 (-3.18) | 30.95 (+3.27) |
<!-- Disabled rows:
 | **SFT**      | sft_lr2e5_bs32_acc1_pack | 20.52 (-5.36) | 40.61 (+2.51) | 4.31 (-3.02) | 38.43 (-2.29) | 17.79 (-7.05) | 26.34 (-2.85) | 24.67 (-3.01) |
-->