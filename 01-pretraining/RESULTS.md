# Pretraining Results

## Reference Points
| Model | Data | Tokens | FineWeb val loss | HellaSwag acc_norm |
|---|---|---|---|---|
| GPT-2 124M (OpenAI) | WebText | - | - | ≈29.5% |
| GPT-3 125M | GPT-3 mix | 300B | - | ≈33.7% |
| llm.c GPT-2 124M (modded-nanogpt target) | FineWeb | 10B | ≈3.28 | - |

Our val split is a different 10M tokens of FineWeb, so loss gaps under ~0.02 vs these numbers mean nothing.

## Anchor
- **Tier S LR sweep:** AdamW at 5e-4 / 1e-3 / 2e-3. The best LR is the tier S baseline for the ablations.
- **Noise floor:** the best LR with seeds 0 / 1 / 2. The val-loss spread is the smallest gap we trust.
- **Anchor:** GPT-2 124M, 2.5B tokens, AdamW 6e-4, WSD.

Findings:
- Best tier S LR is 1e-3. 2e-3 is close (+0.016), 5e-4 is clearly worse (+0.161).
- Seed noise at tier S: 0.0084 val loss. Smaller ablation gaps are ties.
- HellaSwag at tier S is near chance (25.4–26.1%), so tier S ablations are judged by val loss.
- GPT-2 124M reaches 3.495 val loss and 27.63% HellaSwag after 2.5B tokens. llm.c reaches ≈3.28 after 10B tokens.

## All Runs
Filled by `python -m pretraining.results --update_results`. Ablations are in [studies/architecture](studies/architecture/RESULTS.md).

<!-- results:start -->
| Run | Non-emb params | Tokens | Val loss | Val bpb | HellaSwag acc_norm | Hours |
|---|---|---|---|---|---|---|
| l_baseline | 85.0M | 2.50B | 3.4950 | 1.1474 | 27.63% | 4.62 |
| s_adamw_lr1e-3 | 25.2M | 0.50B | 3.9705 | 1.2918 | 25.40% | 0.49 |
| s_adamw_lr1e-3_seed1 | 25.2M | 0.50B | 3.9789 | 1.2945 | 26.10% | 0.49 |
| s_adamw_lr1e-3_seed2 | 25.2M | 0.50B | 3.9718 | 1.2922 | 25.84% | 0.50 |
| s_adamw_lr2e-3 | 25.2M | 0.50B | 3.9867 | 1.2970 | 25.78% | 0.51 |
| s_adamw_lr5e-4 | 25.2M | 0.50B | 4.1315 | 1.3442 | 25.42% | 0.49 |
<!-- results:end -->

## Notes
TODO:
- Data: cleaning, mixture ratio
- Training stability
- learning rate, batch size
- attention kernels

Others:
- gated linear units: SwiGLU + ...
- GQA ratio, RoPE theta/base, optional QK-norm, depth-vs-width balance, MoE at large scale??

Priors to test:
- Optimizer: AdamW (0.9, 0.95, 1e-8), Muon > ...
- Gradient clip = 1.0, dropout = 0
- Scheduler: cosine/linear decay, warmup-stable-decay (WSD)
- Pre-norm > post-norm, RMSNorm > LayerNorm
- RoPE > absolute positional embeddings
- SwiGLU > ReLU/GELU FFNs
- MHA > GQA/MQA