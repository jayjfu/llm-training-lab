# Training Efficiency (WIP)

A speedrun: wall-clock time to 3.50 FineWeb val loss with a 124M model. The 01-pretraining GPT-2 baseline takes 4.6 h.

## Protocol
- Systems changes keep the math: a 100-step benchmark (tok/s, MFU, peak memory) plus a matching loss curve.
- Algorithmic changes need fewer tokens: a full time-to-target run.
- One row per step in `RESULTS.md`: time to 3.50, tok/s, MFU, peak memory, change.

## Ladder
| Step | Techniques | Kind |
|---|---|---|
| 0 | 01-pretraining baseline, profiled | reference |
| 1 | TF32, bf16 weights, fused AdamW, vocab padding, async data loading | systems |
| 2 | `torch.compile`, faster attention backends, FlexAttention | systems |
| 3 | RoPE, QK-norm, ReLU², untied head, zero-init projections, Muon | algorithmic |
| 4 | value embeddings, U-net skips, logit soft-cap, sliding window with window warmup | algorithmic |
| 5 | batches aligned to document starts, batch size and sequence length schedules, tail weight averaging | algorithmic |
| 6 | Polar Express, NorMuon, cautious weight decay | algorithmic |
| 7 | fused cross entropy and ReLU² MLP kernels, bf16 cross entropy | systems |
| 8 | FP8 head and MLP matmuls, CUDA-graph capture of the training step | systems |
| 9 | multi-token prediction, smear module, small bigram hash embedding | algorithmic |

Multi-GPU records (gradient communication overlap, reduce-scatter) are notes only.