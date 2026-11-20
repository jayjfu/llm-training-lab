# Inference Speed

Greedy decoding of a 01-pretraining checkpoint, one optimization per step. Results: [RESULTS.md](RESULTS.md).

## Steps
| Step | Mode | Idea |
|---|---|---|
| 0 | `naive` | recompute the whole sequence for every token |
| 1 | `kv_cache` | cache keys and values, decode one token at a time |
| 2 | `static_cache` | preallocate the cache to a fixed length |
| 2 | `compile` | `torch.compile` the decode step, possible because shapes are fixed |
| 3 | `cuda_graph` | capture the decode step once and replay it, no per-kernel launch overhead |
| 3 | `compile_graph` | compiled (fused) decode step inside a CUDA graph, as in gpt-fast |

## Run
From `src/`:
```bash
python -m fast_inference.benchmark --update_results                     # all modes, 124m_baseline
python -m fast_inference.benchmark --batch_size 8 --max_new_tokens 512
python -m fast_inference.benchmark --dtype fp32 --modes naive kv_cache compile   # exact-match check
```
- Metrics: time to first token, decode latency per token, decode throughput, peak memory.
- "Match" compares tokens with the first mode. fp32 matches exactly. bf16 can drift when top logits are close.
- `--ckpt` takes any 01-pretraining checkpoint, including the ablation variants.