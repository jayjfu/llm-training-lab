# Inference Results

Filled by `python -m fast_inference.benchmark --update_results`, one table per batch size.

## Batch 1
<!-- batch1:start -->
124m_baseline, bf16, batch 1, prompt 128, 256 new tokens, NVIDIA GeForce RTX 5090. Median of 3 runs after warmup.

| Mode | TTFT (ms) | Decode (ms/token) | Decode (tokens/s) | Peak memory (MB) | Match vs first mode |
|---|---|---|---|---|---|
| naive | 1.0 | 1.12 | 892 | 314 | 100.0% |
| kv_cache | 1.1 | 1.03 | 970 | 293 | 100.0% |
| static_cache | 1.4 | 1.23 | 816 | 302 | 100.0% |
| compile | 1.7 | 0.83 | 1,206 | 302 | 100.0% |
| cuda_graph | 1.5 | 0.75 | 1,338 | 462 | 100.0% |
| compile_graph | 1.6 | 0.53 | 1,885 | 430 | 100.0% |
<!-- batch1:end -->

## Batch 64
<!-- batch64:start -->
124m_baseline, bf16, batch 64, prompt 256, 512 new tokens, NVIDIA GeForce RTX 5090. Median of 3 runs after warmup.

| Mode | TTFT (ms) | Decode (ms/token) | Decode (tokens/s) | Peak memory (MB) | Match vs first mode |
|---|---|---|---|---|---|
| naive | 16.9 | 37.68 | 1,698 | 1,069 | 100.0% |
| kv_cache | 20.0 | 3.54 | 18,094 | 2,150 | 100.0% |
| compile_graph | 25.5 | 2.57 | 24,905 | 4,244 | 100.0% |
<!-- batch64:end -->

## Findings
- Batch 1 is launch-bound: one step reads ~250 MB of weights (~0.15 ms), the rest is launching a few hundred small kernels. `compile_graph` (fused kernels in a CUDA graph) gives 2.1x over naive.
- Batch 64 is compute-bound: naive recompute grows with the square of the length. KV cache gives 10.6x, `compile_graph` 14.7x.
- Static cache alone is slower: it attends over the full preallocated length with a mask. It exists to make shapes fixed for compile and graphs.
- The static cache preallocates the full length (~1.8 GB of KV at batch 64, length 768), plus the CUDA-graph memory pool.
- Generation computes logits for the last position only. Full-sequence logits at batch 64 need ~2 GB per step.
- Benchmark on an idle GPU. The batch 64 run shared the GPU with another job.