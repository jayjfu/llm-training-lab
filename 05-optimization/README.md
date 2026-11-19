## Introduction
TODO: inference throughout optimization (tokens/s) + training efficiency

1. inference optimization: 
  - kv cache , pagedAttention, quantization, flashAttention, GQA...
2. training efficiency:
  - mixed precision, torch.compile, fused kernels, activation checkpointing, distributed training...

## References

### Training Efficiency
**Code**
- KellerJordan/modded-nanogpt: [GitHub](https://github.com/KellerJordan/modded-nanogpt)
- pytorch/torchtitan: [GitHub](https://github.com/pytorch/torchtitan), TorchTitan (2024): [arXiv](https://arxiv.org/abs/2410.06511)
- The Ultra-Scale Playbook (2025): [Hugging Face](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

**Kernels and precision**
- Liger Kernel, fused Triton kernels for LLM training (2024): [arXiv](https://arxiv.org/abs/2410.10989)
- FlashAttention-3 (2024): [arXiv](https://arxiv.org/abs/2407.08608)
- FlashAttention-2 (2023): [arXiv](https://arxiv.org/abs/2307.08691)
- FP8 Formats for Deep Learning (2022): [arXiv](https://arxiv.org/abs/2209.05433)
- FlashAttention (2022): [arXiv](https://arxiv.org/abs/2205.14135)
- Mixed Precision Training (2017): [arXiv](https://arxiv.org/abs/1710.03740)

**Memory and parallelism**
- MFU, defined in PaLM (2022): [arXiv](https://arxiv.org/abs/2204.02311)
- Megatron-LM, efficient large-scale training on GPU clusters (2021): [arXiv](https://arxiv.org/abs/2104.04473)
- ZeRO, Memory Optimizations Toward Training Trillion Parameter Models (2019): [arXiv](https://arxiv.org/abs/1910.02054)
- Megatron-LM, tensor parallelism (2019): [arXiv](https://arxiv.org/abs/1909.08053)
- Activation checkpointing, Training Deep Nets with Sublinear Memory Cost (2016): [arXiv](https://arxiv.org/abs/1604.06174)

### Inference Speed
**Code**
- vllm-project/vllm: [GitHub](https://github.com/vllm-project/vllm)
- sgl-project/sglang: [GitHub](https://github.com/sgl-project/sglang)

**Attention and KV cache**
- MLA, DeepSeek-V2 (2024): [arXiv](https://arxiv.org/abs/2405.04434)
- Flash-Decoding (2023): [Blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- GQA (2023): [arXiv](https://arxiv.org/abs/2305.13245)
- KV cache and partitioning, Efficiently Scaling Transformer Inference (2022): [arXiv](https://arxiv.org/abs/2211.05102)
- MQA, Fast Transformer Decoding: One Write-Head is All You Need (2019): [arXiv](https://arxiv.org/abs/1911.02150)

**Serving and batching**
- RadixAttention, SGLang (2023): [arXiv](https://arxiv.org/abs/2312.07104)
- PagedAttention, vLLM (2023): [arXiv](https://arxiv.org/abs/2309.06180)
- Continuous batching, Orca (2022): [OSDI](https://www.usenix.org/conference/osdi22/presentation/yu)

**Speculative decoding**
- EAGLE (2024): [arXiv](https://arxiv.org/abs/2401.15077)
- Medusa (2024): [arXiv](https://arxiv.org/abs/2401.10774)
- Accelerating LLM Decoding with Speculative Sampling (2023): [arXiv](https://arxiv.org/abs/2302.01318)
- Fast Inference from Transformers via Speculative Decoding (2022): [arXiv](https://arxiv.org/abs/2211.17192)

**Quantization**
- AWQ (2023): [arXiv](https://arxiv.org/abs/2306.00978)
- SmoothQuant (2022): [arXiv](https://arxiv.org/abs/2211.10438)
- GPTQ (2022): [arXiv](https://arxiv.org/abs/2210.17323)
- LLM.int8() (2022): [arXiv](https://arxiv.org/abs/2208.07339)