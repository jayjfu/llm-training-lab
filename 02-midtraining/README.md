# 02 - Midtraining

We continue training the pretrained model on a changed data diet before alignment. It starts from the modern baseline in [01-pretraining](../01-pretraining/README.md) and uses the same evaluation.

## Roadmap
- [ ] **Annealing (WIP):** decay the LR on high-quality data (FineWeb-Edu, math, code).
- [ ] **Long context (WIP):** extend context from 1k to 4k+ with RoPE scaling and long documents.
- [ ] **Chat warmup (WIP):** add conversation and multiple-choice formats, as in nanochat, before SFT.

## References
- karpathy/nanochat: [GitHub](https://github.com/karpathy/nanochat)
- OLMo 2, mid-training on curated data (2024): [arXiv](https://arxiv.org/abs/2501.00656)
- Llama 3, annealing on high-quality data (2024): [arXiv](https://arxiv.org/abs/2407.21783)
- MiniCPM, WSD decay with high-quality data (2024): [arXiv](https://arxiv.org/abs/2404.06395)
- Effective Long-Context Scaling of Foundation Models (2023): [arXiv](https://arxiv.org/abs/2309.16039)
- YaRN, RoPE context extension (2023): [arXiv](https://arxiv.org/abs/2309.00071)