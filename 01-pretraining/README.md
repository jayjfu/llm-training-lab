# 01 - Pretraining

We pretrain small GPT models from scratch. We start from a GPT-2 baseline, change one component at a time, and keep what improves the final model.
Note: training efficiency is covered in [05-optimization](../05-optimization).

## Project Structure
```
01-pretraining/
├── src/pretraining/      # shared core, used by every study
│   ├── tokenizer.py      # GPT-2 tokenizer
│   ├── data.py           # FineWeb to uint16 .bin, memmap reader
│   ├── model.py          # GPT-2, every ablation is a flag
│   ├── optim.py          # AdamW, Muon, Lion
│   ├── schedule.py       # cosine, WSD
│   ├── train.py          # training loop
│   ├── hellaswag.py      # HellaSwag eval
│   └── results.py        # RESULTS.md tables, loss plots
├── studies/              # one folder per study: run scripts, README, RESULTS
│   ├── architecture/     # one-change ablations of model, optimizer, schedule
│   └── scaling_laws/     # isoFLOP sweep, model size vs tokens
├── scripts/
│   └── anchor.sh         # LR sweep, seed noise floor, GPT-2 124M anchor
└── RESULTS.md            # anchor results and all runs
```
Studies reuse the core and only add their own scripts. A study that becomes a small switch moves into the core as a flag.

## Roadmap
Goal: the best final model for a fixed compute budget. S and L times are measured, M and XL are estimates.

| Tier | Model | Tokens | Time (est.) | Use |
|---|---|---|---|---|
| S | 8L × 512 (~25M non-emb) | 0.5B | 0.5 h | screen ablations |
| M | 10L × 640 (~50M non-emb) | 1B | ~1.3 h | confirm winners |
| L | 12L × 768 (GPT-2 124M) | 2.5B | 4.6 h | anchor runs |
| XL | 24L × 1024 (GPT-2 350M) | 7B | ~1.5 days | final scale-up |

Micro-batch × accumulation at seq 1024: S 16 × 16, L 8 × 64, XL 4 × 128.

- [x] [Setup](#setup): FineWeb with the GPT-2 tokenizer.
- [x] [Anchor](#anchor): LR sweep, noise floor, GPT-2 124M reference run.
- [ ] [Ablations](#ablations): one change at a time at tier S, winners confirmed at tier M.
- [ ] [Modern baseline (WIP)](#modern-baseline-wip): all winners combined.
- [ ] [Scaling (WIP)](#scaling-wip): bigger model and an isoFLOP sweep.
- [ ] [Data (WIP)](#data-wip): datasets, mixtures and tokenizers.

## Setup
FineWeb sample-10BT with the GPT-2 tokenizer, so losses compare with nanoGPT, llm.c and modded-nanogpt. The first 10M tokens are the val split.
```bash
cd src
pip install torch numpy tqdm datasets tiktoken
python -m pretraining.data --dataset fineweb --num_tokens 2_500_000_000  # 7B+ tokens for tier XL
```

## Baseline
A plain GPT-2: learned positions, pre-LayerNorm, GELU, tied embeddings, no dropout. AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, WSD schedule.
```bash
cd src
python -m pretraining.train --run_name s_baseline --n_layer 8 --n_head 8 --n_embd 512 --sequence_length 1024 \
  --batch_size 16 --acc_steps 16 --iterations 1907 --lr 1e-3 --warmup_steps 100 --eval_interval 200 --hellaswag

python -m pretraining.train --run_name s_baseline ... --resume_from ../ckpts/s_baseline/latest.pt  # resume
python -m pretraining.hellaswag --ckpt ../ckpts/s_baseline/latest.pt                              # eval a checkpoint
python -m pretraining.results --update_results                                                      # refresh RESULTS.md
```
Logs in `output/`, checkpoints in `ckpts/`.

## Anchor
Find the best tier S LR, measure seed noise, and train GPT-2 124M as the reference for every later run.
```bash
bash scripts/anchor.sh sweep          # AdamW lr 5e-4 / 1e-3 / 2e-3 at tier S
bash scripts/anchor.sh seeds 1e-3     # best lr, 2 more seeds
bash scripts/anchor.sh anchor         # GPT-2 124M, 2.5B tokens
```
| Run | Val loss | HellaSwag acc_norm |
|---|---|---|
| Tier S, best lr 1e-3 (3 seeds) | 3.971–3.979 | 25.4–26.1% |
| GPT-2 124M, 2.5B tokens | 3.495 | 27.63% |

Seed noise at tier S is ~0.01 val loss. All runs: [RESULTS.md](RESULTS.md).

## Ablations
Change one component at a time: optimizer, schedule, norm, activation, positions, attention, embeddings, stability, shape, batch size. Each variant gets its own LR sweep.
```bash
python studies/architecture/ablations.py screen --base_lr 1e-3 --groups optimizer schedule
python studies/architecture/ablations.py screen --base_lr 1e-3 --lr_mults 1 2 \
  --groups norm activation positions attention embeddings stability shape batch
python studies/architecture/ablations.py confirm --base_lr 1e-3 --variants muon swiglu --lr_mults 0.5 1
```
Details: [studies/architecture](studies/architecture/README.md). Results: [studies/architecture/RESULTS.md](studies/architecture/RESULTS.md).

## Modern Baseline (WIP)
Combine the ablation winners, re-tune LR and batch size, and rerun at tier L against the anchor.

## Scaling (WIP)
Train the modern baseline at tier XL, and run a small isoFLOP sweep to find the compute-optimal size. Details: [studies/scaling_laws](studies/scaling_laws/README.md).

## Data (WIP)
Compare FineWeb, FineWeb-Edu and mixes with code and math. Also tokenizer size, and best-fit packing vs plain concatenation.

## Evaluation
1. **Val loss and bpb** on a fixed 10M-token FineWeb val split.
2. **HellaSwag** acc_norm on 10,042 val examples. GPT-2 124M ≈ 29.5%, GPT-3 125M ≈ 33.7%.
3. **CORE-lite (WIP):** HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande, LAMBADA, chance-centered accuracy.
4. **Stability:** grad norm, loss spikes, max logit.

## References
**Code**
- karpathy/nanochat: [GitHub](https://github.com/karpathy/nanochat)
- KellerJordan/modded-nanogpt: [GitHub](https://github.com/KellerJordan/modded-nanogpt)
- epfml/llm-baselines: [GitHub](https://github.com/epfml/llm-baselines)

**Model**
- NoPE, The Impact of Positional Encoding on Length Generalization (2023): [arXiv](https://arxiv.org/abs/2305.19466)
- GQA (2023): [arXiv](https://arxiv.org/abs/2305.13245)
- ReLU², Primer (2021): [arXiv](https://arxiv.org/abs/2109.08668)
- RoPE, RoFormer (2021): [arXiv](https://arxiv.org/abs/2104.09864)
- QK-norm, Query-Key Normalization for Transformers (2020): [arXiv](https://arxiv.org/abs/2010.04245)
- SwiGLU / GeGLU, GLU Variants Improve Transformer (2020): [arXiv](https://arxiv.org/abs/2002.05202)
- Pre-norm vs post-norm, On Layer Normalization in the Transformer Architecture (2020): [arXiv](https://arxiv.org/abs/2002.04745)
- MQA, Fast Transformer Decoding: One Write-Head is All You Need (2019): [arXiv](https://arxiv.org/abs/1911.02150)
- RMSNorm, Root Mean Square Layer Normalization (2019): [arXiv](https://arxiv.org/abs/1910.07467)
- GPT-2, Language Models are Unsupervised Multitask Learners (2019): [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Transformer, Attention Is All You Need (2017): [arXiv](https://arxiv.org/abs/1706.03762)
- Tied embeddings, Using the Output Embedding to Improve Language Models (2016): [arXiv](https://arxiv.org/abs/1608.05859)

**Optimization and stability**
- Muon (2024): [Blog](https://kellerjordan.github.io/posts/muon/), Muon is Scalable for LLM Training (2025): [arXiv](https://arxiv.org/abs/2502.16982)
- Logit soft-capping, Gemma 2 (2024): [arXiv](https://arxiv.org/abs/2408.00118)
- WSD, Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (2024): [arXiv](https://arxiv.org/abs/2405.18392)
- Lion, Symbolic Discovery of Optimization Algorithms (2023): [arXiv](https://arxiv.org/abs/2302.06675)
- z-loss, PaLM (2022): [arXiv](https://arxiv.org/abs/2204.02311)
- AdamW, Decoupled Weight Decay Regularization (2017): [arXiv](https://arxiv.org/abs/1711.05101)

**Data, scaling and evaluation**
- FineWeb (2024): [arXiv](https://arxiv.org/abs/2406.17557)
- CORE metric, DataComp-LM (DCLM) (2024): [arXiv](https://arxiv.org/abs/2406.11794)
- Chinchilla, Training Compute-Optimal Large Language Models (2022): [arXiv](https://arxiv.org/abs/2203.15556)
- Scaling Laws for Neural Language Models (2020): [arXiv](https://arxiv.org/abs/2001.08361)
- HellaSwag (2019): [arXiv](https://arxiv.org/abs/1905.07830)
- BPE, Neural Machine Translation of Rare Words with Subword Units (2015): [arXiv](https://arxiv.org/abs/1508.07909)