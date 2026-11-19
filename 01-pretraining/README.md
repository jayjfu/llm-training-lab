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
│   └── results.py        # RESULTS.md tables, loss curves
├── studies/              # one folder per study: run scripts and README
│   ├── architecture/     # one-change ablations of model, optimizer, schedule
│   └── scaling_laws/     # isoFLOP sweep, model size vs tokens
├── scripts/
│   └── hparam_tuning.sh  # LR sweep, seed variance, GPT-2 124M baseline
└── RESULTS.md            # all results: tuning, baseline, studies, all runs
```
Studies reuse the core and only add their own scripts. All results go to `RESULTS.md`. A study that becomes a small switch moves into the core as a flag.

## Roadmap
Goal: the best final model for a fixed compute budget. 25M and 124M times are measured, 50M and 350M are estimates.

| Model | Config | Tokens | Time | Role |
|---|---|---|---|---|
| 25M | 8L × 512 | 0.5B | 0.5 h | ablation screen |
| 50M | 10L × 640 | 1B | ~1.3 h | ablation confirm |
| 124M | 12L × 768 (GPT-2 small) | 2.5B | 4.6 h | baseline |
| 350M | 24L × 1024 (GPT-2 medium) | 7B | ~1.5 days | scale-up |

25M and 50M count non-embedding parameters, 124M and 350M are the GPT-2 names. Micro-batch × accumulation at seq 1024: 25M 16 × 16, 124M 8 × 64, 350M 4 × 128. Run names start with the model size, e.g. `25m_adamw_lr1e-3`, `124m_baseline`.

- [x] [Data Preparation](#data-preparation): FineWeb with the GPT-2 tokenizer.
- [x] [Hyperparameter Tuning](#hyperparameter-tuning): LR sweep, seed variance, GPT-2 124M baseline.
- [ ] [Ablations](#ablations): one change at a time at 25M, winners confirmed at 50M.
- [ ] [Modern Baseline (WIP)](#modern-baseline-wip): all winners combined.
- [ ] [Scaling Laws (WIP)](#scaling-laws-wip): 350M run and an isoFLOP sweep.
- [ ] [Data Ablations (WIP)](#data-ablations-wip): datasets, mixtures and tokenizers.

## Data Preparation
FineWeb sample-10BT with the GPT-2 tokenizer, so losses compare with nanoGPT, llm.c and modded-nanogpt. The first 10M tokens are the val split.
```bash
python -m pretraining.data --dataset fineweb --num_tokens 2_500_000_000  # 7B+ tokens for 350M
```

## Baseline
GPT-2: learned positions, pre-LayerNorm, GELU, tied embeddings, no dropout. AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, WSD schedule.
```bash
python -m pretraining.train --run_name 25m_baseline --n_layer 8 --n_head 8 --n_embd 512 --sequence_length 1024 \
  --batch_size 16 --acc_steps 16 --iterations 1907 --lr 1e-3 --warmup_steps 100 --eval_interval 200 --hellaswag

python -m pretraining.train --run_name 25m_baseline ... --resume_from ../ckpts/25m_baseline/latest.pt  # resume
python -m pretraining.hellaswag --ckpt ../ckpts/25m_baseline/latest.pt                              # eval a checkpoint
python -m pretraining.results --update_results                                                      # refresh RESULTS.md
```
Logs in `output/`, checkpoints in `ckpts/`.

## Hyperparameter Tuning
Sweep the AdamW LR at 25M, measure seed variance, and train the GPT-2 124M baseline that every later run is compared to.
```bash
bash scripts/hparam_tuning.sh lr_sweep      # 25M, lr 5e-4 / 1e-3 / 2e-3
bash scripts/hparam_tuning.sh seeds 1e-3    # best lr, 2 more seeds
bash scripts/hparam_tuning.sh baseline      # GPT-2 124M, 2.5B tokens
```
| Run | Val loss | HellaSwag acc_norm |
|---|---|---|
| 25M, lr 1e-3 (3 seeds) | 3.971–3.979 | 25.4–26.1% |
| GPT-2 124M, 2.5B tokens | 3.495 | 27.63% |

Seed variance at 25M is ~0.01 val loss. All runs: [RESULTS.md](RESULTS.md).

## Ablations
Change one component at a time: optimizer, schedule, norm, activation, positions, attention, embeddings, stability, shape, batch size. Each variant gets its own LR sweep.
```bash
python studies/architecture/ablations.py screen --base_lr 1e-3 --groups optimizer schedule
python studies/architecture/ablations.py screen --base_lr 1e-3 --lr_mults 1 2 \
  --groups norm activation positions attention embeddings stability shape batch
python studies/architecture/ablations.py confirm --base_lr 1e-3 --variants muon swiglu --lr_mults 0.5 1
```
Details: [studies/architecture](studies/architecture/README.md). Results: [RESULTS.md](RESULTS.md#ablations).

## Modern Baseline (WIP)
Combine the ablation winners, re-tune LR and batch size, and compare with the GPT-2 124M baseline.

## Scaling Laws (WIP)
Train the modern baseline at 350M, and run a small isoFLOP sweep for the compute-optimal model size. Details: [studies/scaling_laws](studies/scaling_laws/README.md).

## Data Ablations (WIP)
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