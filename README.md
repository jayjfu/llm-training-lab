# LLM Training Lab

### A hands-on lab for building, running, and comparing every stage of LLM training: pretraining, midtraining, post-training, evaluation, and optimization, plus a tracker of frontier models.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

Every stage of the LLM pipeline is implemented here as runnable code, then used as a testbed: swap one component, rerun, and measure what changes.

- **Build:** minimal, readable implementations of each stage, from a tokenizer to GRPO.
- **Experiment:** controlled ablations on one component at a time, with results logged next to the code.
- **Compare:** deep dives that contrast methods, papers, and frontier model reports.

### Lab Roadmap
- [x] Prototype: end-to-end mini pipeline (tokenizer → pretrain → SFT → RL)
- [ ] Baselines: a stronger reference setup for each stage, with logged results
- [ ] Ablations: architecture, optimizers, schedules, data mixtures, hyperparameters
- [ ] Comparisons: SFT vs. preference optimization vs. RL vs. distillation; inference and training efficiency techniques
- [ ] Deep research: write-ups that connect experiment results with papers and frontier tech reports

### Prototype
A toy end-to-end pipeline (tokenizer → pretrain → SFT → RL) that runs on a CPU in minutes. Details in [prototype](prototype/README.md).

## Pretraining
WIP: We pretrain small GPT models from scratch on FineWeb, starting from a GPT-2 baseline and changing one component at a time. Details are in [01-pretraining](01-pretraining/README.md), and all runs are in [RESULTS.md](01-pretraining/RESULTS.md).

- Anchor: tier S LR sweep, seed noise floor, and a GPT-2 124M reference run.
- Ablations: optimizer, schedule, norm, activation, positions, attention, embeddings, stability, shape, batch size.
- Next: modern baseline from the winners, scaling (isoFLOP sweep), data mixtures.

**Results:** GPT-2 124M anchor after 2.5B tokens.

| Model | Tokens | Val loss | HellaSwag acc_norm |
|---|---|---|---|
| GPT-2 124M (ours) | 2.5B | 3.495 | 27.63% |
| GPT-2 124M (OpenAI) | - | - | ≈29.5% |
| llm.c GPT-2 124M | 10B | ≈3.28 | - |

## Midtraining
TODO:
- Annealing, long-context, domain/reasoning warmup

## Post-training
WIP: We turn Qwen3.5-0.8B-Base into an instruct model. Details are in [03-post-training](03-post-training/README.md), and all runs are in [RESULTS.md](03-post-training/RESULTS.md).

- SFT: We fine-tune on smol-smoltalk with full fine-tuning and LoRA.
- Preference: We train DPO on MATH pairs sampled from the SFT model.
- RL: We train GRPO on MATH and compare the `grpo`, `dapo` and `dr_grpo` losses.
- Distillation: TODO.

DPO and GRPO each have a TRL version and a from-scratch version.

**Results:** The table shows the best run per stage.

<!-- results:start -->
| Stage | Model | IFEval | MuSR | MATH Lvl-5 | BBH | MMLU-Pro | GPQA | Avg |
|---|---|---|---|---|---|---|---|---|
| **Base** | Qwen/Qwen3.5-0.8B-Base | 25.88 | 38.10 | 7.33 | 40.72 | 24.84 | 29.19 | 27.68 |
| **Instruct** | Qwen/Qwen3.5-0.8B | 53.97 (+28.09) | 42.72 (+4.62) | 12.99 (+5.66) | 33.76 (-6.96) | 16.27 (-8.57) | 26.01 (-3.18) | 30.95 (+3.28) |
<!-- results:end -->

## Evaluation
Run a full local evaluation suite with Inspect AI and vLLM, or a quick GSM8K / MMLU check:
```bash
bash run_eval_suit.sh
python simple_eval.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --max_problems 50
```

TODO:
- benchmarks, inference throughout

## Optimization
TODO:
1. inference optimization: kv cache , pagedAttention, quantization, flashAttention, GQA, ...
2. training efficiency: ...

## Frontier Tracker
Tech reports / blogs / papers

Families: kimi/, qwen/deepseek/, glm/, llama/, gemini/, claude/, gpt/, ...

## Others
Tokenizers, distributed training...

## Reference & Acknowledgement:

- epfml/llm-baselines: [GitHub](https://github.com/epfml/llm-baselines)
- labmlai/annotated_deep_learning_paper_implementations: [GitHub](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- karpathy/nanochat: [GitHub](https://github.com/karpathy/nanochat)
- allenai/OLMo-core: [GitHub](https://github.com/allenai/OLMo-core/)