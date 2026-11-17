# LLM Training Lab

### A hands-on lab for building, running, and comparing every stage of LLM training: pretraining, midtraining, post-training, evaluation, and optimization, plus a tracker of frontier models.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

Every stage of the LLM pipeline is implemented here as runnable code, then used as a testbed: swap one component, rerun, and measure what changes.

- **Build:** minimal, readable implementations of each stage, from a tokenizer to GRPO.
- **Experiment:** controlled ablations on one component at a time, with results logged next to the code.
- **Compare:** deep dives that contrast methods, papers, and frontier model reports.

### Lab Roadmap
- [x] Prototype: end-to-end mini pipeline (tokenizer → pretrain → SFT → RL → eval)
- [ ] Baselines: a stronger reference setup for each stage, with logged results
- [ ] Ablations: architecture, optimizers, schedules, data mixtures, hyperparameters
- [ ] Comparisons: SFT vs. preference optimization vs. RL vs. distillation; inference and training efficiency techniques
- [ ] Deep research: write-ups that connect experiment results with papers and frontier tech reports

### Prototype
**Model Architecture:** A mini GPT-2 (inspired by nanochat).
- Positions: RoPE
- Norm: RMSNorm
- Attention: MHA or GQA
- MLP: ReLU²
- Embed / head: untied
- Optimizer: Muon on matrices + AdamW on embeddings and lm_head

**Tokenizer:** Regex-pretokenized BPE, byte pairs merged by frequency up to vocab_size.
```bash
python tok_train.py --vocab-size 1024
```

**Document Packing:** Best-fit into a row of exactly seq_len, place the largest document that still fits, and crop the shortest only if nothing fits.

**Training:** Next-token pretrain on documents, --depth as size dial. Muon on matrices, AdamW on the rest.
```bash
python train.py --depth 2 --steps 400 --batch-size 8 --seq-len 128
```

**SFT:** Post-training with loss only on assistant tokens.
```bash
python train_sft.py --steps 100 --seq-len 64
```

**Reinforcement Learning:** Post-training with a minimal GRPO update on a verified reward.
```bash
python rl.py
```

**Simple Eval:** generative (GSM8K) and categorical (MMLU) evaluation, backend by vLLM
```bash
python eval.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --max-problems 50
python eval.py --task mmlu --model Qwen/Qwen2.5-0.5B-Instruct --max-problems 100
```

## Pretraining

### Baseline
WIP: Baseline model + all results in [RESULTS.md](01-pretraining/src/baseline/RESULTS.md)

- Model Architecture: RoPE + pre-norm + RMSNorm + SwiGLU
- Data: cleaning, mixture ratio
- Training: AdamW + scheduler (cosine/WSD)
- Hyperparameter: learning rate, batch size

## Midtraining
TODO:
- Annealing, long-context, domain/reasoning warmup

## Post-training
TODO:
- SFT, preference optimization, RL, distillation

**Results:**  

| Stage          | Model                  | IFEval         | MuSR          | MATH Lvl-5    | BBH           | MMLU-Pro      | GPQA          | Avg           |
|----------------|------------------------|----------------|---------------|---------------|---------------|---------------|---------------|---------------|
| **Base**       | Qwen/Qwen3.5-0.8B-Base | 25.88          | 38.10         | 7.33          | 40.72         | 24.84         | 29.19         | 27.68         |
| **SFT**        |                        | 0              | 0             | 0             | 0             | 0             | 0             | 0             |
| **RL**         |                        | 0              | 0             | 0             | 0             | 0             | 0             | 0             |
| **Preference** |                        | 0              | 0             | 0             | 0             | 0             | 0             | 0             |
| **Instruct**   | Qwen/Qwen3.5-0.8B      | 53.97 (+28.09) | 42.72 (+4.62) | 12.99 (+5.66) | 33.76 (-6.96) | 16.27 (-8.57) | 26.01 (-3.18) | 30.95 (+3.27) |

## Evaluation
Run a full local evaluation suite with Inspect AI and vLLM:
```bash
bash run_eval_suit.sh
```

TODO:
- Scaling laws: [paper-reproduction](04-evaluation/scaling_laws/paper-reproductions.md)
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