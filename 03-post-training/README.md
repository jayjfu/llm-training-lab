## Introduction
This part turns Qwen3.5-0.8B-Base into an instruct model with SFT, DPO, GRPO and distillation. Each run is one config, and every run is evaluated on the same benchmarks.

TODO: The roadmap has three stages.
1. Build each method with the Hugging Face stack (TRL, PEFT and Unsloth).
2. Reimplement each method from scratch without TRL.
3. Explore Megatron-style parallelism and efficiency.

## Project Structure
```
03-post-training/
├── configs/<stage>/<run_name>.yaml
├── src/
│   ├── common.py
│   ├── data.py
│   ├── rewards.py
│   ├── sft_trl.py
│   ├── grpo_trl.py
│   ├── grpo.py
│   ├── make_pairs.py
│   ├── dpo_trl.py
│   └── dpo.py
└── scripts/
    ├── eval_leaderboard.sh
    └── collect_results.py
```

## SFT
We fine-tune on smol-smoltalk with full fine-tuning and LoRA.
```bash
python src/sft_trl.py --configs configs/sft/*.yaml [--resume]
```

## Preference
We train DPO from the SFT checkpoint on the MATH train split. It shares the sampling settings and reward with RL, so the two are comparable.
```bash
python src/make_pairs.py --configs configs/preference/dpo_math.yaml
python src/dpo_trl.py --configs configs/preference/dpo_trl_math.yaml
python src/dpo.py --configs configs/preference/dpo_math.yaml
```

## RL
We train GRPO from the SFT checkpoint on the MATH train split.
```bash
python src/grpo_trl.py --configs configs/rl/grpo_trl_math*.yaml
python src/grpo.py --configs configs/rl/grpo_math.yaml
```

## Distillation
TODO: Add distillation from a larger teacher model.

## Evaluation
The script evaluates all runs by default and skips finished tasks.
```bash
bash scripts/eval_leaderboard.sh [run_name ...]
```

## Results
The table shows the best run per stage on Open LLM Leaderboard v2. All runs are in [RESULTS.md](RESULTS.md), and more evaluations are in [04-evaluation](../04-evaluation/README.md).

<!-- results:start -->
| Stage | Model | IFEval | MuSR | MATH Lvl-5 | BBH | MMLU-Pro | GPQA | Avg |
|---|---|---|---|---|---|---|---|---|
| **Base** | Qwen/Qwen3.5-0.8B-Base | 25.88 | 38.10 | 7.33 | 40.72 | 24.84 | 29.19 | 27.68 |
| **Instruct** | Qwen/Qwen3.5-0.8B | 53.97 (+28.09) | 42.72 (+4.62) | 12.99 (+5.66) | 33.76 (-6.96) | 16.27 (-8.57) | 26.01 (-3.18) | 30.95 (+3.28) |
<!-- results:end -->

## Benchmarks
1. Coding: TODO: Add HumanEval+ and MBPP+, and later LiveCodeBench.
2. Agentic: We use IFEval. TODO: Add IFBench and BFCL.
3. Reasoning, knowledge and multimodal:
   - Reasoning: We use MuSR, MATH Lvl-5 and BBH. TODO: Add MATH-500 and GSM8K.
   - Knowledge: We use MMLU-Pro and GPQA. TODO: Add MMLU-Redux.
   - Multimodal: TODO: Add MMMU, MathVista (mini) and OCRBench.
4. Cybersecurity: TODO.

## Others
TODO: Document common failure modes such as reward hacking, KL collapse, length bias, eval contamination, verbosity bias and template mismatches.

## Reference & Acknowledgement:

- huggingface/trl: [GitHub](https://github.com/huggingface/trl)
- allenai/open-instruct: [GitHub](https://github.com/allenai/open-instruct)
- EleutherAI/lm-evaluation-harness: [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)