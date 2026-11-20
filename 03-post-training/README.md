# 03 - Post-training

This part turns Qwen3.5-0.8B-Base into an instruct model with SFT, DPO, GRPO and distillation. Each run is one config, and all runs share the same benchmarks.

## Project Structure
```
03-post-training/
├── configs/<stage>/<run_name>.yaml  # one config per run: sft, preference, rl
├── src/
│   ├── common.py                    # config loading, model I/O, log-probs
│   ├── data.py                      # SFT, MATH and KodCode datasets
│   ├── rewards.py                   # math and code rewards
│   ├── sandbox.py                   # runs model code with limits
│   ├── code_eval.py                 # HumanEval+ and MBPP+ eval
│   ├── sft_trl.py                   # SFT with TRL
│   ├── grpo.py                      # GRPO, optional vLLM rollouts
│   ├── make_pairs.py                # DPO pairs from SFT samples
│   └── dpo.py                       # DPO
├── scripts/
│   ├── eval_leaderboard.sh          # evaluate all runs
│   └── collect_results.py           # fill the results tables
└── RESULTS.md                       # all runs
```

## SFT
We fine-tune on smol-smoltalk with full fine-tuning and LoRA.
```bash
python src/sft_trl.py --configs configs/sft/*.yaml [--resume]
```

## Preference
We train DPO from the SFT checkpoint on MATH. It shares sampling settings and rewards with RL.
```bash
python src/make_pairs.py --configs configs/preference/dpo_math.yaml
python src/dpo.py --configs configs/preference/dpo_math.yaml
```

## RL
We train GRPO from the SFT checkpoint on MATH and on KodCode-Light-RL-10K (code, CC BY-NC 4.0).
```bash
python src/grpo.py --configs configs/rl/grpo_math.yaml
python src/grpo.py --configs configs/rl/grpo_code.yaml
```

Code rewards run the last ```python block against the problem's tests in the sandbox:
- `code_pass`: 1 if all tests pass.
- `code_fraction`: share of tests passed.
- `code_format`: 0.1 for a code block.

Train options:
- `scale_rewards`: divide advantages by the group std.
- `epsilon_high`: upper clip bound (default: `epsilon`).
- `micro_batch_size`: completions per backward pass. It saves memory without changing the update.
- `use_vllm`: generate rollouts with vLLM in the same process (written for vLLM 0.29). Weights sync after every step.
- `log_rollouts_every`: write completions, rewards and test results to `rollouts.jsonl` every N steps.

With vLLM, `vllm_logprob_gap` in `metrics.jsonl` should stay small. Growth means vLLM has stale weights. If memory is tight, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Distillation
TODO: Add distillation from a larger teacher model.

## Evaluation
The script evaluates all runs and skips finished tasks. Coding tasks need `pip install evalplus pytest`.
```bash
bash scripts/eval_leaderboard.sh [run_name ...]
python src/code_eval.py --model ckpts/<run>/final --task humaneval_plus --output_dir results/<run>/humaneval_plus
```

`code_eval.py` scores HumanEval+ and MBPP+ with the official EvalPlus checker. Use `--no_chat_template` for base models and `--num_samples` for pass@k.

## Sandbox
`src/sandbox.py` runs model code in a separate process with time, memory and file-size limits, and without network when `unshare -rn` works. It does not stop code built to escape; use a container for that.

Canonical solutions score the same as unsandboxed EvalPlus.

## Results
The table shows the best run per stage on Open LLM Leaderboard v2. All runs are in [RESULTS.md](RESULTS.md), and more evaluations are in [04-evaluation](../04-evaluation/README.md).

<!-- results:start -->
| Stage | Model | IFEval | MuSR | MATH Lvl-5 | BBH | MMLU-Pro | GPQA | Avg |
|---|---|---|---|---|---|---|---|---|
| **Base** | Qwen/Qwen3.5-0.8B-Base | 25.88 | 38.10 | 7.33 | 40.72 | 24.84 | 29.19 | 27.68 |
| **Instruct** | Qwen/Qwen3.5-0.8B | 53.97 (+28.09) | 42.72 (+4.62) | 12.99 (+5.66) | 33.76 (-6.96) | 16.27 (-8.57) | 26.01 (-3.18) | 30.95 (+3.28) |
<!-- results:end -->

Coding benchmarks, pass@1 (greedy):

<!-- coding:start -->
TODO: run `bash scripts/eval_leaderboard.sh` to fill this table.
<!-- coding:end -->

## Benchmarks
1. Coding: We use HumanEval+ and MBPP+ (EvalPlus). TODO: Add LiveCodeBench.
2. Agentic: We use IFEval. TODO: Add IFBench and BFCL.
3. Reasoning, knowledge and multimodal:
   - Reasoning: We use MuSR, MATH Lvl-5 and BBH. TODO: Add MATH-500 and GSM8K.
   - Knowledge: We use MMLU-Pro and GPQA. TODO: Add MMLU-Redux.
   - Multimodal: TODO: Add MMMU, MathVista (mini) and OCRBench.
4. Cybersecurity: TODO.

## Others
TODO: Document common failure modes such as reward hacking, KL collapse, length bias, eval contamination, verbosity bias and template mismatches.

## References
**Code**
- huggingface/trl: [GitHub](https://github.com/huggingface/trl)
- allenai/open-instruct: [GitHub](https://github.com/allenai/open-instruct)
- EleutherAI/lm-evaluation-harness: [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)

**Papers**
- DeepSeek-R1, RL for reasoning (2025): [arXiv](https://arxiv.org/abs/2501.12948)
- GRPO, DeepSeekMath (2024): [arXiv](https://arxiv.org/abs/2402.03300)
- DPO, Direct Preference Optimization (2023): [arXiv](https://arxiv.org/abs/2305.18290)
- InstructGPT, Training Language Models to Follow Instructions with Human Feedback (2022): [arXiv](https://arxiv.org/abs/2203.02155)
- PPO, Proximal Policy Optimization Algorithms (2017): [arXiv](https://arxiv.org/abs/1707.06347)