## Introduction
TODO: inference throughout (tokens/s), custom evaluation, ...

## Simple Eval
A quick check of any Hugging Face model with vLLM: generative (GSM8K) and categorical (MMLU).
```bash
python simple_eval.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --max_problems 50
python simple_eval.py --task mmlu --model Qwen/Qwen2.5-0.5B-Instruct --max_problems 100
```

## Custom Evaluation
TODO:
1. Compare the pass@k curves of SFT, DPO and GRPO on MATH-500.
2. Check how much vision ability survives text-only post-training on MMMU, MathVista (mini) and OCRBench.
3. Judge chat quality with an LLM on Arena-Hard v2 and AlpacaEval 2.
4. Track response length, clean stops and refusal rate.

## Local Eval Suit
A local evaluation suite for measuring model quality across several standard benchmarks using Inspect AI:
```bash
# Install dependencies
pip install inspect-ai inspect-evals vllm

# Run a full local evaluation suite with Inspect AI and vLLM
bash run_eval_suit.sh

# Interactive viewer
inspect view
```

## Reference & Acknowledgement:

- UKGovernmentBEIS/inspect_ai: [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)