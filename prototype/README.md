# Prototype

A toy end-to-end LLM pipeline: tokenizer → pretrain → SFT → RL. Everything runs on a CPU in a few minutes.
The real implementations live in the numbered folders.

## Run
```bash
cd prototype
python tok_train.py --vocab_size 1024                                  # BPE on TinyShakespeare
python pretrain.py --depth 2 --steps 400 --batch_size 8 --seq_len 128  # -> ckpts/base.pt
python sft.py --steps 120                                              # -> ckpts/sft.pt
python rl.py                                                           # GRPO on a verified reward
```

## Pieces
- **Tokenizer:** regex-pretokenized byte-level BPE. Byte pairs are merged by frequency up to `vocab_size`.
- **Model:** a mini GPT (nanochat-style). RoPE, RMSNorm, QK-norm, MHA or GQA, ReLU², untied embedding and head. `--depth` sets the size.
- **Optimizer:** Muon on matrices, AdamW on embeddings and lm_head. `--adamw_only` for comparison.
- **Packing:** best-fit. Each row is exactly `seq_len`. The largest document that fits goes in first. The shortest is cropped only if nothing fits.
- **SFT:** chat rendering with special tokens. Loss only on assistant tokens.
- **RL:** minimal GRPO. Group-relative advantages on an exact-match reward.

## Project Structure
```
prototype/
├── tokenizer.py   # BPE tokenizer
├── tok_train.py   # train the tokenizer
├── data.py        # TinyShakespeare shards, best-fit packing, loader
├── model.py       # mini GPT
├── optim.py       # Muon + AdamW
├── pretrain.py    # next-token pretraining, val bpb, samples
├── sft.py         # chat rendering, masked SFT
└── rl.py          # GRPO
```