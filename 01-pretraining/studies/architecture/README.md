# Architecture Study

One-change ablations on the shared core at 25M, winners confirmed at 50M. The winners form the modern baseline. Results are in [RESULTS.md](../../RESULTS.md#ablations).

## Protocol
- Change one thing from the baseline. Keep data order, tokens and eval set fixed. Match params within ±5%.
- Sweep LR over ×{0.5, 1, 2} per variant and keep the best.
- Differences smaller than the 3-seed variance of the baseline are ties.

## Switches
| Group | Variants | Flag |
|---|---|---|
| Optimizer | AdamW, Muon, Lion | `--optimizer`, `--lr`, `--muon_lr` |
| Schedule | cosine vs WSD, decay 10–30% | `--scheduler`, `--wsd_fract_decay` |
| Norm | LayerNorm vs RMSNorm, pre vs post, QK-norm | `--norm`, `--norm_position`, `--qk_norm` |
| Activation | GELU, ReLU², SwiGLU, GeGLU (hidden = 8/3 × d) | `--activation` |
| Positions | learned, RoPE (θ 10k / 500k), NoPE | `--pos_emb`, `--rope_theta` |
| Attention | MHA, GQA 4:1, MQA | `--n_kv_head` |
| Embeddings | tied vs untied | `--untie_embeddings` |
| Stability | z-loss, logit soft-cap, depth-scaled init | `--z_loss`, `--logit_softcap`, `--depth_scaled_init` |
| Shape | depth vs width at fixed params | `--n_layer`, `--n_embd` |
| Batch size | 0.13M / 0.26M / 0.52M tokens | `--acc_steps` |

## Run
From `01-pretraining/`. `--base_lr 1e-3` is the best 25M LR from hyperparameter tuning.
```bash
# optimizer and schedule shift the best LR the most: lr x {0.5, 1, 2}
python studies/architecture/ablations.py screen --base_lr 1e-3 --groups optimizer schedule

# the rest: lr x {1, 2}, since 5e-4 was clearly worse for the baseline
python studies/architecture/ablations.py screen --base_lr 1e-3 --lr_mults 1 2 \
  --groups norm activation positions attention embeddings stability shape batch

# confirm winners at 50M, then collect results
python studies/architecture/ablations.py confirm --base_lr 1e-3 --variants muon swiglu --lr_mults 0.5 1
cd src && python -m pretraining.results --ablations 25m_adamw --update_results
```
- About 27 hours of 25M runs. Finished runs are skipped. `--dry_run` prints the commands.
- Muon sweeps its matrix LR around 0.02. Lion is centered at 0.2× the base LR.
- If the best LR is at the sweep edge, add the missing multiplier, e.g. `--lr_mults 4`.

The modern stack in one run:
```bash
python -m pretraining.train --run_name modern --optimizer muon --norm rmsnorm --qk_norm \
  --activation swiglu --pos_emb rope --untie_embeddings
```