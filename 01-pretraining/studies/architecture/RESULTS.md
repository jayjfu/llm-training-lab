# Architecture Study Results

## Tier S Screen
One change from the tier S baseline per run. Each variant gets its own LR sweep; the table keeps its best LR.
Verdict: better / worse if |Δ| is above the noise floor, else tie. "LR at sweep edge" means extend the sweep before trusting it.
Filled by `python -m pretraining.results --runs "s_*" --ablations s_adamw --update_results`.

<!-- ablations:start -->
<!-- ablations:end -->