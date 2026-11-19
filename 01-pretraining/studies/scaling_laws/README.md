# Scaling Laws Study (WIP)

A small isoFLOP sweep to check compute-optimal model size vs tokens.

**Takeaways:**
1. Model size and data should be scaled roughly equally.
2. 20 tokens per parameter from empirical experience.  
Note: In actual model training, no one trains strictly to the pure ~20 tokens-per-parameter rule due to inference cost.

TODO: papernotes + reproductions??
- Training Compute-Optimal Large Language Models (2022): [arXiv](https://arxiv.org/abs/2203.15556)
- Scaling Laws for Neural Language Models (2020): [arXiv](https://arxiv.org/abs/2001.08361)