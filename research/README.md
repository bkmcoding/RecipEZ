# RecipEZ Research Suite

This directory contains the reproducible comparison of the four active model modes and the retained PCA vs UMAP diagnostic.

## Outputs

- [`metrics.json`](metrics.json) — model, pairwise, and dataset metrics
- [`model-comparison.md`](model-comparison.md) — current comparison and interpretation
- [`figures/`](figures/) — eight generated comparison charts and the retained PCA vs UMAP diagnostic

## Reproduce

```bash
python pipeline/evaluate_models.py
```

The evaluation script reads the four model bundles under `web/model_data/`. It does not rerun SBERT, TF-IDF, SVD, or UMAP.
