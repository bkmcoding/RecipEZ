# Model Evaluation

The evaluation suite in [`pipeline/evaluate_models.py`](../pipeline/evaluate_models.py) compares the four model modes without rerunning their expensive embedding or projection stages.

## Metrics

### Neighbor tag purity

The proportion of a model's five nearest neighbors that share the same ontology cluster as the seed recipe.

### Mean ingredient Jaccard similarity

The mean Jaccard similarity between a seed recipe's ingredient set and each neighbor's ingredient set:

```text
|A ∩ B| / |A ∪ B|
```

### 3D silhouette score

Silhouette score is calculated over exported `x`, `y`, and `z` coordinates using ontology labels. This evaluates whether the visualization separates labeled clusters. It does **not** evaluate retrieval quality in the original high-dimensional space.

### Semantic-to-geometric neighbor overlap

For each recipe, the five nearest neighbors in exported 3D space are compared with the five precomputed semantic neighbors using Jaccard overlap. A value of `1.0` would mean the 3D cloud perfectly preserves the retrieval neighborhood.

### Pairwise model overlap

For each recipe, two models' five-neighbor sets are compared with Jaccard overlap. The report records the mean overlap for every model pair.

### Model distinctiveness

For each model, distinctiveness is:

```text
1 - mean(pairwise overlap with the other models)
```

Higher values indicate that the model tends to retrieve different neighborhoods.

### Cluster metrics

Tag purity and ingredient Jaccard are also calculated separately for each of the 21 ontology clusters. This reveals regions where models agree strongly, such as Dessert Core, and sparse regions where averages should be interpreted cautiously.

## Current Results

| Model | Tag purity | Ingredient Jaccard | 3D silhouette | 3D/semantic overlap | Distinctiveness |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.5995 | 0.1143 | -0.1185 | 0.1248 | 0.9613 |
| SBERT Combined | 0.5406 | 0.1182 | -0.1367 | 0.1421 | 0.8880 |
| SBERT Ingredients | 0.3704 | 0.1785 | -0.1618 | 0.1031 | 0.9698 |
| SBERT Names | 0.4553 | 0.0969 | -0.1627 | 0.1533 | 0.9052 |

## Interpretation

- **TF-IDF has the strongest ontology-tag agreement**, suggesting the combined lexical signal remains effective for this tag taxonomy.
- **SBERT Ingredients has the highest literal ingredient overlap**, as expected from its input representation.
- **SBERT Names has the weakest ingredient overlap**, but the strongest semantic-to-geometric overlap; it emphasizes dish concept over ingredient inventory.
- **All 3D silhouette scores are negative**, meaning ontology labels are not cleanly separated in the 3D visual space.
- **Visual proximity is not equivalent to retrieval ranking.** The 3D/semantic overlap remains low for every mode.
- **Most model pairs disagree on the majority of recipe neighborhoods.** The strongest relationship is SBERT Combined × SBERT Names at `0.2367` mean overlap; every other pair is below `0.06`.

## Reproducible Outputs

```bash
python pipeline/evaluate_models.py
```

This writes:

- [`research/metrics.json`](../research/metrics.json) — model, pairwise, distinctiveness, cluster, and dataset metrics
- eight generated PNG files under [`research/figures/`](../research/figures/)
- the retained [`pca_vs_umap.png`](../research/figures/pca_vs_umap.png) legacy diagnostic is not regenerated
