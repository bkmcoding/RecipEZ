# Build Pipeline

The canonical builder is [`pipeline/build_models.py`](../pipeline/build_models.py).

## Input

```text
data/raw/RAW_recipes.csv
```

The build loads all rows, then deterministically samples 30,000 recipes with `random.seed(42)`.

## Feature Construction

For every sampled recipe:

1. Parse `ingredients`, `tags`, and `steps`.
2. Remove a small set of preparation prefixes such as `diced`, `chopped`, `crushed`, `minced`, `sliced`, and `ground`.
3. Assign a visual ontology cluster and color using the tag mapping.
4. Build three feature inputs:
   - combined: name, ingredients, and tags
   - ingredients only
   - recipe name only

## Model Paths

### SBERT modes

The three SBERT modes encode their feature strings with `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional vectors. Each mode independently runs:

- UMAP to 3D with `n_components=3`, `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`, and `random_state=42`
- cosine K-NN with six neighbors, from which the first self-match is removed

### TF-IDF mode

The TF-IDF mode processes the combined feature string:

1. `TfidfVectorizer(max_df=0.90, min_df=5, max_features=10000)`
2. `TruncatedSVD(n_components=100, random_state=42)`
3. UMAP to 3D with the same settings as the SBERT modes
4. Cosine K-NN over the 100-dimensional SVD matrix

SVD is used only in the TF-IDF path. SBERT does not use SVD.

## Output

All JSON files are written to:

```text
web/model_data/
```

| File | Model mode |
| --- | --- |
| `sbert_all_data.json` | Combined SBERT |
| `sbert_ingredients_data.json` | Ingredients-only SBERT |
| `sbert_names_data.json` | Name-only SBERT |
| `tfidf_data.json` | TF-IDF + SVD |

Each recipe includes metadata, 3D coordinates, ontology color/cluster, and five precomputed neighbor IDs.

## Reproduction

```bash
python pipeline/build_models.py
python pipeline/evaluate_models.py
```

The evaluation script requires the four generated JSON files to exist. It does not rerun SBERT, TF-IDF, SVD, or UMAP.
