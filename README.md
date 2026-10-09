<div align="center">

# RecipeEZ
### A Static, Ingredient-First Recipe Search and Visualization Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]()
[![Build Status](https://img.shields.io/badge/build-in_progress-yellow.svg)]()
![Research](https://img.shields.io/badge/research-four_model_comparison-2dd4bf.svg)

[Overview](#overview) • [Architecture](#architecture--pipeline) • [Features](#key-features) • [Research](#research) • [Documentation](#documentation)

---
**A purely data-driven interface for recipe retrieval.**

<img src="https://github.com/user-attachments/assets/e79ec594-6fe5-49a0-aba2-f3b9e6dbed7e" alt="RecipeEZ Demo" width="540">

</div>

## Overview

Modern recipe websites are optimized for search engines and long-form narrative content rather than for cooking. RecipEZ is the opposite: a distraction-free, browser-rendered galaxy where the recipe data is the interface.

RecipEZ works as an inverse recipe search engine. Instead of searching for a specific dish, users enter ingredients they already have, receive a matching recipe seed, and explore its five semantically nearest recipes in a 3D point cloud.

The application ships without a live backend. Model vectors, UMAP coordinates, and K-NN neighbors are precomputed offline and loaded as static JSON.

## Key Features

- **Ingredient-first search:** Enter available ingredients to find a contextual recipe seed, then inspect its precomputed neighbors.
- **Four-model ablation:** Toggle between combined SBERT, ingredient-only SBERT, name-only SBERT, and TF-IDF retrieval spaces.
- **Model hover details:** Each model button shows a short technical description on hover.
- **Zero-backend infrastructure:** Host the generated files with any static web server.
- **30,000-recipe galaxy:** Explore a WebGL point cloud with interactive recipe details, ontology filters, and nearest-neighbor links.
- **Reproducible research:** Regenerate model metrics and comparison figures without rerunning embedding or UMAP training.

## Architecture & Pipeline

**Offline pipeline**

1. Sample and clean recipes from the source dataset.
2. Build four feature representations:
   - MiniLM-L6-v2 embeddings for combined text
   - MiniLM-L6-v2 embeddings for ingredients
   - MiniLM-L6-v2 embeddings for recipe names
   - TF-IDF followed by 100-component SVD
3. Project each vector space to 3D with cosine-based UMAP.
4. Precompute five nearest neighbors in each original retrieval space.
5. Export static JSON bundles for the frontend.

**Runtime frontend**

1. Load one selected model bundle.
2. Render recipe coordinates as a Three.js point cloud.
3. Search locally by name or ingredient.
4. Open a recipe to display ingredients, instructions, and its five model-specific neighbors.

## Project Structure

```text
RecipEZ/
├── web/                  # Static Three.js frontend and generated model payloads
├── pipeline/             # Offline model builder and evaluation script
├── docs/                 # Architecture, pipeline, evaluation, and sequence diagrams
├── research/             # Reproducible metrics, comparison report, and figures
├── data/raw/             # Ignored raw dataset location
└── old/                  # Archived legacy apps, docs, tests, and presentations
```

## Research

The active evaluation compares all four retrieval modes over the same 30,000 recipes, 21 ontology clusters, and 8,436 unique ingredient strings.

| Model | Tag purity | Ingredient Jaccard | 3D overlap |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.5995 | 0.1143 | 0.1248 |
| SBERT Combined | 0.5406 | 0.1182 | 0.1421 |
| SBERT Ingredients | 0.3704 | 0.1785 | 0.1031 |
| SBERT Names | 0.4553 | 0.0969 | 0.1533 |

The result is not a single winner. TF-IDF best matches the current tag ontology, ingredient-only SBERT best preserves literal ingredients, and the SBERT modes offer concept-oriented alternatives. Most pairwise neighbor overlaps are below 0.06, so switching models meaningfully changes retrieval.

The browser starts in the **Statistical (TF-IDF)** mode by default, and it is the first model in the switcher.

Useful references:

- [Model behavior radar](research/figures/model_behavior_radar.png)
- [UMAP projection gallery](research/figures/galaxy_projection_gallery.png)
- [Model neighbor-overlap heatmap](research/figures/model_neighbor_overlap_heatmap.png)
- [Cluster tag-purity heatmap](research/figures/cluster_tag_purity_heatmap.png)
- [Cluster ingredient-overlap heatmap](research/figures/cluster_ingredient_overlap_heatmap.png)
- [Neighbor tag purity](research/figures/neighbor_tag_purity.png)
- [Ingredient Jaccard similarity](research/figures/ingredient_jaccard_similarity.png)
- [Semantic vs geometry overlap](research/figures/semantic_vs_geometry_overlap.png)
- [PCA vs UMAP comparison](research/figures/pca_vs_umap.png)

See [Model comparison research](research/model-comparison.md) for the complete report.

## Documentation

- [Documentation index](docs.md)
- [Architecture](docs/architecture.md)
- [Build pipeline](docs/pipeline.md)
- [Model evaluation](docs/model-evaluation.md)
- [Sequence diagrams](docs/sequence-diagram.md)
- [Dataset and generated artifacts](docs/data.md)
- [Model comparison research](research/model-comparison.md)
- [Legacy archive map](old/README.md)

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Provide the raw dataset

Download [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-reviews) and place `RAW_recipes.csv` at:

```text
data/raw/RAW_recipes.csv
```

### 3. Generate the four static model bundles

```bash
python pipeline/build_models.py
```

This writes four 30,000-recipe JSON files to `web/model_data/`:

- `sbert_all_data.json`
- `sbert_ingredients_data.json`
- `sbert_names_data.json`
- `tfidf_data.json`

### 4. Serve the static frontend

```bash
python -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/web/
```

### 5. Reproduce the research metrics

```bash
python pipeline/evaluate_models.py
```

This writes `research/metrics.json` and eight generated figures under `research/figures/`. The evaluation reads the model bundles and does not rerun SBERT, TF-IDF, SVD, or UMAP.

## Model Modes

| Mode | Browser label | Feature input | Vector space |
| --- | --- | --- | --- |
| TF-IDF | Statistical | Name, ingredients, and tags | TF-IDF followed by 100-component SVD |
| SBERT Combined | Combined | Name, ingredients, and tags | 384D MiniLM-L6-v2 embeddings |
| SBERT Ingredients | Ingredients | Ingredient list | 384D MiniLM-L6-v2 embeddings |
| SBERT Names | Names | Recipe title | 384D MiniLM-L6-v2 embeddings |

## License

This project source is licensed under GPL-3.0. Dataset use is subject to the terms of the source dataset provider.
