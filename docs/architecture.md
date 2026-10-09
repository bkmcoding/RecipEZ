# Architecture

RecipEZ separates expensive machine-learning work from the user-facing search experience.

## Design Summary

- **Offline:** Python builds four vector spaces, projects each to 3D, precomputes five nearest neighbors, and exports static JSON.
- **Runtime:** A browser loads one of those JSON bundles, performs a local lexical scan to select a seed recipe, and reads that seed’s precomputed `similar_recipes` list.
- **No live backend:** There is no Python service, database, vector index, authentication layer, or network API at runtime.
- **Visualization and retrieval are distinct:** UMAP coordinates drive the 3D galaxy, while the five similar recipes are retrieved from the original high-dimensional model space.

## Model Variants

| Variant | Input | Vectorization | Dimensionality before UMAP |
| --- | --- | --- | ---: |
| TF-IDF | Name + ingredients + tags | TF-IDF + 100-component SVD | 100 |
| SBERT Combined | Name + ingredients + tags | `all-MiniLM-L6-v2` | 384 |
| SBERT Ingredients | Ingredient list | `all-MiniLM-L6-v2` | 384 |
| SBERT Names | Recipe title | `all-MiniLM-L6-v2` | 384 |

`all-MiniLM-L6-v2` emits 384-dimensional vectors. Earlier drafts that cite 768 dimensions are outdated and are retained only in `old/`.

## Runtime Data Flow

1. `web/index.html` loads Three.js, `style.css`, and `script.js`.
2. The browser starts with the **Statistical (TF-IDF)** model, listed first in the switcher, and fetches `tfidf_data.json` from `web/model_data/`.
3. A recipe-name search performs a lexical scan over loaded names.
4. An ingredient search scans recipe ingredient strings, rewards multiple requested ingredients, and lightly penalizes very long recipes.
5. Once a seed recipe is selected, `similar_recipes` is read directly from that seed.
6. Three.js moves the camera to the seed and displays the seed plus its five semantic neighbors.

Browser search latency therefore depends on local JSON scanning and rendering, not live model inference.

## Static JSON Recipe Schema

```json
{
  "id": "string",
  "name": "string",
  "galaxy_cluster": "string",
  "star_color": "#rrggbb",
  "ingredients": ["string"],
  "steps": ["string"],
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "similar_recipes": ["recipe_id", "recipe_id"]
}
```

The four model bundles contain the same recipe IDs and metadata, but their coordinates and neighbor lists differ because each model defines similarity differently.

## Visualization Constraint

UMAP creates an interpretable 3D point cloud, not the final retrieval index. A 3D projection necessarily loses information. Two recipes can be visual neighbors while having different high-dimensional nearest-neighbor lists. The evaluation report measures this distinction explicitly.
