# Sequence Diagrams

RecipEZ has two separate phases. The old single-runtime diagram incorrectly implied that a Python engine and vector database were involved during a browser search. The corrected flow is split into offline model generation and static runtime lookup.

## Build-Time Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant CSV as RAW_recipes.csv
    participant Prep as Preprocessing
    participant SBERT as SBERT MiniLM-L6-v2
    participant TFIDF as TF-IDF + SVD
    participant UMAP as UMAP
    participant KNN as 5-NN Search
    participant Export as Static JSON
    participant Files as Model JSON Files

    CSV->>Prep: Load 30,000 sampled recipes
    Prep->>Prep: Parse ingredients, tags, and steps
    Prep->>Prep: Assign ontology cluster and color

    Prep->>SBERT: Build combined, ingredients, and names features
    SBERT->>UMAP: Project 384D embeddings to 3D
    SBERT->>KNN: Precompute cosine neighbors in 384D space

    Prep->>TFIDF: Build combined feature text
    TFIDF->>UMAP: Project 100D SVD matrix to 3D
    TFIDF->>KNN: Precompute cosine neighbors in 100D space

    KNN->>Export: Attach five neighbor IDs to each recipe
    UMAP->>Export: Attach x, y, and z coordinates
    Export->>Files: Write four model JSON bundles
```

## Runtime Search

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Browser as RecipEZ Browser
    participant Bundle as Static Model JSON
    participant Search as Local Search Logic
    participant Render as Three.js Renderer

    Browser->>Bundle: Fetch selected model JSON
    Bundle-->>Browser: Return 30,000 recipes, coordinates, and neighbor lists

    User->>Browser: Enter recipe name
    Browser->>Search: Lexical scan of loaded names
    Search-->>Browser: Return best matching recipe

    User->>Browser: Enter ingredients
    Browser->>Search: Scan ingredient strings and score matches
    Search-->>Browser: Return optimal seed recipe

    Browser->>Bundle: Read seed.similar_recipes
    Bundle-->>Browser: Return five precomputed neighbor IDs
    Browser->>Render: Update selected node and highlight neighbors
    Render-->>User: Show 3D galaxy, recipe details, and similar recipes
```

No Python, database, or vector index participates at runtime.
