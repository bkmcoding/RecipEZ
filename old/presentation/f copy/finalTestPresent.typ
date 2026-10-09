#set document(title: "RecipEZ Documentation & Architecture", author: "Michael Hannan")

#set page(
  paper: "us-letter",
  margin: (x: 1.5in, y: 1.5in),
  numbering: "1",
  fill: rgb("#111111")
)

#set text(font: ("Linux Libertine", "Times New Roman", "Georgia"), size: 10.5pt, fill: rgb("#e2e8f0"))
#set par(justify: true, leading: 0.7em, spacing: 1.2em)
#set heading(numbering: "1.1.")

#show link: it => [
  #set text(fill: rgb("#2dd4bf"))
  #underline(stroke: 0.5pt + rgb("#2dd4bf"), offset: 2pt)[#it]
]

#show heading: set text(font: ("Inter", "Helvetica Neue", "Arial"), fill: white)

#show heading.where(level: 1): it => block(above: 2.5em, below: 1em)[
  #text(size: 11pt, weight: "bold", tracking: 0.05em)[#upper(it)]
]

#show heading.where(level: 2): it => block(above: 2em, below: 0.8em)[
  #text(size: 10.5pt, weight: "semibold")[#it]
]

#let def-block(title, body) = block(
  above: 1.5em, below: 1.5em,
  width: 100%,
  [
    #text(font: ("Inter", "Helvetica Neue LT Pro"), weight: "bold", size: 10.5pt)[#title]
    #v(0.2em)
    #body
  ]
)

#let viz-figure(path, caption-text) = figure(
  block(
    width: 100%,
    clip: true,
    stroke: 0.5pt + rgb("#2dd4bf"),
    image(path, width: 100%, fit: "contain")
  ),
  caption: caption-text
)

#align(left)[
  #text(font: ("Inter", "Helvetica Neue LT Pro"), weight: 800, size: 16pt)[RecipEZ: Architecture & Documentation]
  #v(0.2em)
  #text(size: 10.5pt, style: "italic")[Michael Hannan | 3D Recipe Search Engine Pipeline]
  #v(3em)
]

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    block(
      width: 100%,
      height: 2in,
      clip: true,
      stroke: 0.5pt + rgb("#e2e8f0"),
      image("./view1.png", width: 100%, height: 100%, fit: "cover")
    ),
    block(
      width: 100%,
      height: 2in,
      clip: true,
      stroke: 0.5pt + rgb("#e2e8f0"),
      image("./view2.png", width: 100%, height: 100%, fit: "cover")
    )
  ),
  caption: [Comparison between V2.1 and V1.3]
) <cluster-gallery>


= Preface <intro>

Wandering on the modern web is such a major nuisance. Scouring the web you can find bloat covered ads about practically any topic, everything is built to either gather data or make money off of you. RecipEZ was intended to be a fun personal project that helped solve this problem in the context of cooking and specifically recipes. Recipes tend to be a type of content that falls heavily for this bloat trap, with ads for cookware, meal kits, and grocery delivery services littering the search results and life stories about authors that have nothing to do with the actual recipe. The goal of RecipEZ was to create a simple, interactive search engine that accepts a list of raw ingredients and outputs the top recipe matches in a visually appealing 3D space.

= Brainstorming and Architecture

With the constraints defined—accepting a list of raw ingredients and outputting top recipe matches—several approaches were evaluated to establish a mathematical or semantic relationship between the input and the output.

== Initial Brainstorm

The immediate thought was to simply pass the ingredients into a Large Language Model (LLM) and have it generate a recipe. However, this approach was discarded as it is computationally expensive, abstracts away the core technical challenge of building a search engine, and introduces hallucination risks. Fine-tuned Sequence-to-Sequence (Seq2Seq) models were also evaluated. While excellent for machine translation, they require massive amounts of training data and compute resources that were out of scope for this architecture.

== Data

One of the biggest challenges was dealing with data. The initial dataset was riddled with inconsistencies and was not considered normalized. For example, the same ingredient could be represented in multiple ways ("chopped onions", "onions, chopped", "3/4 cups of onions"). Because the vectorization process treats unique text strings as separate features, these semantic equivalents were corrupting the vector distance calculations. After several attempts to write custom cleaning algorithms, the most effective solution was pivoting to an entirely separate, pre-normalized dataset to ensure clean vectorization.

== Model Selection

When initially researching I had come across multiple different techniques to search for recipes through ingredients. There are two main approaches: classification or clustering. Classification would involve training a model to predict the recipe category based on the ingredients, while clustering groups similar recipes together in a high-dimensional space. The clustering approach was chosen because the goal is to filter and find recipes similar to the input ingredients, allowing for a more flexible and intuitive search experience. This led to the selection of TF-IDF for vectorization, UMAP for dimensionality reduction, and K-NN for searching. SBERT was later explored as an alternative vectorization strategy.

= The Data Pipeline <pipeline>

The core engine of RecipEZ operates in three distinct phases to take text and turn it into a 3D searchable galaxy.

#v(1em)
#align(center)[
  #text(font: ("Inter", "Helvetica Neue"), size: 9pt)[PIPELINE FLOW: INPUT $arrow.r$ #link(<tfidf>)[TF-IDF] $arrow.r$ #link(<svd>)[SVD] $arrow.r$ #link(<umap>)[UMAP] $arrow.r$ #link(<knn>)[K-NN] $arrow.r$ #link(<threejs>)[THREE.JS]]
]
#v(1em)

The full pipeline transformation across all stages can be traced visually in #link(<viz6>)[Figure 6], which shows how individual recipes from each culinary category are reshaped at every step from raw ingredient counts through to final UMAP coordinates.

#viz-figure("./viz6_pipeline_trace.png", [
  #link(<viz6>)[VIZ 6] — Pipeline Trace: parallel coordinates showing recipe transformation across all stages. Each polyline is one recipe; colors encode culinary category. The convergence from chaotic left-side axes to structured right-side UMAP coordinates demonstrates the pipeline collapsing high-dimensional noise into geometry.
]) <viz6>

== Vectorization <vectorization-section>

The first step converts textual data—ingredients and tags—into a numerical format that machine learning algorithms can process. #link(<tfidf>)[TF-IDF] weighs the importance of each word in the context of the entire dataset. Later iterations also explored SBERT, an advanced method using pre-trained language models to generate dense vectors that capture deeper semantic relationships.

The key mechanism TF-IDF exploits is the inverse document frequency penalty: common ingredients like "salt" or "water" that appear in nearly every recipe are mathematically suppressed, while rare and distinctive ingredients like "saffron" or "za'atar" receive high weight. The full spectrum of this penalty across the vocabulary is shown in #link(<viz1>)[Figure 1].

#viz-figure("./viz1_idf_decay.png", [
  #link(<viz1>)[VIZ 1] — Ingredient Gravity Spectrum: IDF decay curve sorted by ascending score. The shaded pink zone on the left marks pantry staples penalized to near-zero. The green zone on the right marks high-IDF ingredients that carry the most discriminative weight in the vector space. Annotated words reveal which specific ingredients fall at key inflection points.
]) <viz1>

Before vectorization, the raw co-occurrence structure of ingredients across the corpus can be explored directly. #link(<viz7>)[Figure 7] shows the 80 most frequent ingredients as a force-directed graph, where edge weight encodes how often two ingredients appear together.

#viz-figure("./viz7_cooccurrence_graph.png", [
  #link(<viz7>)[VIZ 7] — Ingredient Constellation: co-occurrence force graph. Node size scales with ingredient frequency; edge brightness and thickness scale with PMI-weighted co-occurrence strength. Tight clusters of tightly-bound ingredients (e.g., the garlic–olive oil–tomato group) reveal the latent flavor groups that SVD will later discover formally.
]) <viz7>

== Dimensionality Reduction <dimreduction-section>

After vectorization, the data exists in an extremely high-dimensional space (up to 10,000 dimensions). To render this in a web browser, the dimensions must be reduced while preserving local and global structure. #link(<svd>)[SVD] is used first to discard noise and identify latent features, compressing the sparse TF-IDF matrix into 100 dense latent components. Following this, #link(<umap>)[UMAP] compresses the data into a 3D coordinate system, creating the clusters visualized in the final application.

The latent components discovered by SVD are not arbitrary mathematical constructs — they correspond to interpretable culinary flavor axes. #link(<viz2>)[Figure 2] shows the top positive and negative ingredient loadings for the first six SVD components, revealing that the algorithm recovers groupings like Italian, East Asian, and Dessert/Baking without any supervision.

#viz-figure("./viz2_svd_flavor_profiles.png", [
  #link(<viz2>)[VIZ 2] — The Hidden Palate: diverging bar charts of ingredient loadings for SVD components 1–6. Teal bars indicate positive loadings (the "positive pole" of the flavor axis); pink bars indicate negative loadings (the "negative pole"). Each component reads as a legible culinary contrast, validating that the 100 latent dimensions are semantically meaningful despite capturing only ~40% of raw variance.
]) <viz2>

A critical question for any dimensionality reduction step is whether the compression preserves the distances that matter. #link(<viz3>)[Figure 3] directly tests this: for 2,500 randomly sampled recipe pairs, it plots their cosine distance in 100D SVD space against their Euclidean distance in 2D UMAP space.

#viz-figure("./viz3_umap_fidelity.png", [
  #link(<viz3>)[VIZ 3] — UMAP Fidelity Witness: each point is a pair of recipes. The x-axis is their true distance in 100D SVD space; the y-axis is their distance after UMAP compression. The amber moving-average line demonstrates a strong monotonic relationship, confirming that UMAP preserves the rank ordering of similarities — recipes close in high-dimensional space remain close in the 3D galaxy.
]) <viz3>

== Clustering and Searching <search-section>

With recipes mapped in 3D space, the search engine functionality is powered by #link(<knn>)[K-NN]. When a user inputs ingredients, the text is vectorized, placed into the space, and the algorithm retrieves the closest recipe nodes using cosine similarity, which are then passed to the frontend.

#link(<viz4>)[Figure 4] shows a live query dissection: a concrete set of input ingredients is vectorized, projected into 2D UMAP space, and the Top-8 returned recipes are annotated with their names and cosine similarity scores.

#viz-figure("./viz4_knn_anatomy.png", [
  #link(<viz4>)[VIZ 4] — Search Oracle: K-NN query anatomy for the query "eggs, bacon, cheddar, sourdough, butter." The teal star marks the projected query vector. Concentric dashed rings represent cosine distance thresholds. Pink highlighted points are the Top-8 returned recipes; each is annotated with its recipe name and cosine similarity score. The spatial clustering of results confirms that K-NN is recovering semantically coherent neighborhoods.
]) <viz4>

To validate that the spatial clusters in the 3D galaxy correspond to real culinary categories, #link(<viz5>)[Figure 5] cross-tabulates K-Means spatial clusters against the Tag Ontology labels.

#viz-figure("./viz5_purity_matrix.png", [
  #link(<viz5>)[VIZ 5] — Galaxy Cluster Purity Matrix: rows are the 12 K-Means spatial clusters found in 2D UMAP coordinates; columns are culinary tag categories from the Tag Ontology. Each cell shows what fraction of that spatial cluster belongs to each tag. A strong diagonal indicates that the unsupervised geometry is recovering human culinary taxonomy. Off-diagonal noise reflects genuine cross-cuisine overlap rather than model failure.
]) <viz5>

= Decisions <challenges>

== UMAP over PCA

While PCA is a common dimensionality reduction technique, it is linear and often fails to capture the complex relationships in high-dimensional text data. UMAP, on the other hand, is non-linear and excels at preserving both local and global structures, making it ideal for creating meaningful clusters in the 3D space. The fidelity comparison in #link(<viz3>)[Figure 3] quantifies this preservation directly.

== Custom Ontology for Visualization

Simply plotting the recipes in 3D space would not be visually engaging or informative. To enhance the user experience, a custom ontology was created that categorizes recipes into named clusters (e.g., "Desserts", "Soups", "Grilled Meats") and assigns distinct colors to each cluster. The purity matrix in #link(<viz5>)[Figure 5] validates that these color assignments are geometrically grounded — the spatial clusters align with the ontology labels.

== Web Visualization

Rendering this high-dimensional math in a browser required a robust frontend. #link(<threejs>)[Three.js] was selected as the modern standard for web-based 3D graphics. While it featured a moderate learning curve, it was essential to achieving the interactable "galaxy" theme envisioned for the project.

== Evaluating Accuracy

A lingering challenge is quantitatively testing the accuracy of the K-NN clusters. While visually the clusters make semantic sense, the purity matrix in #link(<viz5>)[Figure 5] represents the first formal attempt at a cluster validity metric. Establishing additional quantitative measures remains a primary focus for future iterations.

#pagebreak()

= Appendix: Technical Definitions <appendix>

#def-block("TF-IDF (Term Frequency-Inverse Document Frequency)")[
  An NLP technique that converts text into numerical vectors. It multiplies how often a word appears in a specific document (Term Frequency) by the inverse of how often it appears across all documents (Inverse Document Frequency). In RecipEZ, this naturally penalizes common ingredients like "water" or "salt", while giving high mathematical weight to unique identifiers like "saffron" or "truffle oil", allowing for distinct culinary clustering.

  _Diagnostic:_ See #link(<viz1>)[Figure 1 — IDF Decay Curve] for a visualization of this penalty spectrum across the full vocabulary.

  _Presentation:_ See the #link(<slides-tfidf>)[TF-IDF slides] for a step-by-step derivation.

  #v(0.3em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <tfidf>

#def-block("SVD (Singular Value Decomposition)")[
  A mathematical technique used to reduce data dimensionality by identifying latent features. In this pipeline, it acts as a preprocessing step before UMAP to filter noise and compress the 10,000-feature TF-IDF matrix into 100 dense latent components. The Eckart-Young theorem guarantees that the truncated reconstruction minimizes the Frobenius-norm error among all rank-$k$ approximations.

  _Diagnostic:_ See #link(<viz2>)[Figure 2 — SVD Latent Flavor Profiles] for the ingredient loadings of the first six components, and #link(<viz6>)[Figure 6 — Pipeline Trace] for how SVD coordinates compare to raw and UMAP coordinates.

  _Presentation:_ See the #link(<slides-svd>)[SVD slides] for the full decomposition walkthrough.

  #v(0.3em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <svd>

#def-block("UMAP (Uniform Manifold Approximation and Projection)")[
  A non-linear dimensionality reduction technique. Unlike linear methods like PCA, UMAP preserves both the local relationships (recipes near each other) and the global structure (how the dessert cluster relates to the meat cluster) when compressing high-dimensional vectors into a 3D visual space.

  _Diagnostic:_ See #link(<viz3>)[Figure 3 — UMAP Fidelity Witness] for a direct measurement of distance preservation before and after compression.

  _Presentation:_ See the #link(<slides-umap>)[UMAP slides] for the attraction/repulsion optimization walkthrough.

  #v(0.3em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <umap>

#def-block("K-NN (K-Nearest Neighbors)")[
  A spatial search algorithm used here to retrieve the closest recipe vectors to a user's query. Because TF-IDF creates a vectorized representation of the culinary universe, K-NN measures the cosine similarity from the query vector to all recipe vectors in the database, returning the $K$ closest as results.

  _Diagnostic:_ See #link(<viz4>)[Figure 4 — K-NN Query Anatomy] for a dissected live query, and #link(<viz5>)[Figure 5 — Cluster Purity Matrix] for a spatial cluster validity assessment.

  _Presentation:_ See the #link(<slides-knn>)[K-NN slides] for the KD-Tree pruning walkthrough.

  #v(0.3em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <knn>

#def-block("Three.js")[
  A JavaScript library for rendering 3D graphics in a web browser via WebGL. In RecipEZ, each recipe is rendered as a point in a navigable 3D galaxy, with position determined by UMAP coordinates and color determined by the Tag Ontology cluster. Three.js handles the scene graph, camera, raycasting for click interactions, and instanced geometry for 30,000 simultaneous recipe points.

  #v(0.3em)
  #link(<challenges>)[$arrow.l$ Return to Decisions]
] <threejs>

#def-block("Diagnostic Visualizations")[
  The following seven diagnostic plots were generated from the live pipeline to provide empirical evidence that each mathematical stage is functioning as intended.

  #v(0.4em)
  - *#link(<viz1>)[Figure 1] — IDF Decay Curve.* Proves TF-IDF is correctly penalizing ubiquitous ingredients and amplifying rare ones.
  - *#link(<viz2>)[Figure 2] — SVD Flavor Profiles.* Proves SVD latent components are semantically interpretable culinary axes, not random noise.
  - *#link(<viz3>)[Figure 3] — UMAP Fidelity Scatter.* Proves UMAP preserves high-dimensional distance rank ordering in the 3D projection.
  - *#link(<viz4>)[Figure 4] — K-NN Query Anatomy.* Proves K-NN is retrieving semantically coherent neighborhoods for a concrete query.
  - *#link(<viz5>)[Figure 5] — Tag Purity Matrix.* Proves spatial UMAP clusters correspond to human culinary taxonomy categories.
  - *#link(<viz6>)[Figure 6] — Pipeline Trace.* Shows the full transformation path of individual recipes across every pipeline stage simultaneously.
  - *#link(<viz7>)[Figure 7] — Ingredient Constellation.* Reveals the raw co-occurrence graph structure that SVD is formalizing in its latent decomposition.
]
