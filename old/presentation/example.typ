// Make sure this is at the top of your document
#import "@preview/polylux:0.4.0": *

// ============================================================
// SHARED SLIDE STYLE
// ============================================================
#set page(paper: "presentation-16-9", fill: rgb("#0a0a0c"), margin: 2.5em)
#set text(font: ("JetBrains Mono", "Courier New"), size: 20pt, fill: rgb("#e2e8f0"))
#show heading.where(level: 1): it => block(below: 1em)[
  #text(fill: rgb("#2dd4bf"), weight: "bold")[ ]
  #text(fill: white, weight: "bold")[#upper(it.body)]
]
// Level-2 headings are used ONLY for separator pages — styled as section titles
#show heading.where(level: 2): it => text(fill: rgb("#2dd4bf"), weight: "bold", size: 48pt)[#upper(it.body)]

#pagebreak()

// ============================================================
// ❶ SEPARATOR — TF-IDF
// ============================================================
#page(
  paper: "presentation-16-9",
  fill: rgb("#0a0a0c"),
  margin: 2.5em
)[
  #align(horizon)[
    == TF-IDF
    #v(0.4em)
    #text(size: 22pt, fill: rgb("#a1a1aa"), font: ("JetBrains Mono", "Courier New"))[
      Term Frequency — Inverse Document Frequency \
      Turning ingredient lists into comparable number arrays.
    ]
    #v(1.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2dd4bf"))
  ]
]

// ============================================================
// TF-IDF SLIDES
// ============================================================

#slide[
  = Vectorization: TF-IDF

  #text(fill: rgb("#a1a1aa"))[Problem: A computer can't compare "Chicken Parm" and "Eggplant Parm" — it can only compare numbers.]
  #v(0.8em)

  #only(1)[
    *The idea:* Turn each recipe into a list of numbers — one per ingredient.
    If an ingredient appears in the recipe, its number goes up.
    #v(0.6em)
    `Recipe A: ["Chicken", "Garlic", "Salt"]` \
    `Recipe B: ["Beef",    "Garlic", "Salt"]`
    #v(0.6em)
    $ v_A = ["Chicken": 1, "Garlic": 1, "Salt": 1] $
  ]

  #only(2)[
    *The problem with raw counts:* \
    "Salt" is in 99% of recipes. It drowns out everything else. \
    Every recipe ends up pointing in roughly the same direction — toward "Salt."
    #v(0.6em)
    $ v_A = ["Chicken": 1, "Garlic": 1, underbrace("Salt": 1, "same in every recipe")] $
    #v(0.4em)
    #text(fill: rgb("#f43f5e"))[Result: recipes cluster by pantry staples, not by flavor.]
  ]
]

#slide[
  = TF-IDF: Penalizing Common Ingredients

  #text(fill: rgb("#a1a1aa"))[Fix: scale down ingredients that appear everywhere. Scale up ingredients that are distinctive.]
  #v(0.8em)

  #only(1)[
    *Inverse Document Frequency* — the rarity penalty:
    $ "IDF"(t) = log frac(N, "DF"(t)) $
    $N$ = total recipes. $"DF"(t)$ = how many recipes contain $t$. \
    Rarer ingredient = higher score.
    #v(0.5em)
    - "Chicken" in 100/10,000 recipes $arrow$ $"IDF" = log(100) = 2.0$
    - "Salt" in 9,900/10,000 recipes $arrow$ $"IDF" = log(1.01) approx 0.004$
  ]

  #only(2)[
    *Multiply TF × IDF:*
    $ v_A = [2.0,\ 0.5,\ 0.004] $
    #v(0.5em)
    #text(fill: rgb("#2dd4bf"))[>> execution_success] \
    "Salt" is nearly zeroed out. "Chicken" dominates. \
    Two recipes with the same *distinctive* ingredients now point in the same direction \
    — and K-NN can find them.
  ]
]

#pagebreak()

// ============================================================
// ❷ SEPARATOR — UMAP
// ============================================================
#page(
  paper: "presentation-16-9",
  fill: rgb("#0a0a0c"),
  margin: 2.5em
)[
  #align(horizon)[
    == UMAP
    #v(0.4em)
    #text(size: 22pt, fill: rgb("#a1a1aa"), font: ("JetBrains Mono", "Courier New"))[
      Uniform Manifold Approximation and Projection \
      Compressing 5,000 dimensions into a renderable 3D space.
    ]
    #v(1.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2dd4bf"))
  ]
]

// ============================================================
// UMAP SLIDES
// ============================================================

#slide[
  = UMAP: The Dimension Problem

  #text(fill: rgb("#a1a1aa"))[We have 5,000 ingredients. Each one is its own axis. That's 5,000 dimensions.]
  #v(0.8em)

  #only(1)[
    After TF-IDF, every recipe is a coordinate in $RR^(5000)$. \
    That's mathematically valid — but completely unrenderable.
    #v(0.5em)
    $ M in RR^(10{,}000 times 5{,}000) quad arrow.r quad "Three.js needs" RR^3 $
    #v(0.4em)
    We need to compress 5,000 axes down to 3 *without scrambling which recipes are similar to which.*
  ]

  #only(2)[
    *Why not just drop 4,997 axes?* \
    Any single axis (e.g. "Garlic") tells you almost nothing alone. \
    The *relationships between axes* is where the information lives.
    #v(0.5em)
    *Why not PCA?* \
    PCA finds straight lines of variance. Flavor space is curved — \
    "Ramen" is close to both "Soups" and "Asian Noodles" simultaneously. \
    #text(fill: rgb("#2dd4bf"))[We need something that follows the shape of the data.]
  ]
]

#slide[
  = UMAP: Mapping the Neighborhood Structure

  #text(fill: rgb("#a1a1aa"))[UMAP records which recipes are neighbors in 5,000D, then rebuilds those relationships in 3D.]
  #v(0.5em)

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      #only(2)[
        *Step 1 — Local neighborhoods* \
        For each recipe, find its $k$ nearest neighbors ($k=15$). \
        The distance to the closest one sets the local scale $rho_i$:
        $ rho_i = min_(j in "kNN"(i)) d(i,j) $
      ]
      #only(3)[
        *Step 2 — Fuzzy edges* \
        Connections are weighted probabilities, not hard lines:
        $ w_(i j) = exp(-(d(i,j) - rho_i) \/ sigma_i) $
        Nearby recipes: $w approx 1$ \
        Distant recipes: $w approx 0$
      ]
    ],
    align(center)[
      #only(2)[
        #block(stroke: 1pt + rgb("#2dd4bf"), radius: 4pt, clip: true)[
          #image("./umap_radius.png", width: 100%)
        ]
      ]
      #only(3)[
        #block(stroke: 1pt + rgb("#2dd4bf"), radius: 4pt, clip: true)[
          #image("./umap_topology.png", width: 100%)
        ]
      ]
    ]
  )
]

#slide[
  = UMAP: Rebuilding the Map in 3D

  #text(fill: rgb("#a1a1aa"))[Recipes start at random 3D positions. Two forces push and pull them into the right shape.]
  #v(0.5em)

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *Attraction* — pulls neighbors together
      $ F_"att" = frac(-2 a b d^(2b-2), 1 + a d^(2b)) $
      #text(size: 17pt, fill: rgb("#a1a1aa"))[High-$w$ pairs (similar recipes) \get pulled close.]
    ],
    [
      *Repulsion* — pushes non-neighbors apart
      $ F_"rep" = frac(2b, (epsilon + d^2)(1 + a d^(2b))) $
      #text(size: 17pt, fill: rgb("#a1a1aa"))[Low-$w$ pairs (unlike recipes) \get pushed away.]
    ]
  )

  #v(0.5em)
  ~200 optimization steps. Converges when the 3D map matches the 5,000D map. \
  #text(fill: rgb("#2dd4bf"))[Proximity in 3D space = culinary similarity.]
]

#slide[
  = UMAP: Output

  #text(fill: rgb("#a1a1aa"))[Each recipe is now an $(x, y, z)$ coordinate. Similar recipes are physically close.]
  #v(0.5em)

  #only(1)[
    $ v_("ChickenParm")   = [+14.2,  -3.8,  +8.1] $
    $ v_("EggplantParm")  = [+13.8,  -3.5,  +8.4] $
    $ v_("ChocolateCake") = [-22.1, +17.3,  -5.9] $
    #v(0.5em)
    "Chicken Parm" ↔ "Eggplant Parm": *0.6 units apart* \
    "Chicken Parm" ↔ "Chocolate Cake": *37 units apart* \
    #text(fill: rgb("#2dd4bf"))[The geometry now reflects culinary reality.]
  ]

  #only(2)[
    #text(fill: rgb("#2dd4bf"))[>> render_ready] \
    5,000 ingredient axes → 3 spatial coordinates. \
    Three.js can place every recipe as a point in a navigable 3D galaxy.
    #v(0.8em)
    `TF-IDF` #text(fill: rgb("#2dd4bf"))[✓] ` → UMAP` #text(fill: rgb("#2dd4bf"))[✓] ` → SVD → K-NN`
  ]
]

#pagebreak()

// ============================================================
// ❸ SEPARATOR — SVD
// ============================================================
#page(
  paper: "presentation-16-9",
  fill: rgb("#0a0a0c"),
  margin: 2.5em
)[
  #align(horizon)[
    == SVD
    #v(0.4em)
    #text(size: 22pt, fill: rgb("#a1a1aa"), font: ("JetBrains Mono", "Courier New"))[
      Singular Value Decomposition \
      Discovering hidden flavor dimensions from ingredient co-occurrence.
    ]
    #v(1.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2dd4bf"))
  ]
]

// ============================================================
// SVD SLIDES
// ============================================================

#slide[
  = SVD: Finding Hidden Flavor Groups

  #text(fill: rgb("#a1a1aa"))[TF-IDF treats every ingredient as independent. But ingredients travel in packs.]
  #v(0.8em)

  #only(1)[
    Basil, Oregano, Tomato, and Mozzarella almost always appear together. \
    They're not four separate axes of information — they're one concept: *Italian*. \
    #v(0.5em)
    TF-IDF has no way to know this. The 5,000D space is full of \
    redundant, correlated dimensions.
    #v(0.4em)
    #text(fill: rgb("#f43f5e"))[Consequence: K-NN is comparing noise alongside signal.]
  ]

  #only(2)[
    *SVD discovers these "flavor axes" automatically* by analyzing \
    which ingredients co-occur across the whole corpus.
    #v(0.5em)
    $ M_(10{,}000 times 5{,}000) arrow.r "SVD" arrow.r tilde(M)_(10{,}000 times 128) $
    #v(0.4em)
    5,000 raw ingredient axes → 128 *latent flavor dimensions*. \
    #text(fill: rgb("#2dd4bf"))[91% of the information. 2.5% of the dimensions.]
  ]
]

#slide[
  = SVD: The Decomposition

  #text(fill: rgb("#a1a1aa"))[Any matrix can be factored into three simpler matrices.]
  #v(0.5em)

  $ M = U Sigma V^T $

  #v(0.4em)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
      *$U$* \
      Shape: $N times N$ \
      Each row = one recipe as a mix of flavor dimensions. \
      #text(size: 17pt, fill: rgb("#a1a1aa"))[$U^T U = I$]
    ],
    [
      *$Sigma$* \
      Diagonal. \
      $sigma_1 >= sigma_2 >= dots >= 0$ \
      Each value = how much variance that flavor dimension explains. \
      #text(size: 17pt, fill: rgb("#a1a1aa"))[Sorted by importance.]
    ],
    [
      *$V^T$* \
      Shape: $D times D$ \
      Each row = the ingredient blend for one flavor dimension. \
      #text(size: 17pt, fill: rgb("#a1a1aa"))[$V V^T = I$]
    ]
  )
]

#slide[
  = SVD: Keeping Only What Matters

  #text(fill: rgb("#a1a1aa"))[Keep the top $k$ dimensions. Drop the rest — they're noise.]
  #v(0.5em)

  $ M approx tilde(M) = U_k Sigma_k V_k^T $
  $ ||M - tilde(M)||_F = sqrt(sum_(i=k+1)^r sigma_i^2) quad arrow.r quad "minimized" $

  #v(0.5em)
  The Eckart-Young theorem guarantees this is the *best possible* rank-$k$ approximation. \
  At $k = 128$: variance explained = *91.3%*, dimensions dropped = *97.4%*.
  #v(0.4em)
  The small $sigma_i$ values we discard encode coincidences — \
  not real culinary relationships. \
  #text(fill: rgb("#2dd4bf"))[We keep the signal. We discard the noise.]
]

#slide[
  = SVD: What the Flavor Dimensions Look Like

  #text(fill: rgb("#a1a1aa"))[The rows of $V_k^T$ are not hand-labeled — the algorithm discovers these groupings on its own.]
  #v(0.4em)

  #text(size: 17pt)[
    `Dim 1  (σ=142):  Basil +0.41 · Tomato +0.38 · Oregano +0.35 · Mozzarella +0.29` \
    #text(fill: rgb("#2dd4bf"))[→ Italian / Mediterranean] \
    #v(0.3em)
    `Dim 2  (σ=119):  Soy Sauce +0.44 · Ginger +0.39 · Sesame +0.31 · Rice Wine +0.28` \
    #text(fill: rgb("#2dd4bf"))[→ East Asian] \
    #v(0.3em)
    `Dim 3  (σ=97):   Cumin +0.42 · Chili +0.40 · Coriander +0.33 · Lime +0.27` \
    #text(fill: rgb("#2dd4bf"))[→ South Asian / Mexican] \
    #v(0.3em)
    `Dim 4  (σ=84):   Butter +0.45 · Sugar +0.43 · Vanilla +0.38 · Flour +0.35` \
    #text(fill: rgb("#2dd4bf"))[→ Desserts / Baking]
  ]
  #v(0.4em)
  `ChickenParm:` #text(fill: rgb("#2dd4bf"))[`Dim1=0.91`] ` · Dim2=0.04 · Dim3=0.07 · Dim4=0.02`
]

#slide[
  = SVD: Output

  #text(size: 18pt)[
    *Input:* \
    `M_sparse   shape=(10000, 5000)   nnz=198,421   density=0.4%`
    #v(0.4em)
    *Fit — randomized SVD, $k=128$:* \
    `variance_explained = 91.3%` \
    `top singular values: [142.3, 118.7, 97.1, 84.2, ...]`
    #v(0.4em)
    *Output:* \
    `M_latent   shape=(10000, 128)   dtype=float32   density=100%`
  ]
  #v(0.5em)
  #text(fill: rgb("#2dd4bf"))[>> execution_success] \
  `TF-IDF` #text(fill: rgb("#2dd4bf"))[✓] ` → UMAP` #text(fill: rgb("#2dd4bf"))[✓] ` → SVD` #text(fill: rgb("#2dd4bf"))[✓] ` → K-NN`
]

#pagebreak()

// ============================================================
// ❹ SEPARATOR — K-NN
// ============================================================
#page(
  paper: "presentation-16-9",
  fill: rgb("#0a0a0c"),
  margin: 2.5em
)[
  #align(horizon)[
    == K-Nearest Neighbors
    #v(0.4em)
    #text(size: 22pt, fill: rgb("#a1a1aa"), font: ("JetBrains Mono", "Courier New"))[
      K-Nearest Neighbors \
      Efficiently finding the closest recipes in vector space.
    ]
    #v(1.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2dd4bf"))
  ]
]

// ============================================================
// K-NN SLIDES
// ============================================================

#slide[
  = K-NN: Finding Similar Recipes

  #text(fill: rgb("#a1a1aa"))[Every recipe is now a 128D coordinate. "Similar recipes" means "nearby coordinates."]
  #v(0.8em)

  #only(1)[
    A user clicks "Chicken Parmesan." \
    We need the $K$ recipes whose vectors sit closest to $tilde(v)_("ChickenParm")$.
    #v(0.5em)
    *Brute-force:* compare the query against every recipe.
    $ "Cost" = O(N dot D) = O(10{,}000 times 128) = 1.28 times 10^6 quad "ops/query" $
    Fast enough now — but at 1M recipes × raw 5K dims: $5 times 10^9$ ops. \
    #text(fill: rgb("#f43f5e"))[We need a smarter structure.]
  ]

  #only(2)[
    *The insight:* if we spatially index the vectors, we can skip most comparisons. \
    Recipes that are far away geometrically *can never be close neighbors* \
    — so we don't check them.
    #v(0.5em)
    We use a *KD-Tree* — a binary tree that partitions vector space into nested regions. \
    #text(fill: rgb("#2dd4bf"))[Build once. Every query becomes $O(log N)$ instead of $O(N)$.]
  ]
]

#slide[
  = KD-Tree: Splitting Space

  #text(fill: rgb("#a1a1aa"))[Recursively cuts the search space in half along the most spread-out dimension.]
  #v(0.5em)

  #only(1)[
    At each node, split at the median of the widest dimension:
    $ pi_(d,t) = { x in RR^D : x_d = t }, quad d = arg max_j "Var"(x_j) $
    Left: $x_d < t$. Right: $x_d >= t$. Recurse until leaves hold $<= 40$ recipes. \
    #v(0.4em)
    Build cost: $O(D N log N)$ — paid once. \
    #text(fill: rgb("#2dd4bf"))[For 10,000 recipes: ~14 branch decisions to reach a leaf.]
  ]

  #only(2)[
    *Pruning — the key trick:* \
    After checking the nearest leaf, walk back up the tree. \
    At each branch, measure the distance to the splitting plane:
    $ d_pi = |tilde(v)_"query"[d] - t| $
    If $d_pi >=$ current best distance: *skip the entire subtree.* \
    #v(0.4em)
    For clustered recipe data: *96% of the tree is pruned per query.* \
    #text(fill: rgb("#2dd4bf"))[312 nodes visited out of 10,000.]
  ]
]

#slide[
  = K-NN: Distance Metric

  #text(fill: rgb("#a1a1aa"))[We use cosine similarity — it measures the angle between vectors, not their length.]
  #v(0.5em)

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *Euclidean* — measures gap \
      $ d = sqrt(sum_i (A_i - B_i)^2) $
      #v(0.3em)
      A recipe scaled to double portions looks far from its twin despite identical flavor. \
      #text(fill: rgb("#f43f5e"))[Fooled by scale.]
    ],
    [
      *Cosine* — measures angle \
      $ "sim"(A, B) = frac(A dot B, ||A|| dot ||B||) $
      #v(0.3em)
      Same flavor ratios = 1.0 regardless of recipe size. \
      #text(fill: rgb("#2dd4bf"))[Scale-invariant.]
    ]
  )

  #v(0.5em)
  L2-normalize all vectors: $hat(v) = v \/ ||v||$. \
  On the unit sphere, cosine similarity equals Euclidean distance. \
  #text(fill: rgb("#2dd4bf"))[One step — best of both metrics.]
]

#slide[
  = K-NN: Output

  `find_neighbors(ChickenParm, K=5, metric=cosine)`
  #v(0.6em)

  #only(1)[
    #text(size: 17pt)[
      `[cos=0.97]  Eggplant Parmesan   →  [13.8, -3.5,  8.4, ...]` \
      `[cos=0.94]  Chicken Milanese    →  [14.9, -4.1,  7.6, ...]` \
      `[cos=0.91]  Veal Parmesan       →  [13.1, -3.2,  9.0, ...]` \
      `[cos=0.88]  Baked Ziti          →  [12.4, -5.0,  8.8, ...]` \
      `[cos=0.85]  Chicken Marsala     →  [15.3, -2.9,  7.1, ...]`
    ]
  ]

  #only(2)[
    #text(fill: rgb("#2dd4bf"))[>> query_success] \
    `latency: ~2ms  |  nodes_visited: 312 / 10,000  |  pruned: 96.9%`
    #v(0.8em)
    `TF-IDF` #text(fill: rgb("#2dd4bf"))[✓] ` → UMAP` #text(fill: rgb("#2dd4bf"))[✓] ` → SVD` #text(fill: rgb("#2dd4bf"))[✓] ` → K-NN` #text(fill: rgb("#2dd4bf"))[✓]
  ]
]

#pagebreak()

// ============================================================
// ❺ SEPARATOR — SBERT
// ============================================================
#page(
  paper: "presentation-16-9",
  fill: rgb("#0a0a0c"),
  margin: 2.5em
)[
  #align(horizon)[
    == SBERT
    #v(0.4em)
    #text(size: 22pt, fill: rgb("#a1a1aa"), font: ("JetBrains Mono", "Courier New"))[
      Sentence-BERT \
      A second vectorization strategy — this time, biased toward meaning.
    ]
    #v(1.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2dd4bf"))
  ]
]

// ============================================================
// SBERT SLIDES
// ============================================================

#slide[
  = SBERT: A Second Vectorizer

  #text(fill: rgb("#a1a1aa"))[TF-IDF vectorizes by token frequency. SBERT is a completely separate model that vectorizes by meaning.]
  #v(0.7em)

  #only(1)[
    Both produce vectors that feed into the same UMAP visualization pipeline. \
    They are two different answers to the same question: \
    *"How do we turn a recipe into a point in space?"*
    #v(0.5em)
    #grid(
      columns: (1fr, 1fr),
      gutter: 1.5em,
      [
        *TF-IDF* \
        Counts ingredient tokens. \
        "Chicken" and "Poultry" \
        are unrelated axes. \
        #text(fill: rgb("#a1a1aa"))[Lexical.]
      ],
      [
        *SBERT* \
        Encodes semantic meaning. \
        "Chicken" and "Poultry" \
        land near each other. \
        #text(fill: rgb("#2dd4bf"))[Semantic.]
      ]
    )
  ]

  #only(2)[
    *Why have both?* \
    They produce different 3D galaxies — different lenses on the same data. \
    TF-IDF clusters by *shared ingredients*. \
    SBERT clusters by *culinary meaning and description*. \
    #v(0.5em)
    A recipe called "Braised Short Ribs" and one called "Slow-Cooked Beef in Red Wine" \
    will be far apart in TF-IDF space — and close together in SBERT space. \
    #text(fill: rgb("#2dd4bf"))[Users can switch between views.]
  ]
]

#slide[
  = SBERT: How It Works

  #text(fill: rgb("#a1a1aa"))[SBERT is a fine-tuned BERT model that outputs a single fixed-size vector per sentence.]
  #v(0.6em)

  #only(1)[
    Standard BERT outputs one vector *per token*, not per sentence. \
    To get a single sentence vector, we average all token outputs — *mean pooling:*
    #v(0.4em)
    $ u = frac(1, n) sum_(i=1)^n h_i in RR^(768) $
    #v(0.3em)
    Where $h_i$ is the output for the $i$-th token. \
    Always 768 dimensions, regardless of input length.
  ]

  #only(2)[
    *How meaning is trained in:* \
    SBERT is trained with triplet loss — similar sentences are pulled together, \
    dissimilar ones pushed apart:
    $ cal(L) = max(0, ||u_a - u_p||^2 - ||u_a - u_n||^2 + epsilon) $
    After training: *geometric distance ≈ semantic distance.* \
    "Spicy chicken stir-fry" and "fiery poultry wok dish" land close together. \
    #text(fill: rgb("#2dd4bf"))[Same concept → small angle → high cosine similarity.]
  ]
]

#slide[
  = SBERT: Three Visualization Modes

  #text(fill: rgb("#a1a1aa"))[The site runs three separate SBERT models in parallel, each producing a different 3D galaxy.]
  #v(0.5em)

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
      *Ingredients* \
      #text(fill: rgb("#2dd4bf"))[Mode A] \
      #v(0.3em)
      SBERT encodes the *ingredient list* as a sentence. \
      #v(0.2em)
      Clusters by what's in the recipe. \
      Similar to TF-IDF but meaning-aware — "olive oil" and "EVOO" land together.
    ],
    [
      *Recipe Name* \
      #text(fill: rgb("#2dd4bf"))[Mode B] \
      #v(0.3em)
      SBERT encodes only the *recipe title.* \
      #v(0.2em)
      Clusters by dish concept and cuisine. \
      "Chicken Tikka" and "Chicken Curry" cluster even if ingredients differ.
    ],
    [
      *Combined* \
      #text(fill: rgb("#2dd4bf"))[Mode C] \
      #v(0.3em)
      SBERT encodes *name + ingredients* together. \
      #v(0.2em)
      Balances both signals. \
      The default view — captures both what a dish *is* and what it *contains.*
    ]
  )
]

#slide[
  = SBERT: Same Pipeline, Different Lens

  #text(fill: rgb("#a1a1aa"))[Each SBERT mode feeds directly into the same UMAP → Three.js pipeline as TF-IDF.]
  #v(0.6em)

  #only(1)[
    All three SBERT models output 768D vectors. \
    UMAP compresses each set independently into its own 3D coordinates. \
    #v(0.4em)
    #text(size: 18pt)[
      `Mode A (Ingredients):  v_ChickenParm = [+11.3, -2.1, +6.8]` \
      `Mode B (Name):         v_ChickenParm = [+8.7,  +4.2, -1.3]` \
      `Mode C (Combined):     v_ChickenParm = [+12.9, -1.7, +5.2]`
    ]
    #v(0.4em)
    Three different maps. Same recipes. Different groupings depending \
    on which signal matters most to the user.
  ]

  #only(2)[
    #text(fill: rgb("#2dd4bf"))[>> all_modes_ready] \
    #v(0.5em)
    #text(size: 18pt)[
      `TF-IDF vectorizer    →  UMAP  →  3D galaxy (token frequency)`  \
      `SBERT / Ingredients  →  UMAP  →  3D galaxy (ingredient meaning)` \
      `SBERT / Name         →  UMAP  →  3D galaxy (dish concept)` \
      `SBERT / Combined     →  UMAP  →  3D galaxy (full context)`
    ]
    #v(0.5em)
    The user selects a mode. The visualization updates. \
    Same K-NN search engine underneath — only the vector space changes.
  ]
]

#pagebreak()

// ============================================================
// RETURN TO STANDARD DOCUMENT FORMAT
// ============================================================
#set page(paper: "us-letter", fill: rgb("#111111"), margin: (x: 1.5in, y: 1.5in))
#set text(font: "Linux Libertine", size: 10.5pt, fill: rgb("#e2e8f0"))

= K-NN Evaluation
Returning to the primary text...
