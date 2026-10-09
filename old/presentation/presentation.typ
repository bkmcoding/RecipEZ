#set document(title: "RecipEZ Documentation & Architecture", author: "Michael Hannan")

// Page Setup: Clean, balanced margins for dense reading
#set page(
  paper: "us-letter",
  margin: (x: 1.5in, y: 1.5in),
  numbering: "1",
  fill: rgb("#111111")
)

// Font Setup: Dense, raw serif for body text to maximize readability
#set text(font: ("Linux Libertine", "Times New Roman", "Georgia"), size: 10.5pt, fill: rgb("#e2e8f0"))

#set par(justify: true, leading: 0.7em, spacing: 1.2em)
#set heading(numbering: "1.1.")

// Minimalist Link Styling: No bright colors, just a clean underline
#show link: it => [
  #set text(fill: rgb("#111111"))
  #underline(stroke: 0.5pt + rgb("#111111"), offset: 2pt)[#it]
]

// Headings: Small, sharp, sans-serif, and completely out of the way
#show heading: set text(font: ("Inter", "Helvetica Neue", "Arial"), fill: white)

#show heading.where(level: 1): it => block(above: 2.5em, below: 1em)[
  #text(size: 11pt, weight: "bold", tracking: 0.05em)[#upper(it)]
]

#show heading.where(level: 2): it => block(above: 2em, below: 0.8em)[
  #text(size: 10.5pt, weight: "semibold")[#it]
]

// Raw, unboxed definition blocks
#let def-block(title, body) = block(
  above: 1.5em, below: 1.5em,
  width: 100%,
  [
    #text(font: ("Inter", "Helvetica Neue LT Pro"), weight: "bold", size: 10.5pt)[#title]
    #v(0.2em)
    #body
  ]
)

// --- DOCUMENT START ---

#align(left)[
  #text(font: ("Inter", "Helvetica Neue LT Pro"), weight: 800, size: 16pt)[RecipEZ: Architecture & Documentation]
  #v(0.2em)
  #text(size: 10.5pt, style: "italic")[Michael Hannan | 3D Recipe Search Engine Pipeline]
  #v(3em)
]

#figure(
  grid(
    columns: (1fr, 1fr), // Two equal columns
    gutter: 1.5em,       // Spacing between the images
    
    // Left Image
    block(
      width: 100%, 
      height: 2in,                  // Lock both to the exact same height
      clip: true,                   // Crops any part of the image                 // Softens the abrasive sharp corners
      stroke: 0.5pt + rgb("#e2e8f0"), // Adds a subtle border to separate black from white
      image("./view1.png", width: 100%, height: 100%, fit: "cover")
    ),
    
    // Right Image
    block(
      width: 100%, 
      height: 2in, 
      clip: true, 
      stroke: 0.5pt + rgb("#e2e8f0"),
      image("./view2.png", width: 100%, height: 100%, fit: "cover")
    )
  ),
    caption: [Shows comparison between V2.1 and V1.3]
) <cluster-gallery>


= Preface <intro>
Wandering on the modern web is such a major nuisance. Scouring the web you can find bloat covered ads about practically any topic, everything is built to either gather data or make money off of you. RecipEZ was intended to be a fun personal project that helped solve this problem in the context of cooking and specifically recipes. Recipes tend to be a type of content that falls heavily for this bloat trap, with ads for cookware, meal kits, and grocery delivery services littering the search results and life stories about authors that have nothing to do with the actual recipe and that most people do not care about. The goal of RecipEZ was to create a simple, interactive search engine that accepts a list of raw ingredients and outputs the top recipe matches in a visually appealing 3D space.

= Brainstorming and Architecture
With the constraints defined—accepting a list of raw ingredients and outputting top recipe matches—several approaches were evaluated to establish a mathematical or semantic relationship between the input and the output. 

== Initial Brainstorm
The immediate thought was to simply pass the ingredients into a Large Language Model (LLM) and have it generate a recipe. However, this approach was discarded as it is computationally expensive, abstracts away the core technical challenge of building a search engine, and introduces hallucination risks. Fine-tuned Sequence-to-Sequence (Seq2Seq) models were also evaluated. While excellent for machine translation, they require massive amounts of training data and compute resources that were out of scope for this architecture.

== Data
One of the biggest challenges was dealing with data. The initial dataset was provided by the instructor but I had unfortunately came across the fact that it was riddled with inconsistencies and was not considered "normalized" as I would call it. For example, the same ingredient could be represented in multiple ways ("chopped onions", "onions, chopped", "3/4 cups of onions"). Because the vectorization process treats unique text strings as separate features, these semantic equivalents were ruining the vector distance calculations. After several attempts to write custom cleaning algorithms, the most effective solution was pivoting to an entirely separate, pre-normalized dataset to ensure clean vectorization.

== Model Selection 
When initially researching I had come across multiple different techniques to search for recipes through ingredients. Firstly, there are two main approaches to using this data, either through classification or through clustering. Classification would involve training a model to predict the recipe category (e.g., "dessert", "soup") based on the ingredients, while clustering would involve grouping similar recipes together in a high-dimensional space. I chose the clustering approach because it makes more sense for this use case as the goal is in technicality is to filter and find recipes that are similar to the input ingredients, and clustering allows for a more flexible and intuitive search experience. This caused me to stumble across the obvious choice of TF-IDF for vectorization, UMAP for dimensionality reduction, and K-NN for searching. I later stumbled upon SBERT as well for vectorization

= The Data Pipeline <pipeline>
The core engine of RecipEZ operates in three distinct phases to take text and turn it into a 3D searchable galaxy.

#v(1em)
#align(center)[
  #text(font: ("Inter", "Helvetica Neue"), size: 9pt)[PIPELINE FLOW: INPUT $arrow.r$ #link(<tfidf>)[TF-IDF] $arrow.r$ #link(<knn>)[K-NN] $arrow.r$ #link(<threejs>)[THREE.JS]]
]
#v(1em)

== Vectorization
The first step converts textual data—ingredients and instructions—into a numerical format processed by machine learning algorithms. #link(<tfidf>)[TF-IDF] weighs the importance of each word in the context of the entire dataset. Later iterations also explored SBERT, an advanced method using pre-trained language models to generate dense vectors that capture deeper semantic relationships.

== Dimensionality Reduction
After vectorization, the data exists in an extremely high-dimensional space. To render this in a web browser, the dimensions must be reduced while preserving the local and global structure. #link(<svd>)[SVD] is used initially to discard noise and identify latent features. Following this, #link(<umap>)[UMAP] compresses the data into a 3D coordinate system, creating the clusters visualized in the final application.

== Clustering and Searching
With the recipes mapped in 3D space, the actual search engine functionality is powered by #link(<knn>)[K-NN]. When a user inputs ingredients, the text is vectorized, placed into the space, and the algorithm retrieves the closest recipe nodes, which are then passed to the frontend.

= Decisions <challenges>

== UMAP over PCA
While PCA is a common dimensionality reduction technique, it is linear and often fails to capture the complex relationships in high-dimensional text data. UMAP, on the other hand, is non-linear and excels at preserving both local and global structures, making it ideal for creating meaningful clusters in the 3D space.

== Custom Ontology for Visualization
I felt that simply plotting the recipes in 3D space would not be visually engaging or informative. To enhance the user experience, I created a custom ontology that categorized recipes into clusters (e.g., "Desserts", "Soups", "Grilled Meats") and assigned distinct colors to each cluster. This not only made the visualization more appealing but also provided users with immediate visual cues about the type of recipes they were exploring.

== Web Visualization
Rendering this high-dimensional math in a browser required a robust frontend. #link(<threejs>)[Three.js] was selected as the modern standard for web-based 3D graphics. While it featured a moderate learning curve, it was absolutely essential to achieving the interactable "galaxy" theme envisioned for the project. There were major challenges in optimizing performance, handling user interactions, and ensuring the visualization was both responsive and informative. Graphics libraries in fact are not easy to work with and almost never do exactly what you want them to, so a lot of time was spent debugging and tweaking the rendering process to get it just right.

== Evaluating Accuracy
A lingering challenge is quantitatively testing the accuracy of the K-NN clusters. While visually the clusters make semantic sense, establishing a rigid metric to prove the search model is returning mathematically optimal recipes remains a primary focus for future iterations.

#pagebreak()

= Appendix: Technical Definitions <appendix>

#def-block("TF-IDF (Term Frequency-Inverse Document Frequency)")[
  An NLP technique that converts text into numerical vectors. It multiplies how often a word appears in a specific document (Term Frequency) by the inverse of how often it appears across all documents (Inverse Document Frequency). In RecipEZ, this naturally penalizes common ingredients like "water" or "salt", while giving high mathematical weight to unique identifiers like "saffron" or "truffle oil", allowing for distinct culinary clustering.
  
  #v(0.5em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <tfidf>

#def-block("UMAP (Uniform Manifold Approximation and Projection)")[
  A non-linear dimensionality reduction technique. Unlike linear methods (like PCA), UMAP excels at preserving both the local relationships (recipes right next to each other) and the global structure (how the dessert cluster relates to the meat cluster) when squashing high-dimensional TF-IDF vectors down into an interactable 3D visual space.
  
  #v(0.5em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <umap>

#def-block("K-NN (K-Nearest Neighbors)")[
  A supervised learning algorithm used here for spatial searching. Because TF-IDF creates a vectorized representation of the culinary universe, when a user inputs a query, K-NN simply measures the distance (often using Cosine Similarity) from the user's input vector to the $K$ closest recipe vectors in the database, returning them as the search results.
  
  #v(0.5em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <knn>

#def-block("SVD (Single Value Decomposition)")[
  A mathematical technique used to reduce data dimensionality by identifying latent features. In this pipeline, it acts as a preprocessing step before UMAP to filter out noise and improve performance. It ensures the 3D visualization captures the most important features of the recipes without affecting the underlying accuracy of the K-NN search.
  
  #v(0.5em)
  #link(<pipeline>)[$arrow.l$ Return to Data Pipeline]
] <svd>

  #v(0.5em)
  #link(<challenges>)[$arrow.l$ Return to Challenges]
] <threejs>