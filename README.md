<div align="center">

# RecipeEZ
### A Static, Ingredient-First Recipe Search and Visualization Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]()
![Build Status](https://img.shields.io/badge/build-in_progress-yellow.svg)

[Overview](#the-issue) • [Architecture](#architecture--pipeline) • [Features](#key-features) • [Documentation](docs.md)

---
**A purely data-driven interface for recipe retrieval.**

<img src="https://github.com/user-attachments/assets/e79ec594-6fe5-49a0-aba2-f3b9e6dbed7e" alt="RecipeEZ Demo" width="540">

</div>

## The Issue
Modern recipe websites are heavily optimized for search engines, prioritizing long-form narrative content and advertisements over functional data. RecipeEZ addresses this by providing a minimalist, distraction-free environment. 

It functions as an inverse search engine: rather than searching for a specific dish, users input the ingredients they currently have, and the system retrieves recipes based on culinary similarity and semantic clustering.

## Key Features
* **Semantic Seed Search:** Input raw ingredients to find contextually relevant recipes using pre-computed K-Nearest Neighbors (K-NN) arrays.
* **Feature Ablation Toggles:** Dynamically switch between vectorization models (SBERT Combined, SBERT Ingredients, SBERT Names, TF-IDF) to observe how different machine learning inputs cluster the data.
* **Zero-Backend Infrastructure:** Hosted entirely via static files. The frontend requires no live server, making deployments to platforms like Vercel fast and free.
* **3D Data Visualization:** Explores 30,000 recipes rendered as an interactive point cloud in the browser using WebGL.

## Architecture & Pipeline
To ensure high-accuracy results while maintaining a static frontend, the application is decoupled into an offline processing script and a client-side interface.

**1. Offline Processing (Python)**
* **Vectorization:** Sentence-BERT (SBERT) and TF-IDF are used to convert textual recipe data into high-dimensional numerical vectors.
* **Dimensionality Reduction:** UMAP compresses the vectors into 3D spatial coordinates ($X, Y, Z$) for visualization.
* **Clustering:** K-NN pre-calculates the top mathematical matches for every recipe.
* **Export:** Processed data is exported as static `.json` payloads.

**2. Client-Side Rendering (JavaScript & Three.js)**
* **Visualization:** Three.js renders the 3D coordinates using `BufferGeometry` for high-performance point cloud rendering.
* **Live Search:** The browser performs a rapid lexical scan to find a root "Seed" recipe, then instantly reads its pre-computed K-NN array to display semantic matches without running live ML inference.

## Quick Start
Because the application uses a decoupled architecture, Python is only required to generate the data payloads. The frontend can be served by any basic HTTP server.

```bash
# Clone the repository
git clone [https://github.com/yourusername/RecipeEZ.git](https://github.com/yourusername/RecipeEZ.git)
cd RecipeEZ

# NOTE: YOU MUST DOWNLOAD DATA YOURSELF OR PROVIDE YOUR OWN DATASET AND CONFIGURE BINDINGS + PATHS

# 1. Generate the JSON data payloads (Requires Python 3.8+)
pip install -r requirements.txt
python ./main/{modelName}.py

# 2. Serve the static frontend locally
python -m http.server 8000

```
Data is provided by shuyangli94 - [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-reviews](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-reviews) 
