<div align="center">

# 🍳 RecipeEZ
### The "Anti-Blog" Recipe Galaxy Search Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]()
![Build Status](https://img.shields.io/badge/build-in_progress-yellow.svg)
<!--[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)]()-->




[Overview](#-the-issue) • [Features](#-key-features) • [Tech Stack](#-the-pipeline) • [Documentation](docs.md)

---
**Stop scrolling through life stories. Start cooking.**

<img src="https://github.com/user-attachments/assets/e79ec594-6fe5-49a0-aba2-f3b9e6dbed7e" alt="RecipeEZ Demo" width="540">

</div>

## 📌 The Issue (Search is not finished)
Most recipe sites are buried under ads, trackers, and 2,000-word essays about a summer in Tuscany. **RecipeEZ** is built for the "pantry-first" cook. You input what you have; we give you the match. No fluff, just food.

## Key Features
* **Pantry-First Search:** Input ingredients you already own to find matching recipes.
* **Zero Bloat:** No ads, no life stories, just ingredients and instructions.
* **Smart Matching:** Uses advanced vector search to find recipes even if the wording differs.

## The Pipeline
To ensure high-accuracy results, the project utilizes a sophisticated machine learning backend:

1.  **Vectorization:** TF-IDF for ingredient importance weighting.
2.  **Dimensionality Reduction:** UMAP to optimize the search space.
3.  **Clustering/Search:** K-Nearest Neighbors (K-NN) for real-time recipe retrieval.

## 🚀 Quick Start (Newest Version Unfinished)
```bash
# MODIFY DATA PATH to your local data if needed

# Clone the repository
git clone [https://github.com/yourusername/RecipeEZ.git](https://github.com/yourusername/RecipeEZ.git)

# Install dependencies
pip install -r requirements.txt

# Run the engine
python main.py

``` 
Data is provided by shuyangli94 - [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-reviews](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-reviews)
