import csv
import ast
import json
import random
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# --- GLOBAL PLOT SETTINGS (Brutalist Dark Theme) ---
plt.style.use('dark_background')
BG_COLOR = '#0a0a0c'
ACCENT_COLOR = '#2dd4bf'
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#a1a1aa',
    'ytick.color': '#a1a1aa',
    'axes.edgecolor': '#27272a'
})

print("Loading data...")
all_data = []

with open("./RAW_recipes.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        all_data.append(row)

SAMPLE_SIZE = 30000
print(f"Loaded {len(all_data)} total recipes. Randomly sampling {SAMPLE_SIZE}...")
random.seed(42)
raw_data = random.sample(all_data, SAMPLE_SIZE)

print("Mapping color tags...")

TAG_ONTOLOGY = {
    'vegan':       {'cluster': 'Vegan Sector',       'color': '#00ff00'}, 
    'vegetarian':  {'cluster': 'Vegetarian Sector',  'color': '#228b22'}, 
    'salad':       {'cluster': 'Salad System',       'color': '#7cfc00'}, 
    'dessert':     {'cluster': 'Dessert Core',       'color': '#ff00ff'}, 
    'baking':      {'cluster': 'Baking Sector',      'color': '#9370db'}, 
    'cookie':      {'cluster': 'Cookie Cluster',     'color': '#ffb6c1'}, 
    'cake':        {'cluster': 'Cake Nebula',        'color': '#da70d6'}, 
    'mexican':     {'cluster': 'Mexican Cuisine',    'color': '#ff4500'}, 
    'asian':       {'cluster': 'Asian Cuisine',      'color': '#ff8c00'}, 
    'indian':      {'cluster': 'Indian Cuisine',     'color': '#ffd700'}, 
    'italian':     {'cluster': 'Italian Cuisine',    'color': '#dc143c'}, 
    'french':      {'cluster': 'French Cuisine',     'color': '#c71585'}, 
    'seafood':     {'cluster': 'Seafood System',     'color': '#00ffff'}, 
    'poultry':     {'cluster': 'Poultry System',     'color': '#1e90ff'}, 
    'beef':        {'cluster': 'Beef System',        'color': '#000080'}, 
    'pork':        {'cluster': 'Pork System',        'color': '#ff69b4'}, 
    'soup':        {'cluster': 'Soup Sector',        'color': '#d2691e'}, 
    'stew':        {'cluster': 'Stew Cluster',       'color': '#8b4513'}, 
    'breakfast':   {'cluster': 'Breakfast Belt',     'color': '#ffffe0'}, 
    'brunch':      {'cluster': 'Brunch Belt',        'color': '#ffebcd'}, 
    'beverages':   {'cluster': 'Beverage Ocean',     'color': '#7fffd4'}, 
    'cocktails':   {'cluster': 'Cocktail Nebula',    'color': '#40e0d0'}, 
}

def assign_ontology(tags_list):
    if not tags_list:
        return 'Uncharted Stars', '#fdfbf7'
        
    t_str = " ".join([t.lower() for t in tags_list])
    
    for target_tag, properties in TAG_ONTOLOGY.items():
        if target_tag in t_str:
            return properties['cluster'], properties['color']

    return 'Untagged Core', '#475569'

print("Building feature strings and settings colors...")
prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']
master_features = []  
cluster_counts = {}

for row in raw_data:
    parsed_ingredients = ast.literal_eval(row['ingredients'])
    parsed_tags = ast.literal_eval(row['tags'])
    
    cluster_name, star_color = assign_ontology(parsed_tags)
    row['galaxy_cluster'] = cluster_name
    row['star_color'] = star_color
    
    # Tracking for Plot 1
    cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + 1
    
    ingreds = []
    for item in parsed_ingredients:
        for word in prep_words:
            item = item.replace(word, "")
        ingreds.append(item.strip().replace(" ", "_"))
    
    tags = ["TAG_" + tag.replace(" ", "_") for tag in parsed_tags]
    master_features.append(" ".join(ingreds + tags))

# === PLOT 1: Tag Ontology Distribution ===
print("-> Generating Tag Ontology Distribution Plot...")
plt.figure(figsize=(12, 8))
sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:15]
c_names = [x[0] for x in sorted_clusters]
c_vals = [x[1] for x in sorted_clusters]
c_colors = [next((v['color'] for k, v in TAG_ONTOLOGY.items() if v['cluster'] == name), '#475569') for name in c_names]

# --- NEW: Updated to comply with Seaborn v0.14 API ---
sns.barplot(x=c_vals, y=c_names, hue=c_names, palette=c_colors, legend=False)

plt.title("Top 15 Culinary Clusters by Volume", fontsize=16, weight='bold')
plt.xlabel("Number of Recipes")
plt.tight_layout()
plt.savefig("plot_1_ontology_distribution.png", dpi=300)
plt.close()

print("TF-IDF Vectorizing...")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, max_features=10000)
tfidf_matrix = vectorizer.fit_transform(master_features)

# === PLOT 2: TF-IDF Sparsity (Spy Plot) ===
print("-> Generating TF-IDF Sparsity Plot...")
plt.figure(figsize=(8, 8))
# Plotting a 500x500 slice of the matrix to show sparsity without blowing up RAM
plt.spy(tfidf_matrix[:500, :500], markersize=1, color=ACCENT_COLOR, aspect='auto')
plt.title("TF-IDF Matrix Sparsity (500x500 Slice)", fontsize=16, weight='bold')
plt.xlabel("Feature Index (Ingredients/Tags)")
plt.ylabel("Recipe Index")
plt.tight_layout()
plt.savefig("plot_2_tfidf_sparsity.png", dpi=300)
plt.close()

print(f"{tfidf_matrix.shape[1]} Dimension Reduction with SVD...")
svd = TruncatedSVD(n_components=100, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

# === PLOT 3: SVD Scree Plot ===
print("-> Generating SVD Scree Plot...")
plt.figure(figsize=(10, 6))
cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
plt.plot(range(1, 101), cumulative_variance, color=ACCENT_COLOR, linewidth=2)
plt.fill_between(range(1, 101), cumulative_variance, color=ACCENT_COLOR, alpha=0.1)
plt.axhline(y=cumulative_variance[-1], color='#f43f5e', linestyle='--', alpha=0.5)
plt.title("SVD Cumulative Explained Variance", fontsize=16, weight='bold')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Retained")
plt.tight_layout()
plt.savefig("plot_3_svd_scree.png", dpi=300)
plt.close()

print("UMAP projecting to 2D (for diagnostics) and 3D (for export)...")
# 2D for our diagnostic plots
reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_2d = reducer_2d.fit_transform(svd_matrix)

# 3D for the actual Three.js engine
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_3d = reducer_3d.fit_transform(svd_matrix)

# === PLOT 4: PCA vs UMAP Comparison ===
print("-> Generating PCA vs UMAP Dimensionality Comparison...")
pca = PCA(n_components=2, random_state=42)
pca_embedding = pca.fit_transform(svd_matrix.toarray() if hasattr(svd_matrix, "toarray") else svd_matrix)
colors = [row['star_color'] for row in raw_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=colors, s=2, alpha=0.5, edgecolors='none')
ax1.set_title("Linear Reduction: PCA (2D)", fontsize=14, color='white')
ax1.axis('off')

ax2.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=2, alpha=0.5, edgecolors='none')
ax2.set_title("Topological Reduction: UMAP (2D)", fontsize=14, color='white')
ax2.axis('off')

plt.tight_layout()
plt.savefig("plot_4_pca_vs_umap.png", dpi=300)
plt.close()

# === PLOT 5: K-NN Cosine Similarity Heatmap ===
print("-> Generating K-NN Search Heatmap...")
# Pick a random "Search Query" recipe to demonstrate
target_idx = 0 
target_vector = svd_matrix[target_idx].reshape(1, -1)
similarities = cosine_similarity(target_vector, svd_matrix)[0]

# Get top 10 matches
top_10_indices = np.argsort(similarities)[::-1][:10]
top_10_scores = similarities[top_10_indices]
top_10_names = [raw_data[i]['name'].title()[:30] + "..." for i in top_10_indices]

plt.figure(figsize=(10, 6))
# Reshape for heatmap
sns.heatmap(top_10_scores.reshape(10, 1), annot=True, cmap="mako", fmt=".3f", 
            yticklabels=top_10_names, xticklabels=["Cosine Score"], cbar=False)
plt.title(f"K-NN Vector Distances for: '{top_10_names[0]}'", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("plot_5_knn_heatmap.png", dpi=300)
plt.close()

print("Exporting Json...")
export_data = []

for i, row in enumerate(raw_data):
    clean_ingredients = ast.literal_eval(row['ingredients'])
    clean_steps = ast.literal_eval(row['steps'])
    
    export_data.append({
        'id': row['id'],
        'name': row['name'].title(),
        'x': float(embedding_3d[i, 0]),
        'y': float(embedding_3d[i, 1]),
        'z': float(embedding_3d[i, 2]),
        'galaxy_cluster': row['galaxy_cluster'],
        'star_color': row['star_color'],  
        'ingredients': [ing.capitalize() for ing in clean_ingredients],
        'steps': [step.capitalize() for step in clean_steps]
    })

with open("galaxy_data.json", mode="w", encoding="utf-8") as outfile:
    json.dump(export_data, outfile)

print("Pipeline Complete. Data and Plots successfully generated.")