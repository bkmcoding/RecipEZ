import csv
import ast
import json
from pathlib import Path
import random
import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "RAW_recipes.csv"
OUTPUT_DIR = ROOT / "web" / "model_data"

print("Loading data...")
all_data = []
with open(RAW_DATA_PATH, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        all_data.append(row)

SAMPLE_SIZE = 30000
print(f"Randomly sampling {SAMPLE_SIZE} recipes...")
random.seed(42) 
raw_data = random.sample(all_data, SAMPLE_SIZE)

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
    if not tags_list: return 'Uncharted Stars', '#fdfbf7'
    t_str = " ".join([t.lower() for t in tags_list])
    for target_tag, properties in TAG_ONTOLOGY.items():
        if target_tag in t_str: return properties['cluster'], properties['color']
    return 'Main Sequence', '#f4f6f8'

def normalize_galaxy(embeddings):
    centered = embeddings - np.mean(embeddings, axis=0)
    max_distance = np.max(np.abs(centered))
    return centered / max_distance

print("Building features...")
prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']
features_all = []
features_ingredients = []
features_names = []

for row in raw_data:
    parsed_ingredients = ast.literal_eval(row['ingredients'])
    parsed_tags = ast.literal_eval(row['tags'])
    
    cluster_name, star_color = assign_ontology(parsed_tags)
    row['galaxy_cluster'] = cluster_name
    row['star_color'] = star_color
    
    ingreds = [item.replace(word, "") for item in parsed_ingredients for word in prep_words if word in item] or parsed_ingredients
    tags = [tag.replace(" ", "_") for tag in parsed_tags]
    
    features_all.append(f"{row['name']} ingredients: {', '.join(ingreds)}. tags: {' '.join(tags)}")
    features_ingredients.append(", ".join(ingreds))
    features_names.append(row['name'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device.upper()}...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def run_sbert_pipeline(feature_list):
    print(f"Encoding {len(feature_list)} items...")
    embeddings = model.encode(feature_list, show_progress_bar=True)
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)
    embedding_3d = normalize_galaxy(embedding_3d) 
    
    print("Running KNN...")
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(embeddings)
    _, indices = knn.kneighbors(embeddings)
    
    return embedding_3d, indices

print("\nSBERT combined")
coords_all, knn_all = run_sbert_pipeline(features_all)

print("\nSBERT ingredients")
coords_ingred, knn_ingred = run_sbert_pipeline(features_ingredients)

print("\nSBERT names")
coords_name, knn_name = run_sbert_pipeline(features_names)

print("\nTF-IDF")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, max_features=10000)
tfidf_matrix = vectorizer.fit_transform(features_all)

svd = TruncatedSVD(n_components=100, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

reducer_tfidf = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
coords_tfidf = reducer_tfidf.fit_transform(svd_matrix)
coords_tfidf = normalize_galaxy(coords_tfidf) 

knn_tfidf = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_tfidf.fit(svd_matrix)
_, knn_tfidf_indices = knn_tfidf.kneighbors(svd_matrix)


print("\nExporting JSONs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
export_sbert_all = []
export_sbert_ingred = []
export_sbert_name = []
export_tfidf = []

for i, row in enumerate(raw_data):
    clean_ingredients = ast.literal_eval(row['ingredients'])
    clean_steps = ast.literal_eval(row['steps'])
    
    base_recipe = {
        'id': row['id'],
        'name': row['name'].title(),
        'galaxy_cluster': row['galaxy_cluster'],
        'star_color': row['star_color'],  
        'ingredients': [ing.capitalize() for ing in clean_ingredients],
        'steps': [step.capitalize() for step in clean_steps],
    }
    
    node_all = base_recipe.copy()
    node_all['x'], node_all['y'], node_all['z'] = float(coords_all[i,0]), float(coords_all[i,1]), float(coords_all[i,2])
    node_all['similar_recipes'] = [raw_data[idx]['id'] for idx in knn_all[i][1:]]
    export_sbert_all.append(node_all)
    
    
    node_ingred = base_recipe.copy()
    node_ingred['x'], node_ingred['y'], node_ingred['z'] = float(coords_ingred[i,0]), float(coords_ingred[i,1]), float(coords_ingred[i,2])
    node_ingred['similar_recipes'] = [raw_data[idx]['id'] for idx in knn_ingred[i][1:]]
    export_sbert_ingred.append(node_ingred)
    
    
    node_name = base_recipe.copy()
    node_name['x'], node_name['y'], node_name['z'] = float(coords_name[i,0]), float(coords_name[i,1]), float(coords_name[i,2])
    node_name['similar_recipes'] = [raw_data[idx]['id'] for idx in knn_name[i][1:]]
    export_sbert_name.append(node_name)
    
    
    node_tfidf = base_recipe.copy()
    node_tfidf['x'], node_tfidf['y'], node_tfidf['z'] = float(coords_tfidf[i,0]), float(coords_tfidf[i,1]), float(coords_tfidf[i,2])
    node_tfidf['similar_recipes'] = [raw_data[idx]['id'] for idx in knn_tfidf_indices[i][1:]]
    export_tfidf.append(node_tfidf)

with open(OUTPUT_DIR / "sbert_all_data.json", "w", encoding="utf-8") as f: json.dump(export_sbert_all, f)
with open(OUTPUT_DIR / "sbert_ingredients_data.json", "w", encoding="utf-8") as f: json.dump(export_sbert_ingred, f)
with open(OUTPUT_DIR / "sbert_names_data.json", "w", encoding="utf-8") as f: json.dump(export_sbert_name, f)
with open(OUTPUT_DIR / "tfidf_data.json", "w", encoding="utf-8") as f: json.dump(export_tfidf, f)

print("Done")
