import csv
import ast
import json
import random
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("Loading data...")
all_data = []

with open("../RAW_recipeTest/RAW_recipes.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        all_data.append(row)

SAMPLE_SIZE = 30000
print(f"Loaded {len(all_data)} total recipes. Randomly sampling {SAMPLE_SIZE}...")
random.seed(42)
raw_data = random.sample(all_data, SAMPLE_SIZE)

print("Mapping color tags...")

TAG_ONTOLOGY = {
    # Plant-Based (Greens)
    'vegan':       {'cluster': 'Vegan Sector',       'color': '#00ff00'}, 
    'vegetarian':  {'cluster': 'Vegetarian Sector',  'color': '#228b22'}, 
    'salad':       {'cluster': 'Salad System',       'color': '#7cfc00'}, 
    
    # Baking & Sweets (Purples & Pinks)
    'dessert':     {'cluster': 'Dessert Core',       'color': '#ff00ff'}, 
    'baking':      {'cluster': 'Baking Sector',      'color': '#9370db'}, 
    'cookie':      {'cluster': 'Cookie Cluster',     'color': '#ffb6c1'}, 
    'cake':        {'cluster': 'Cake Nebula',        'color': '#da70d6'}, 
    
    # World Cuisine (Warm Colors: Reds, Oranges, Yellows)
    'mexican':     {'cluster': 'Mexican Cuisine',    'color': '#ff4500'}, 
    'asian':       {'cluster': 'Asian Cuisine',      'color': '#ff8c00'}, 
    'indian':      {'cluster': 'Indian Cuisine',     'color': '#ffd700'}, 
    'italian':     {'cluster': 'Italian Cuisine',    'color': '#dc143c'}, 
    'french':      {'cluster': 'French Cuisine',     'color': '#c71585'}, 
    
    # Core Proteins (Blues & Pinks)
    'seafood':     {'cluster': 'Seafood System',     'color': '#00ffff'}, 
    'poultry':     {'cluster': 'Poultry System',     'color': '#1e90ff'}, 
    'beef':        {'cluster': 'Beef System',        'color': '#000080'}, 
    'pork':        {'cluster': 'Pork System',        'color': '#ff69b4'}, 
    
    # Soups & Stews (Earthy/Deep Tones)
    'soup':        {'cluster': 'Soup Sector',        'color': '#d2691e'}, 
    'stew':        {'cluster': 'Stew Cluster',       'color': '#8b4513'}, 
    
    # Breakfast & Brunch (Bright Mornings)
    'breakfast':   {'cluster': 'Breakfast Belt',     'color': '#ffffe0'}, 
    'brunch':      {'cluster': 'Brunch Belt',        'color': '#ffebcd'}, 
    
    # Beverages (Aquas/Teals)
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

    return 'Untagged Core', '#f4f6f8'


print("Building feature strings and settings colors...")
prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']
master_features = []  

for row in raw_data:
    parsed_ingredients = ast.literal_eval(row['ingredients'])
    parsed_tags = ast.literal_eval(row['tags'])
    
    cluster_name, star_color = assign_ontology(parsed_tags)
    row['galaxy_cluster'] = cluster_name
    row['star_color'] = star_color
    
    ingreds = []
    for item in parsed_ingredients:
        for word in prep_words:
            item = item.replace(word, "")
        ingreds.append(item.strip().replace(" ", "_"))
    
    tags = ["TAG_" + tag.replace(" ", "_") for tag in parsed_tags]
    
    master_features.append(" ".join(ingreds + tags))

print("TF-IDF Vectorizing...")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, max_features=10000)
tfidf_matrix = vectorizer.fit_transform(master_features)

print(f"{tfidf_matrix.shape[1]} Dimension Reduction with SVD...")
svd = TruncatedSVD(n_components=100, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

print("UMAP projecting...")
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_3d = reducer.fit_transform(svd_matrix)

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

print("Done")