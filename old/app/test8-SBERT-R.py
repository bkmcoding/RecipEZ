import csv
import ast
import json
import numpy as np
import umap
from sentence_transformers import SentenceTransformer

print("Loading data...")
raw_data = []

with open("../data/RAW_recipes.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for i, row in enumerate(reader):
        # if i >= 30000: break 
        raw_data.append(row)

print("Mapping tags to relational color ontology...")
TAG_ONTOLOGY = {
    'vegan':      {'cluster': 'Vegan Sector',      'color': '#00ff00'}, 
    'vegetarian': {'cluster': 'Vegetarian Sector', 'color': '#228b22'}, 
    'salad':      {'cluster': 'Salad System',      'color': '#7cfc00'}, 
    'dessert':    {'cluster': 'Dessert Core',      'color': '#ff00ff'}, 
    'baking':     {'cluster': 'Baking Sector',     'color': '#9370db'}, 
    'cookie':     {'cluster': 'Cookie Cluster',    'color': '#ffb6c1'}, 
    'cake':       {'cluster': 'Cake Nebula',       'color': '#da70d6'}, 
    'mexican':    {'cluster': 'Mexican Cuisine',   'color': '#ff4500'}, 
    'asian':      {'cluster': 'Asian Cuisine',     'color': '#ff8c00'}, 
    'indian':     {'cluster': 'Indian Cuisine',    'color': '#ffd700'}, 
    'italian':    {'cluster': 'Italian Cuisine',   'color': '#dc143c'}, 
    'french':     {'cluster': 'French Cuisine',    'color': '#c71585'}, 
    'seafood':    {'cluster': 'Seafood System',    'color': '#00ffff'}, 
    'poultry':    {'cluster': 'Poultry System',    'color': '#1e90ff'}, 
    'beef':       {'cluster': 'Beef System',       'color': '#000080'}, 
    'pork':       {'cluster': 'Pork System',       'color': '#ff69b4'}, 
    'soup':       {'cluster': 'Soup Sector',       'color': '#d2691e'}, 
    'stew':       {'cluster': 'Stew Cluster',      'color': '#8b4513'}, 
    'breakfast':  {'cluster': 'Breakfast Belt',    'color': '#ffffe0'}, 
    'brunch':     {'cluster': 'Brunch Belt',       'color': '#ffebcd'}, 
    'beverages':  {'cluster': 'Beverage Ocean',    'color': '#7fffd4'}, 
    'cocktails':  {'cluster': 'Cocktail Nebula',   'color': '#40e0d0'}, 
}

def assign_ontology(tags_list):
    if not tags_list:
        return 'Uncharted Stars', '#adaba7'
        
    t_str = " ".join([t.lower() for t in tags_list])
    
    for target_tag, properties in TAG_ONTOLOGY.items():
        if target_tag in t_str:
            return properties['cluster'], properties['color']

    return 'Main Sequence', '#a2a6a8'   

print("Building feature strings and assigning colors...")
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
        ingreds.append(item.strip())
    
    recipe_name = str(row['name']).strip().title()
    feature_string = f"Recipe for {recipe_name}. It features {', '.join(ingreds)}. Tags include {', '.join(parsed_tags)}."
    
    master_features.append(feature_string)

print("using SBERT...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
embeddings = model.encode(master_features, show_progress_bar=True)

print("UMAP projecting...")
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_3d = reducer.fit_transform(embeddings)

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