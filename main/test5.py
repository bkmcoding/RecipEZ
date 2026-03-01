import csv
import ast
import json
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading data")
raw_data = []

with open("../data/RAW_recipes.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for i, row in enumerate(reader):
        if i >= 5000: break  # small for the local demo
        raw_data.append(row)


TAG_ONTOLOGY = {
    # Plant-Based (Greens)
    'vegan':       {'cluster': 'Vegan Sector',       'color': '#00ff00'}, 
    'vegetarian':  {'cluster': 'Vegetarian Sector',  'color': '#228b22'}, 
    
    # Baking & Sweets (Purples & Pinks)
    'dessert':     {'cluster': 'Dessert Core',       'color': '#ff00ff'}, 
    'baking':      {'cluster': 'Baking Sector',      'color': '#9370db'}, 
    'cookie':      {'cluster': 'Cookie Cluster',     'color': '#ffb6c1'}, 
    
    # World Cuisine (Warm Colors: Reds, Oranges, Yellows)
    'mexican':     {'cluster': 'Mexican Cuisine',    'color': '#ff4500'}, 
    'asian':       {'cluster': 'Asian Cuisine',      'color': '#ff8c00'}, 
    'indian':      {'cluster': 'Indian Cuisine',     'color': '#ffd700'}, 
    'italian':     {'cluster': 'Italian Cuisine',    'color': '#dc143c'}, 
    
    # Core Proteins (Blues)
    'seafood':     {'cluster': 'Seafood System',     'color': '#00ffff'}, 
    'poultry':     {'cluster': 'Poultry System',     'color': '#1e90ff'}, 
    'beef':        {'cluster': 'Beef System',        'color': '#000080'}, 
}

def assign_ontology(tags_list):
    """
    Scans a recipe's tags and returns its assigned Cluster Name and Color.
    """
    if not tags_list:
        return 'Deep Space', '#444444' # Dark grey for uncategorized
        
    t_str = " ".join([t.lower() for t in tags_list])
    
    for target_tag, properties in TAG_ONTOLOGY.items():
        if target_tag in t_str:
            return properties['cluster'], properties['color']
            
    return 'General Savory Space', '#444444'


prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']
master_features = []  

for row in raw_data:
    parsed_ingredients = ast.literal_eval(row['ingredients'])
    parsed_tags = ast.literal_eval(row['tags'])
    
    cluster_name, star_color = assign_ontology(parsed_tags)
    row['galaxy_cluster'] = cluster_name
    row['star_color'] = star_color
    
    # Clean ingredients
    ingreds = []
    for item in parsed_ingredients:
        for word in prep_words:
            item = item.replace(word, "")
        ingreds.append(item.strip().replace(" ", "_"))
    
    # Clean tags
    tags = ["TAG_" + tag.replace(" ", "_") for tag in parsed_tags]
    
    master_features.append(" ".join(ingreds + tags))



print("Vectorizing")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5)
tfidf_matrix = vectorizer.fit_transform(master_features)


print("Running UMAP projection")
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)

# UMAP returns a pure Numpy array (Shape: 5000 rows x 3 columns)
embedding_3d = reducer.fit_transform(tfidf_matrix)

print("Exporting JSON")
export_data = []

for i, row in enumerate(raw_data):
    # Safely convert the raw strings into actual Python lists
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
        # Capitalize the first letter of each ingredient and step
        'ingredients': [ing.capitalize() for ing in clean_ingredients],
        'steps': [step.capitalize() for step in clean_steps]
    })

with open("galaxy_data.json", mode="w", encoding="utf-8") as outfile:
    json.dump(export_data, outfile)

print("Data exported")