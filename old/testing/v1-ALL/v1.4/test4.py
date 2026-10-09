import csv
import ast
import json
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer


raw_data = []

with open("../data/RAW_recipes.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for i, row in enumerate(reader):
        if i >= 5000: break  # Keep it small for the local demo
        raw_data.append(row)


# Improved visual categorization
print("Assigning galaxy clusters...")
def assign_galaxy_cluster(tags_list):
    if not tags_list: return 'Uncharted Space'
    t_str = " ".join([t.lower() for t in tags_list])

    if any(k in t_str for k in ['vegan']): return 'Vegan Cluster'
    if any(k in t_str for k in ['vegetarian']): return 'Vegetarian Cluster'
    if any(k in t_str for k in ['gluten-free']): return 'Gluten-Free Cluster'
    if any(k in t_str for k in ['asian', 'chinese', 'japanese', 'thai']): return 'Asian Cuisine'
    if any(k in t_str for k in ['mexican', 'southwestern']): return 'Mexican Cuisine'
    if any(k in t_str for k in ['italian']): return 'Italian Cuisine'
    if any(k in t_str for k in ['indian']): return 'Indian Cuisine'
    if any(k in t_str for k in ['dessert', 'baking', 'cake', 'cookie', 'brownie']): return 'Dessert Nebula'
    if any(k in t_str for k in ['breakfast', 'brunch', 'pancake']): return 'Breakfast System'
    if any(k in t_str for k in ['beverages', 'smoothie', 'drink']): return 'Beverage System'
    if any(k in t_str for k in ['seafood', 'fish', 'shrimp']): return 'Seafood Sector'
    if any(k in t_str for k in ['poultry', 'chicken', 'turkey']): return 'Poultry Sector'
    if any(k in t_str for k in ['beef', 'pork', 'meat']): return 'Meat Sector'
    if any(k in t_str for k in ['pasta']): return 'Pasta Sector'
    return 'General Savory Space'



# Extra bloat removal
prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']
master_features = []  

for row in raw_data:
    parsed_ingredients = ast.literal_eval(row['ingredients'])
    parsed_tags = ast.literal_eval(row['tags'])
    row['galaxy_cluster'] = assign_galaxy_cluster(parsed_tags)
    
    ingreds = []
    for item in parsed_ingredients:
        for word in prep_words:
            item = item.replace(word, "")
        ingreds.append(item.strip().replace(" ", "_"))
    
    tags = ["TAG_" + tag.replace(" ", "_") for tag in parsed_tags]
    master_features.append(" ".join(ingreds + tags))


print("Vectorizing...")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5)
tfidf_matrix = vectorizer.fit_transform(master_features)



print("Running UMAP projection...")
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)

embedding_3d = reducer.fit_transform(tfidf_matrix)


print("Exporting Mega-JSON for local web demo...")
export_data = []


for i, row in enumerate(raw_data):
    export_data.append({
        'id': row['id'],
        'name': row['name'],
        'x': float(embedding_3d[i, 0]),
        'y': float(embedding_3d[i, 1]),
        'z': float(embedding_3d[i, 2]),
        'galaxy_cluster': row['galaxy_cluster'],
        'steps': row['steps'],
        'ingredients': row['ingredients']
    })

with open("galaxy_data.json", mode="w", encoding="utf-8") as outfile:
    json.dump(export_data, outfile)

print("Data ready for the web!")