import ast
import pandas as pd
import umap
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# Load
print("Loading data...")
df = pd.read_csv("../RAW_recipes.csv").head(5000) 

print("Parsing lists...")
df['parsed_ingredients'] = df['ingredients'].apply(ast.literal_eval)
df['parsed_tags'] = df['tags'].apply(ast.literal_eval)

# Setting tag clusters for sectioning and coloring the graph/clusters.
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

# Apply categories
df['galaxy_cluster'] = df['parsed_tags'].apply(assign_galaxy_cluster)

print("Building feature strings...")
prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']

def build_master_feature_string(row):
    ingreds = []
    for item in row['parsed_ingredients']:
        for word in prep_words:
            item = item.replace(word, "")
        ingreds.append(item.strip().replace(" ", "_"))
    
    tags = ["TAG_" + tag.replace(" ", "_") for tag in row['parsed_tags']]
    return " ".join(ingreds + tags)

df['master_features'] = df.apply(build_master_feature_string, axis=1)

print("Vectorizing")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5)
tfidf_matrix = vectorizer.fit_transform(df['master_features'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# UMAP
print("Running UMAP projection...")
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_3d = reducer.fit_transform(tfidf_matrix)

df['x'] = embedding_3d[:, 0]
df['y'] = embedding_3d[:, 1]
df['z'] = embedding_3d[:, 2]

# 3D plot
print("Rendering plot...")
fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='galaxy_cluster',
    hover_name='name',
    hover_data={'x': False, 'y': False, 'z': False, 'galaxy_cluster': False},
    title="3D Recipe Navigation Map",
    opacity=0.8
)
fig.show()

# Json for plotly
print("Exporting to JSON...")
export_df = df[['id', 'name', 'x', 'y', 'z', 'galaxy_cluster']]
export_df.to_json("recipe_3d_coordinates.json", orient="records")