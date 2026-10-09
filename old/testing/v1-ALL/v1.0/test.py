import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.express as px
import json

# Decoy Example Data
data = [
    {"id": 1, "name": "Spaghetti Marinara", "category": "Italian", "ingredients": ["pasta", "tomato", "garlic", "basil", "olive_oil"]},
    {"id": 2, "name": "Pesto Pasta", "category": "Italian", "ingredients": ["pasta", "basil", "garlic", "pine_nuts", "parmesan", "olive_oil"]},
    {"id": 3, "name": "Chicken Stir Fry", "category": "Asian", "ingredients": ["chicken", "soy_sauce", "garlic", "ginger", "broccoli", "oil"]},
    {"id": 4, "name": "Beef Teriyaki", "category": "Asian", "ingredients": ["beef", "soy_sauce", "sugar", "ginger", "rice"]},
    {"id": 5, "name": "Chocolate Cake", "category": "Dessert", "ingredients": ["flour", "sugar", "cocoa_powder", "eggs", "butter", "baking_soda"]},
    {"id": 6, "name": "Brownies", "category": "Dessert", "ingredients": ["flour", "sugar", "cocoa_powder", "eggs", "butter", "vanilla"]}
]
df = pd.DataFrame(data)

# TF-IDF preprocessing 
# so the vectorizer treats them as a single token.
df['ingredient_string'] = df['ingredients'].apply(lambda x: " ".join(x))

# max_df=0.95 removes ingredients that appear in >95% of recipes (e.g., salt, water)
# min_df=2 removes hyper-rare ingredients that only appear in 1 recipe
vectorizer = TfidfVectorizer(max_df=0.95, min_df=1) 
tfidf_matrix = vectorizer.fit_transform(df['ingredient_string'])

print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (Recipes x Unique Ingredients)")

# UMAP Dimensionality Reduction (Mapping to 3D space)
reducer = umap.UMAP(
    n_components=3,       # 3D
    n_neighbors=2,        # Controls local vs global structure
    min_dist=0.1,         # How tightly UMAP packs similar points together
    metric='cosine',      # The distance calculation method
    random_state=42       # For reproducibility during testing
)

embedding_3d = reducer.fit_transform(tfidf_matrix)

# 5. Append the new 3D coordinates back to your dataframe
df['x'] = embedding_3d[:, 0]
df['y'] = embedding_3d[:, 1]
df['z'] = embedding_3d[:, 2]

# 6. Visualize the 3D space interactively
fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='category',      # Colors the nodes by their category to verify clustering
    hover_name='name',     # Shows the recipe name when you hover
    title="3D Recipe Ingredient Map"
)
fig.show()

# 7. Export file
export_df = df[['id', 'name', 'x', 'y', 'z']]
export_df.to_json("recipeMap.json", orient="records")