import ast
import pandas as pd
import umap
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



print("Loading data...")
df = pd.read_csv("../RAW_recipes.csv").head(5000) 

print("Parsing lists...")
df['parsed_ingredients'] = df['ingredients'].apply(ast.literal_eval)
df['parsed_tags'] = df['tags'].apply(ast.literal_eval)


print("Assigning clusters")
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

df['galaxy_cluster'] = df['parsed_tags'].apply(assign_galaxy_cluster)


print("Building feature strings")
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



print("Running UMAP projection")
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_3d = reducer.fit_transform(tfidf_matrix)

df['x'] = embedding_3d[:, 0]
df['y'] = embedding_3d[:, 1]
df['z'] = embedding_3d[:, 2]



print("Rendering map...")
glow_palette = px.colors.qualitative.Vivid 

fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='galaxy_cluster',   # Group by new system
    color_discrete_sequence=glow_palette,
    hover_name='name', 
    hover_data={'x': False, 'y': False, 'z': False, 'galaxy_cluster': False},
    title="Recipe Galaxy Map"
)

# Strip away the math graph aesthetics to create empty space
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="black",    
    scene=dict(
        bgcolor="black",      
        xaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title="", zeroline=False),
        yaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title="", zeroline=False),
        zaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title="", zeroline=False)
    ),
    margin=dict(l=0, r=0, b=0, t=40) 
)

# Make the recipes look like stars
fig.update_traces(
    marker=dict(size=3, opacity=0.8, line=dict(width=0))
)

fig.show()



print("Exporting to JSON")
export_df = df[['id', 'name', 'x', 'y', 'z', 'galaxy_cluster']]
export_df.to_json("recipe_3d_coordinates.json", orient="records")
print("Pipeline complete!")



def search_galaxy(user_input_string, vectorizer_model, feature_matrix, dataframe, top_k=3):
    print(f"\n--- Searching the Galaxy for: '{user_input_string}' ---")
    
    user_vector = vectorizer_model.transform([user_input_string])
    
    # Calculate similarity between the user and every recipe
    similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()
    
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    
    for i, idx in enumerate(top_indices):
        recipe = dataframe.iloc[idx]
        score = similarity_scores[idx]
        
        # Print if match score above threshold
        if score > 0.0:
            print(f"\nMatch {i+1}: {recipe['name'].title()} (Match Score: {score:.2f})")
            print(f"Cluster: {recipe['galaxy_cluster']}")
            print(f"Coordinates -> X: {recipe['x']:.2f}, Y: {recipe['y']:.2f}, Z: {recipe['z']:.2f}")
        else:
            print(f"\nMatch {i+1}: No strong matches found for these ingredients.")

# Testing galaxy search
search_galaxy("chicken garlic soy sauce ginger", vectorizer, tfidf_matrix, df)
search_galaxy("flour sugar eggs butter chocolate", vectorizer, tfidf_matrix, df)
search_galaxy("ground beef kidney beans chili powder", vectorizer, tfidf_matrix, df)