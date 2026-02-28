import ast
import pandas as pd
import umap
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# 5000 rows of data for testing
df = pd.read_csv("../data/RAW_recipes.csv") 
df = df.head(5000).copy()

# ast.literal_eval is a built-in Python library that safely evaluates strings containing Python expressions
df['parsed_ingredients'] = df['ingredients'].apply(ast.literal_eval)

# Removes spaces for easier feature creation
def format_for_tfidf(ingredient_list):
    return " ".join([item.replace(" ", "_") for item in ingredient_list])

df['tfidf_string'] = df['parsed_ingredients'].apply(format_for_tfidf)

# Test
print("Before:", df['ingredients'].iloc[0])
print("After: ", df['tfidf_string'].iloc[0])


# min_df=5 ignores ingredients that appear in fewer than 5 recipes (removes typos/weird data)
# max_df=0.95 ignores ubiquitous ingredients (like water or salt if they dominate)
vectorizer = TfidfVectorizer(max_df=0.95, min_df=5)
tfidf_matrix = vectorizer.fit_transform(df['tfidf_string'])

# Parse columns into
df['parsed_ingredients'] = df['ingredients'].apply(ast.literal_eval)
df['parsed_tags'] = df['tags'].apply(ast.literal_eval)

# Some manual cleanup
prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']

def build_master_feature_string(row):
    ingredients = []
    for item in row['parsed_ingredients']:
        for word in prep_words:
            item = item.replace(word, "")
        ingredients.append(item.strip().replace(" ", "_"))
    
    # cleans tagas
    tags = ["TAG_" + tag.replace(" ", "_") for tag in row['parsed_tags']]
    
    return " ".join(ingredients + tags)

# Apply the function across the dataframe
df['master_features'] = df.apply(build_master_feature_string, axis=1)


# Vectorize combined data
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5)
tfidf_matrix = vectorizer.fit_transform(df['master_features'])

# UMAP INIT
reducer = umap.UMAP(
    n_components=3,      # 3D
    n_neighbors=15,      # tightness of clusters
    min_dist=0.1,        # packing of nodes
    metric='cosine', 
    random_state=42
)

print("Running UMAP projection...")
embedding_3d = reducer.fit_transform(tfidf_matrix)

df['x'] = embedding_3d[:, 0]
df['y'] = embedding_3d[:, 1]
df['z'] = embedding_3d[:, 2]

# Color coding based on init tag
df['primary_tag'] = df['parsed_tags'].apply(lambda tags: tags[0] if tags else 'none')

# Render using plotly
fig = px.scatter_3d(
    df, 
    x='x', y='y', z='z',
    color='primary_tag',
    hover_name='name',
    hover_data={'x': False, 'y': False, 'z': False, 'primary_tag': False}, # Hide raw coordinates on hover
    title="3D Recipe Navigation Map (Ingredients + Tags)",
    opacity=0.7
)

# Shows plot on browser
fig.show()