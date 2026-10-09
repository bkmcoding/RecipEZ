



import csv
import ast
import json
import random
import numpy as np
import networkx as nx
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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

reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_2d = reducer_2d.fit_transform(svd_matrix)

plt.figure(figsize=(12, 10), facecolor='#0a0a0c') # Match your Typst dark mode
ax = plt.gca()
ax.set_facecolor('#0a0a0c')

# Scatter plot using the star_colors you already defined in your TAG_ONTOLOGY
colors = [row['star_color'] for row in raw_data]

plt.scatter(
    embedding_2d[:, 0], 
    embedding_2d[:, 1], 
    c=colors, 
    s=1,          # Small dots for that "Galaxy" feel
    alpha=0.6, 
    edgecolors='none'
)

# --- ADDED DENSITY VISUALIZATION BLOCK ---
print("Generating density map for documentation...")

plt.figure(figsize=(12, 10), facecolor='#0a0a0c')
ax = plt.gca()
ax.set_facecolor('#0a0a0c')

# Using seaborn to create a 2D density heatmap
sns.kdeplot(
    x=embedding_2d[:, 0], 
    y=embedding_2d[:, 1], 
    fill=True, 
    thresh=0, 
    levels=100, 
    cmap="mako", # A teal-to-black gradient fits your raw/brutalist theme
    alpha=0.8
)

plt.title("RecipEZ: Cluster Density & Radius of Influence", color='white', fontsize=16)
plt.axis('off')
plt.savefig("umap_radius.png", dpi=300, bbox_inches='tight', facecolor='#0a0a0c')
print("Density map saved as umap_radius.png")

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

BG = '#0a0a0c'
TEAL = '#00e5cc'
PINK = '#ff2d78'
GREEN = '#39ff14'
AMBER = '#ffaa00'
GRAY = '#3a3a3c'
LGRAY = '#888888'
WHITE = '#e8e8e8'

plt.rcParams.update({
    'font.family': 'monospace',
    'text.color': WHITE,
    'axes.labelcolor': LGRAY,
    'xtick.color': LGRAY,
    'ytick.color': LGRAY,
    'axes.edgecolor': GRAY,
    'grid.color': GRAY,
    'grid.alpha': 0.3,
    'figure.facecolor': BG,
    'axes.facecolor': BG,
})

def savefig(name):
    plt.savefig(f"{name}.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved {name}.png")


# ── VIZ 1: IDF DECAY CURVE ────────────────────────────────────────────────────
idf_vals = vectorizer.idf_
vocab = vectorizer.get_feature_names_out()
order = np.argsort(idf_vals)
sorted_idf = idf_vals[order]
sorted_vocab = vocab[order]
N = len(sorted_idf)

ANNOTATE = [int(N * p) for p in [0.01, 0.04, 0.10, 0.25, 0.50, 0.70, 0.85, 0.93, 0.98]]

fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.fill_between(range(N), sorted_idf, alpha=0.08, color=TEAL)
ax.plot(sorted_idf, color=TEAL, linewidth=1.2, alpha=0.9)
ax.axvspan(0, int(N * 0.15), alpha=0.06, color=PINK)
ax.axvspan(int(N * 0.80), N, alpha=0.06, color=GREEN)
ax.text(int(N * 0.07), sorted_idf.max() * 0.92, 'PENALTY\nZONE', color=PINK,
        fontsize=7.5, ha='center', alpha=0.8)
ax.text(int(N * 0.90), sorted_idf.max() * 0.92, 'SIGNAL\nZONE', color=GREEN,
        fontsize=7.5, ha='center', alpha=0.8)

for idx in ANNOTATE:
    word = sorted_vocab[idx]
    yval = sorted_idf[idx]
    offset_y = 0.3 if idx < N * 0.5 else -0.4
    ax.annotate(word, xy=(idx, yval), xytext=(idx, yval + offset_y),
        color=WHITE, fontsize=7.5, ha='center',
        arrowprops=dict(arrowstyle='->', color=TEAL, lw=0.8),
        bbox=dict(boxstyle='round,pad=0.2', fc=BG, ec=GRAY, lw=0.5))

ax.set_xlabel('Ingredient rank (ascending IDF)', fontsize=9)
ax.set_ylabel('IDF score', fontsize=9)
ax.set_title('VIZ 1 — Ingredient Gravity Spectrum: IDF Decay Curve', color=WHITE, fontsize=11, pad=14)
ax.spines[['top', 'right']].set_visible(False)
savefig('viz1_idf_decay')


# ── VIZ 2: SVD LATENT FLAVOR PROFILES ─────────────────────────────────────────
vocab_arr = vectorizer.get_feature_names_out()
TOP_N = 10

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
fig.suptitle('VIZ 2 — The Hidden Palate: SVD Latent Flavor Components',
             color=WHITE, fontsize=12, y=1.01)

for i in range(6):
    ax = axes[i // 3][i % 3]
    ax.set_facecolor(BG)
    loadings = svd.components_[i]
    top_pos = np.argsort(loadings)[-TOP_N:]
    top_neg = np.argsort(loadings)[:TOP_N]
    idxs = np.concatenate([top_neg, top_pos])
    vals = loadings[idxs]
    labels = vocab_arr[idxs]
    colors = [PINK if v < 0 else TEAL for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.7, alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_title(f'Component {i + 1}', color=LGRAY, fontsize=8.5)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', labelsize=6)

fig.tight_layout()
savefig('viz2_svd_flavor_profiles')


# ── VIZ 3: UMAP FIDELITY SCATTER ──────────────────────────────────────────────
SAMPLE = 2500
rng = np.random.default_rng(42)
idx_sample = rng.choice(len(svd_matrix), size=SAMPLE, replace=False)
svd_sub = svd_matrix[idx_sample]
umap_sub = embedding_2d[idx_sample]

cos_dists = cosine_distances(svd_sub)
euc_dists = np.sqrt(((umap_sub[:, None] - umap_sub[None, :]) ** 2).sum(-1))

mask = np.triu(np.ones((SAMPLE, SAMPLE), dtype=bool), k=1)
cos_flat = cos_dists[mask]
euc_flat = euc_dists[mask]

ss = rng.choice(len(cos_flat), size=15000, replace=False)
cx = cos_flat[ss]
ey = euc_flat[ss]

norm_cx = (cx - cx.min()) / (cx.max() - cx.min())
point_colors = plt.cm.cool(norm_cx)

fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(cx, ey, c=point_colors, s=1.5, alpha=0.25, edgecolors='none')

sort_order = np.argsort(cx)
smooth_y = np.convolve(ey[sort_order], np.ones(500) / 500, mode='valid')
ax.plot(cx[sort_order][250:-249], smooth_y, color=AMBER, linewidth=1.6, alpha=0.9, label='Moving avg')

ax.set_xlabel('Cosine distance (100D SVD space)', fontsize=9)
ax.set_ylabel('Euclidean distance (2D UMAP space)', fontsize=9)
ax.set_title('VIZ 3 — UMAP Fidelity Witness: High-D vs Low-D Distance Preservation',
             color=WHITE, fontsize=11, pad=14)
ax.legend(fontsize=8, facecolor=BG, edgecolor=GRAY, labelcolor=WHITE)
ax.spines[['top', 'right']].set_visible(False)
savefig('viz3_umap_fidelity')


# ── VIZ 4: K-NN QUERY ANATOMY ─────────────────────────────────────────────────
QUERY = "eggs bacon cheddar sourdough butter"
K = 8

query_tfidf = vectorizer.transform([QUERY])
query_svd = svd.transform(query_tfidf)
cos_to_query = cosine_distances(query_svd, svd_matrix)[0]
top_k_idx = np.argsort(cos_to_query)[:K]
qx, qy = embedding_2d[top_k_idx[0]]

fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
           c='#1a1a1c', s=0.8, alpha=0.5, edgecolors='none', zorder=1)

for r, alpha in [(2.5, 0.08), (4.5, 0.05), (7.0, 0.03)]:
    ax.add_patch(mpatches.Circle((qx, qy), r, fill=True, color=TEAL, alpha=alpha, zorder=2))
    ax.add_patch(mpatches.Circle((qx, qy), r, fill=False, edgecolor=TEAL,
                                  linewidth=0.6, alpha=0.35, linestyle='--', zorder=3))

ax.scatter(embedding_2d[top_k_idx, 0], embedding_2d[top_k_idx, 1],
           c=PINK, s=40, zorder=5, edgecolors=WHITE, linewidth=0.5)
ax.scatter([qx], [qy], marker='*', s=280, c=TEAL, zorder=6, edgecolors=WHITE, linewidth=0.5)

for rank, idx in enumerate(top_k_idx):
    name = raw_data[idx]['name'].title()[:28]
    score = cos_to_query[idx]
    ax.annotate(f'#{rank+1} {name}\n cos={score:.3f}',
        xy=(embedding_2d[idx, 0], embedding_2d[idx, 1]),
        xytext=(embedding_2d[idx, 0] + 0.5, embedding_2d[idx, 1] + 0.4),
        fontsize=6.5, color=WHITE,
        arrowprops=dict(arrowstyle='->', color=PINK, lw=0.7),
        bbox=dict(boxstyle='round,pad=0.2', fc=BG, ec=GRAY, lw=0.4), zorder=7)

ax.set_title(f'VIZ 4 — Search Oracle: K-NN Query Anatomy\nQuery: "{QUERY}"',
             color=WHITE, fontsize=11, pad=14)
ax.axis('off')
savefig('viz4_knn_anatomy')


# ── VIZ 5: TAG PURITY MATRIX ──────────────────────────────────────────────────
N_CLUSTERS = 12
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
spatial_labels = km.fit_predict(embedding_2d)

TAG_COLS = ['vegan', 'vegetarian', 'dessert', 'baking', 'mexican',
            'asian', 'indian', 'italian', 'seafood', 'soup', 'breakfast', 'beef']

purity_rows = []
for c in range(N_CLUSTERS):
    mask_c = np.where(spatial_labels == c)[0]
    purity_rows.append({
        tag: sum(1 for i in mask_c if tag in master_features[i].lower()) / max(len(mask_c), 1)
        for tag in TAG_COLS
    })

purity_df = pd.DataFrame(purity_rows, columns=TAG_COLS,
                          index=[f'Cluster {i}' for i in range(N_CLUSTERS)])

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor(BG)
sns.heatmap(purity_df, ax=ax, cmap='mako', annot=True, fmt='.0%',
            annot_kws={'size': 7, 'color': WHITE},
            linewidths=0.3, linecolor=BG, cbar_kws={'shrink': 0.6})
ax.set_title('VIZ 5 — Galaxy Cluster Purity: Spatial Clusters vs Culinary Tags',
             color=WHITE, fontsize=11, pad=14)
ax.tick_params(colors=LGRAY, labelsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
savefig('viz5_purity_matrix')


# ── VIZ 6: PIPELINE TRACE (PARALLEL COORDINATES) ─────────────────────────────
CATEGORIES = ['Dessert Core', 'Mexican Cuisine', 'Seafood System',
              'Vegan Sector', 'Soup Sector', 'Breakfast Belt',
              'Italian Cuisine', 'Asian Cuisine', 'Baking Sector', 'Beef System']
CAT_COLORS = {
    'Dessert Core': '#ff00ff', 'Mexican Cuisine': '#ff4500',
    'Seafood System': '#00ffff', 'Vegan Sector': '#00ff00',
    'Soup Sector': '#d2691e', 'Breakfast Belt': '#ffffaa',
    'Italian Cuisine': '#dc143c', 'Asian Cuisine': '#ff8c00',
    'Baking Sector': '#9370db', 'Beef System': '#4169e1',
}

selected = []
for cat in CATEGORIES:
    idxs_cat = [i for i, r in enumerate(raw_data) if r['galaxy_cluster'] == cat]
    selected.extend([(i, cat) for i in idxs_cat[:3]])

axes_labels = ['Ingredient\nCount', 'TF-IDF\nL2 Norm', 'SVD Comp.1', 'SVD Comp.2', 'UMAP X', 'UMAP Y']

trace_data = []
for idx, cat in selected:
    trace_data.append({
        'vals': [
            len(raw_data[idx].get('ingredients', [])),
            float(np.linalg.norm(tfidf_matrix[idx].toarray())),
            svd_matrix[idx, 0],
            svd_matrix[idx, 1],
            embedding_2d[idx, 0],
            embedding_2d[idx, 1],
        ],
        'cat': cat, 'color': CAT_COLORS.get(cat, WHITE)
    })

raw_vals = np.array([d['vals'] for d in trace_data])
mins = raw_vals.min(axis=0)
maxs = raw_vals.max(axis=0)
normed = (raw_vals - mins) / (maxs - mins + 1e-9)
n_axes = len(axes_labels)

fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
ax.set_facecolor(BG)

for i, d in enumerate(trace_data):
    ax.plot(range(n_axes), normed[i], color=d['color'], alpha=0.45,
            linewidth=1.1, solid_capstyle='round')

for xi in range(n_axes):
    ax.axvline(xi, color=GRAY, linewidth=0.6, alpha=0.5)

ax.set_xticks(range(n_axes))
ax.set_xticklabels(axes_labels, fontsize=8, color=LGRAY)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(['min', 'mid', 'max'], fontsize=7, color=LGRAY)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_title('VIZ 6 — Pipeline Trace: Recipe Transformation Across Stages',
             color=WHITE, fontsize=11, pad=14)
legend_patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATEGORIES]
ax.legend(handles=legend_patches, fontsize=6.5, loc='upper right',
          facecolor=BG, edgecolor=GRAY, labelcolor=WHITE, ncol=2)
savefig('viz6_pipeline_trace')


# ── VIZ 7: INGREDIENT CO-OCCURRENCE CONSTELLATION ────────────────────────────
TOP_INGREDS = 120

binary = (tfidf_matrix > 0).astype(np.float32)
doc_counts = np.asarray(binary.sum(axis=0)).flatten()
cooc = (binary.T @ binary).toarray()
np.fill_diagonal(cooc, 0)

with np.errstate(divide='ignore', invalid='ignore'):
    pmi = cooc / (doc_counts[:, None] * doc_counts[None, :])
    pmi = np.nan_to_num(pmi, 0)

top_idx = np.argsort(doc_counts)[-TOP_INGREDS:]
vocab_top = vocab_arr[top_idx]
sub_pmi = pmi[np.ix_(top_idx, top_idx)]

all_weights = sub_pmi[np.triu_indices(TOP_INGREDS, k=1)]
all_weights = all_weights[all_weights > 0]
MIN_PMI = float(np.percentile(all_weights, 75)) if len(all_weights) > 0 else 0.0
print(f"Adaptive MIN_PMI threshold: {MIN_PMI:.4f}")

G = nx.Graph()
G.add_nodes_from(vocab_top)
for i in range(TOP_INGREDS):
    for j in range(i + 1, TOP_INGREDS):
        w = sub_pmi[i, j]
        if w > MIN_PMI:
            G.add_edge(vocab_top[i], vocab_top[j], weight=float(w))

degrees = dict(G.degree())
top_keep = sorted(degrees, key=degrees.get, reverse=True)[:80]
G = G.subgraph(top_keep).copy()
pos = nx.spring_layout(G, seed=42, k=2.2, weight='weight')

fig, ax = plt.subplots(figsize=(14, 14), facecolor=BG)
ax.set_facecolor(BG)

edges = list(G.edges(data=True))
w_arr = np.array([d['weight'] for _, _, d in edges]) if edges else np.array([1.0])
w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)

for (u, v, d), wn in zip(edges, w_norm):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                           width=wn * 2.0, alpha=float(wn * 0.6), edge_color=TEAL)

node_sizes = [degrees.get(n, 1) * 18 for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                       node_color=PINK, alpha=0.85, edgecolors=BG, linewidths=0.5)

for node, (x, y) in pos.items():
    fs = max(5.5, min(9.5, 5.5 + degrees.get(node, 1) * 0.4))
    ax.text(x, y + 0.04, node, fontsize=fs, color=WHITE,
            ha='center', va='bottom', alpha=0.9)

ax.set_title('VIZ 7 — Ingredient Constellation: Co-Occurrence Force Graph',
             color=WHITE, fontsize=11, pad=14)
ax.axis('off')
savefig('viz7_cooccurrence_graph')

print("\nAll 7 diagnostics complete.")