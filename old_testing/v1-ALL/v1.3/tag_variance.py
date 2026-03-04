import csv
import ast
import numpy as np
import umap
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def run_evaluation():
    print("Loading raw data for testing...")
    raw_data = []
    all_tags_raw = []
    
    with open("../RAW_recipes.csv", mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i >= 5000: break  # Keep the test set identical to the demo size
            raw_data.append(row)
            all_tags_raw.extend(ast.literal_eval(row['tags']))

    print("Building feature matrix...")
    prep_words = ['diced ', 'chopped ', 'crushed ', 'minced ', 'sliced ', 'ground ']
    master_features = []
    
    for row in raw_data:
        parsed_ingredients = ast.literal_eval(row['ingredients'])
        parsed_tags = ast.literal_eval(row['tags'])
        
        ingreds = [item.replace(w, "") for item in parsed_ingredients for w in prep_words]
        ingreds = [item.strip().replace(" ", "_") for item in ingreds]
        tags = ["TAG_" + tag.replace(" ", "_") for tag in parsed_tags]
        
        master_features.append(" ".join(ingreds + tags))

    print("Vectorizing and calculating 3D UMAP space...")
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(master_features)
    
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_3d = reducer.fit_transform(tfidf_matrix)

    print("Calculating Intra-Cluster Variance for Top 50 Tags...")
    tag_counts = Counter(all_tags_raw)
    top_50_tags = [tag for tag, count in tag_counts.most_common(50)]

    global_centroid = np.mean(embedding_3d, axis=0)
    global_spread = np.mean(np.linalg.norm(embedding_3d - global_centroid, axis=1))

    results = []
    for target_tag in top_50_tags:
        # Find indices of all recipes containing this tag
        indices = [i for i, row in enumerate(raw_data) if target_tag in ast.literal_eval(row['tags'])]
        
        if len(indices) < 10: 
            continue
            
        cluster_points = embedding_3d[indices]
        centroid = np.mean(cluster_points, axis=0)
        avg_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
        
        # Density score: > 1.0 means it is tighter than the global average
        density_score = global_spread / avg_distance if avg_distance > 0 else 0
        
        results.append({
            'Tag': target_tag,
            'Recipe_Count': len(indices),
            'Average_Spread': round(avg_distance, 4),
            'Density_Score': round(density_score, 4)
        })

    # Sort by the tightest, most mathematically valid clusters
    results.sort(key=lambda x: x['Density_Score'], reverse=True)

    print("Exporting documentation to CSV...")
    export_filename = "tag_variance_report.csv"
    
    with open(export_filename, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['Tag', 'Recipe_Count', 'Average_Spread', 'Density_Score'])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nEvaluation saved to '{export_filename}'.")
    
    # Print top 10 to terminal for quick review
    print("\nTop 10 Most Valid Culinary Clusters:")
    print(f"{'TAG':<25} | {'COUNT':<10} | {'DENSITY SCORE'}")
    print("-" * 55)
    for r in results[:100]:
        print(f"{r['Tag']:<25} | {r['Recipe_Count']:<10} | {r['Density_Score']}x Baseline")

if __name__ == "__main__":
    run_evaluation()