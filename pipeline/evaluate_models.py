import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "web" / "model_data"
RESEARCH_DIR = ROOT / "research"
FIGURE_DIR = RESEARCH_DIR / "figures"

MODELS = {
    "SBERT Combined": "sbert_all_data.json",
    "SBERT Ingredients": "sbert_ingredients_data.json",
    "SBERT Names": "sbert_names_data.json",
    "TF-IDF": "tfidf_data.json",
}


def load_model(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def evaluate_model(records):
    ids = [r["id"] for r in records]
    id_to_index = {rid: i for i, rid in enumerate(ids)}
    labels = np.array([r["galaxy_cluster"] for r in records])
    coords = np.array([[r["x"], r["y"], r["z"]] for r in records], dtype=np.float32)
    ingredient_sets = [set(r.get("ingredients", [])) for r in records]
    neighbor_sets = [set(r.get("similar_recipes", [])) for r in records]

    tag_purity = []
    ingredient_jaccard = []

    for i, record in enumerate(records):
        ingredients = ingredient_sets[i]
        for neighbor_id in record.get("similar_recipes", []):
            j = id_to_index[neighbor_id]
            tag_purity.append(labels[j] == labels[i])
            ingredient_jaccard.append(mean_jaccard(ingredients, ingredient_sets[j]))

    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(coords)
    geometric_neighbors = knn.kneighbors(coords, return_distance=False)[:, 1:]
    geometric_sets = [{ids[j] for j in neighbors} for neighbors in geometric_neighbors]
    geometry_overlap = [
        mean_jaccard(precomputed, geometric)
        for precomputed, geometric in zip(neighbor_sets, geometric_sets)
    ]

    return {
        "recipes": len(records),
        "cluster_count": len(set(labels)),
        "neighbor_count": len(records[0].get("similar_recipes", [])),
        "neighbor_tag_purity": float(np.mean(tag_purity)),
        "ingredient_jaccard": float(np.mean(ingredient_jaccard)),
        "silhouette_3d": float(silhouette_score(coords, labels)),
        "semantic_to_geometry_overlap": float(np.mean(geometry_overlap)),
    }


def pairwise_overlap(model_records):
    overlaps = {}
    names = list(model_records)

    for i, name_a in enumerate(names):
        records_a = model_records[name_a]
        neighbors_a = {
            r["id"]: set(r.get("similar_recipes", [])) for r in records_a
        }

        for name_b in names[i + 1 :]:
            records_b = model_records[name_b]
            neighbors_b = {
                r["id"]: set(r.get("similar_recipes", [])) for r in records_b
            }
            values = [
                mean_jaccard(neighbors_a[rid], neighbors_b[rid])
                for rid in neighbors_a.keys() & neighbors_b.keys()
            ]
            overlaps[f"{name_a}|{name_b}"] = float(np.mean(values))

    return overlaps


def cluster_metrics(records):
    ids = [r["id"] for r in records]
    id_to_index = {rid: i for i, rid in enumerate(ids)}
    ingredient_sets = [set(r.get("ingredients", [])) for r in records]
    stats = {}

    for i, record in enumerate(records):
        cluster = record["galaxy_cluster"]
        entry = stats.setdefault(
            cluster,
            {
                "recipes": 0,
                "neighbor_tag_matches": 0,
                "neighbor_count": 0,
                "ingredient_jaccard_sum": 0.0,
            },
        )
        entry["recipes"] += 1

        for neighbor_id in record.get("similar_recipes", []):
            j = id_to_index[neighbor_id]
            entry["neighbor_count"] += 1
            entry["neighbor_tag_matches"] += int(
                records[j]["galaxy_cluster"] == cluster
            )
            entry["ingredient_jaccard_sum"] += mean_jaccard(
                ingredient_sets[i], ingredient_sets[j]
            )

    return {
        cluster: {
            "recipes": values["recipes"],
            "neighbor_tag_purity": values["neighbor_tag_matches"]
            / values["neighbor_count"],
            "ingredient_jaccard": values["ingredient_jaccard_sum"]
            / values["neighbor_count"],
        }
        for cluster, values in stats.items()
    }


def model_distinctiveness(pairwise):
    names = list(MODELS)
    distinctiveness = {}

    for name in names:
        overlaps = [
            pairwise[key]
            for key in pairwise
            if name in key.split("|")
        ]
        distinctiveness[name] = 1.0 - float(np.mean(overlaps))

    return distinctiveness


def bar_chart(path, values, title, xlabel, color):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(values)
    scores = [values[name] for name in names]

    ax.barh(names, scores, color=color)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def heatmap(path, matrix, names):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pairwise Model Neighbor Overlap")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def radar_chart(path, model_metrics, distinctiveness):
    labels = [
        "Tag purity",
        "Ingredient overlap",
        "3D overlap",
        "Distinctiveness",
    ]
    names = list(model_metrics)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(9, 9), subplot_kw={"projection": "polar"}
    )
    colors = ["#38bdf8", "#f97316", "#a855f7", "#2dd4bf"]

    for color, name in zip(colors, names):
        metrics = model_metrics[name]
        values = [
            metrics["neighbor_tag_purity"],
            metrics["ingredient_jaccard"],
            metrics["semantic_to_geometry_overlap"],
            distinctiveness[name],
        ]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model Behavior Profile", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cluster_heatmap(path, cluster_results, value_key, title, cmap):
    clusters = sorted(
        cluster_results["SBERT Combined"],
        key=lambda cluster: cluster_results["SBERT Combined"][cluster]["recipes"],
        reverse=True,
    )
    models = list(cluster_results)
    matrix = np.array(
        [
            [cluster_results[model][cluster][value_key] for model in models]
            for cluster in clusters
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(clusters)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticklabels(clusters)

    for i in range(len(clusters)):
        for j in range(len(models)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    ax.set_title(title, pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def projection_gallery(path, model_records):
    names = list(model_records)
    colors = ["#38bdf8", "#f97316", "#a855f7", "#2dd4bf"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    for ax, name, color in zip(axes.flat, names, colors):
        records = model_records[name]
        sample_size = min(2500, len(records))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(records), size=sample_size, replace=False)
        sample = [records[i] for i in indices]

        x = [record["x"] for record in sample]
        y = [record["y"] for record in sample]
        point_colors = [record["star_color"] for record in sample]

        ax.scatter(
            x,
            y,
            c=point_colors,
            s=4,
            alpha=0.45,
            linewidths=0,
        )
        ax.set_title(name, color=color, pad=12)
        ax.set_facecolor("#000005")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#222230")

    fig.suptitle("UMAP Galaxy Projection by Model", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.015,
        "Each panel shows a deterministic 2,500-recipe sample from the exported 3D coordinates.",
        ha="center",
        color="#a1a1aa",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor="#000005")
    plt.close(fig)
    plt.style.use("default")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model_records = {
        name: load_model(MODEL_DIR / filename) for name, filename in MODELS.items()
    }
    model_metrics = {
        name: evaluate_model(records) for name, records in model_records.items()
    }
    pairwise = pairwise_overlap(model_records)
    distinctiveness = model_distinctiveness(pairwise)
    clusters = {
        name: cluster_metrics(records)
        for name, records in model_records.items()
    }

    names = list(model_records)
    overlap_matrix = np.zeros((len(names), len(names)))
    for i, name_a in enumerate(names):
        overlap_matrix[i, i] = 1.0
        for j, name_b in enumerate(names[i + 1 :], i + 1):
            key = f"{name_a}|{name_b}"
            overlap_matrix[i, j] = pairwise[key]
            overlap_matrix[j, i] = pairwise[key]

    metrics = dict(model_metrics)
    metrics["pairwise_model_overlap"] = pairwise
    metrics["model_distinctiveness"] = distinctiveness
    metrics["cluster_metrics"] = clusters
    metrics["dataset"] = {
        "recipes": len(next(iter(model_records.values()))),
        "unique_ingredients": len(
            {
                ingredient
                for records in model_records.values()
                for record in records
                for ingredient in record.get("ingredients", [])
            }
        ),
    }

    with (RESEARCH_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    purity = {
        name: m["neighbor_tag_purity"] for name, m in model_metrics.items()
    }
    jaccard = {
        name: m["ingredient_jaccard"] for name, m in model_metrics.items()
    }
    geometry = {
        name: m["semantic_to_geometry_overlap"]
        for name, m in model_metrics.items()
    }

    bar_chart(
        FIGURE_DIR / "neighbor_tag_purity.png",
        purity,
        "Neighbor Tag Purity by Model",
        "Mean proportion of neighbors sharing the seed recipe's ontology tag",
        "#2dd4bf",
    )
    bar_chart(
        FIGURE_DIR / "ingredient_jaccard_similarity.png",
        jaccard,
        "Neighbor Ingredient Jaccard Similarity",
        "Mean seed/neighbor ingredient-set Jaccard similarity",
        "#f97316",
    )
    bar_chart(
        FIGURE_DIR / "semantic_vs_geometry_overlap.png",
        geometry,
        "Semantic-to-Geometric Neighbor Overlap",
        "Mean Jaccard overlap between semantic and 3D nearest-neighbor sets",
        "#a855f7",
    )
    heatmap(
        FIGURE_DIR / "model_neighbor_overlap_heatmap.png",
        overlap_matrix,
        names,
    )
    radar_chart(
        FIGURE_DIR / "model_behavior_radar.png",
        model_metrics,
        distinctiveness,
    )
    cluster_heatmap(
        FIGURE_DIR / "cluster_tag_purity_heatmap.png",
        clusters,
        "neighbor_tag_purity",
        "Neighbor Tag Purity by Ontology Cluster",
        "viridis",
    )
    cluster_heatmap(
        FIGURE_DIR / "cluster_ingredient_overlap_heatmap.png",
        clusters,
        "ingredient_jaccard",
        "Neighbor Ingredient Overlap by Ontology Cluster",
        "magma",
    )
    projection_gallery(
        FIGURE_DIR / "galaxy_projection_gallery.png",
        model_records,
    )

    print("Wrote research/metrics.json and eight model-comparison figures.")


if __name__ == "__main__":
    main()
