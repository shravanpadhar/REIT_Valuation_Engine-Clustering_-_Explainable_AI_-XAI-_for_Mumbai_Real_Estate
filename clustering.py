"""
=============================================================================
Phase 2: K-Means Clustering with Elbow Method
=============================================================================
Segments properties into Value / Premium / Luxury clusters.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Features used for clustering
CLUSTER_FEATURES = [
    "Price_per_SqFt", "Carpet_Area_SqFt", "Investment_Yield_Pct",
    "Distance_to_Metro_km", "Amenity_Score", "Floor_Ratio",
    "Building_Age_Years", "Crime_Rate_Index"
]


def load_clean_data():
    path = os.path.join(DATA_DIR, "mumbai_realestate_clean.csv")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records for clustering")
    return df


def prepare_features(df):
    """Extract and standardize clustering features."""
    X = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Feature matrix: {X_scaled.shape}")
    return X_scaled, scaler


def elbow_method(X_scaled, k_range=range(2, 11)):
    """Run Elbow Method to find optimal k."""
    print("\n-- Elbow Method --")
    inertias = []
    sil_scores = []
    db_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        sil_scores.append(sil)
        db_scores.append(db)
        print(f"  k={k}: Inertia={km.inertia_:,.0f} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

    # Save elbow plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].set_title('Elbow Method - Inertia')
    axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.7, label='k=3')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), sil_scores, 'gs-', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (Higher = Better)')
    axes[1].axvline(x=3, color='r', linestyle='--', alpha=0.7, label='k=3')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(list(k_range), db_scores, 'r^-', linewidth=2)
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin Index (Lower = Better)')
    axes[2].axvline(x=3, color='r', linestyle='--', alpha=0.7, label='k=3')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "elbow_method.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nElbow plot saved: {plot_path}")

    return inertias, sil_scores, db_scores


def run_kmeans(X_scaled, k=3):
    """Run final K-Means with optimal k."""
    print(f"\n-- Final K-Means (k={k}) --")
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    print(f"  Silhouette Score:    {sil:.4f}")
    print(f"  Davies-Bouldin Index: {db:.4f}")

    return km, labels, {"silhouette_score": round(sil, 4), "davies_bouldin_index": round(db, 4)}


def assign_cluster_names(df, labels, km, scaler):
    """Map cluster numbers to Value/Premium/Luxury based on mean price."""
    df = df.copy()
    df["Cluster_ID"] = labels

    # Determine cluster order by mean price per sqft
    cluster_means = df.groupby("Cluster_ID")["Price_per_SqFt"].mean().sort_values()
    name_map = {}
    names = ["Value", "Premium", "Luxury"]
    for i, cid in enumerate(cluster_means.index):
        name_map[cid] = names[i]

    df["Cluster_Label"] = df["Cluster_ID"].map(name_map)
    print(f"\nCluster Mapping: {name_map}")

    # Cluster summary
    summary = df.groupby("Cluster_Label").agg(
        Count=("Property_ID", "count"),
        Avg_Price_SqFt=("Price_per_SqFt", "mean"),
        Avg_Carpet_Area=("Carpet_Area_SqFt", "mean"),
        Avg_Yield=("Investment_Yield_Pct", "mean"),
        Avg_Metro_Dist=("Distance_to_Metro_km", "mean"),
        Avg_Amenity_Score=("Amenity_Score", "mean"),
    ).round(2)
    print(f"\n{summary}")

    return df, name_map


if __name__ == "__main__":
    df = load_clean_data()
    X_scaled, scaler = prepare_features(df)

    # Elbow method
    inertias, sil_scores, db_scores = elbow_method(X_scaled)

    # Final clustering with k=3
    km_model, labels, metrics = run_kmeans(X_scaled, k=3)

    # Assign names
    df_clustered, name_map = assign_cluster_names(df, labels, km_model, scaler)

    # Save
    out_path = os.path.join(DATA_DIR, "mumbai_realestate_clustered.csv")
    df_clustered.to_csv(out_path, index=False)
    print(f"\nClustered dataset saved: {out_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"k": 3, "metrics": metrics, "cluster_map": name_map,
                    "features_used": CLUSTER_FEATURES}, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
