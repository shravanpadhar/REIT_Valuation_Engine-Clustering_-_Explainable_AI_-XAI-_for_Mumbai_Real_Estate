"""Micro-market segmentation: dimensionality reduction + clustering.

Reduces the engineered feature space with PCA, fits K-Means for k in
{MIN_CLUSTERS..MAX_CLUSTERS}, selects the best k via Silhouette Score and
Davies-Bouldin Index, then labels each cluster with an institutional tier
name (e.g. "Ultra-Luxury Core") ranked by mean price per sq. ft.

HDBSCAN is offered as an optional density-based alternative for comparison;
K-Means is the production default because it yields a fixed, interpretable
number of tiers that underwriting teams can operationalize, whereas HDBSCAN's
noise points and variable cluster count are harder to action in a dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan as _hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _HDBSCAN_AVAILABLE = False

from src.config import (
    CLUSTER_MODEL_PATH,
    CLUSTER_PROFILE_PATH,
    CLUSTER_TIER_NAMES,
    CLUSTERED_DATA_PATH,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    PCA_MODEL_PATH,
    PCA_VARIANCE_RETAINED,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    get_logger,
)
from src.data_pipeline import CATEGORICAL_FOR_CLUSTERING

logger = get_logger(__name__)


@dataclass
class ClusteringResult:
    """Outputs of the market-segmentation stage, ready for persistence."""

    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    best_k: int
    silhouette: float
    davies_bouldin: float
    cluster_tier_map: dict[int, str]
    cluster_profile: pd.DataFrame


def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    matrix = df[feature_cols].to_numpy(dtype=float)
    return matrix


def reduce_dimensionality(
    matrix: np.ndarray, variance_retained: float = PCA_VARIANCE_RETAINED
) -> tuple[StandardScaler, PCA, np.ndarray]:
    """Standardize features then apply PCA, retaining components that explain
    `variance_retained` of total variance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_retained) + 1)
    n_components = max(2, min(n_components, scaled.shape[1]))

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(scaled)
    logger.info("PCA retained %d components explaining %.1f%% variance",
                n_components, 100 * cumulative[n_components - 1])
    return scaler, pca, reduced


def select_best_kmeans(
    reduced: np.ndarray, min_k: int = MIN_CLUSTERS, max_k: int = MAX_CLUSTERS
) -> tuple[KMeans, int, float, float, dict[int, dict[str, float]]]:
    """Fit K-Means for each k in [min_k, max_k] and pick the one with the best
    combined Silhouette (higher better) / Davies-Bouldin (lower better) score."""
    candidates: dict[int, dict[str, float]] = {}
    best_model: KMeans | None = None
    best_score = -np.inf
    best_k = min_k
    best_sil = 0.0
    best_dbi = 0.0

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(reduced)
        sil = silhouette_score(reduced, labels)
        dbi = davies_bouldin_score(reduced, labels)
        candidates[k] = {"silhouette": sil, "davies_bouldin": dbi}
        logger.info("k=%d -> silhouette=%.4f, davies_bouldin=%.4f", k, sil, dbi)

        combined = sil - 0.15 * dbi
        if combined > best_score:
            best_score, best_model, best_k, best_sil, best_dbi = combined, model, k, sil, dbi

    assert best_model is not None
    logger.info("Selected k=%d (silhouette=%.4f, davies_bouldin=%.4f)", best_k, best_sil, best_dbi)
    return best_model, best_k, best_sil, best_dbi, candidates


def fit_hdbscan(reduced: np.ndarray, min_cluster_size: int = 200) -> np.ndarray | None:
    """Optional density-based clustering for comparison against K-Means.
    Returns None if the hdbscan package is not installed."""
    if not _HDBSCAN_AVAILABLE:
        logger.warning("hdbscan not installed; skipping density-based comparison.")
        return None
    clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(reduced)
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN found %d clusters (+%d noise points)", len(set(labels)) - (1 if -1 in labels else 0), n_noise)
    return labels


def _build_cluster_tier_map(df: pd.DataFrame, cluster_labels: np.ndarray) -> tuple[dict[int, str], pd.DataFrame]:
    """Rank clusters by mean price per sq. ft. (descending) and assign
    institutional tier names from CLUSTER_TIER_NAMES, richest first."""
    working = df.copy()
    working["_cluster"] = cluster_labels

    profile = working.groupby("_cluster").agg(
        n_properties=("price", "size"),
        mean_price=("price", "mean"),
        median_price_per_sqft=("price_per_sqft", "median"),
        mean_area=("area", "mean"),
        mean_amenity_score=("amenity_score", "mean"),
        dominant_zone=("macro_zone", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    ).sort_values("median_price_per_sqft", ascending=False)

    # Highest price/sqft -> "Ultra-Luxury Core"; lowest -> "Value / Affordable".
    n_clusters = len(profile)
    ranked_tier_names = _rank_tier_names(n_clusters)

    tier_map = {int(cluster_id): ranked_tier_names[i] for i, cluster_id in enumerate(profile.index)}
    profile = profile.reset_index().rename(columns={"_cluster": "cluster_id"})
    profile["tier_name"] = profile["cluster_id"].map(tier_map)
    return tier_map, profile


def _rank_tier_names(n_clusters: int) -> list[str]:
    """Return tier names ordered from highest to lowest price/sqft for n_clusters."""
    preferred_order = [
        "Ultra-Luxury Core", "Established Upper-Mid", "High-Growth Suburban",
        "Value / Affordable", "Emerging Peripheral",
    ]
    ordered = [name for name in preferred_order if name in CLUSTER_TIER_NAMES] or preferred_order
    if n_clusters <= len(ordered):
        return ordered[:n_clusters]
    return ordered + [f"Tier {i}" for i in range(len(ordered), n_clusters)]


def run_clustering_pipeline(df: pd.DataFrame | None = None, save: bool = True) -> tuple[pd.DataFrame, ClusteringResult]:
    """End-to-end micro-market segmentation. Returns the input frame with
    `cluster_id` / `market_tier` columns appended, plus the fitted artifacts."""
    if df is None:
        df = pd.read_parquet(PROCESSED_DATA_PATH)

    matrix = _prepare_feature_matrix(df, CATEGORICAL_FOR_CLUSTERING)
    scaler, pca, reduced = reduce_dimensionality(matrix)
    kmeans, best_k, sil, dbi, candidates = select_best_kmeans(reduced)
    cluster_labels = kmeans.predict(reduced)

    tier_map, profile = _build_cluster_tier_map(df, cluster_labels)

    result_df = df.copy()
    result_df["cluster_id"] = cluster_labels
    result_df["market_tier"] = result_df["cluster_id"].map(tier_map)
    result_df["pca_1"] = reduced[:, 0]
    result_df["pca_2"] = reduced[:, 1] if reduced.shape[1] > 1 else 0.0

    result = ClusteringResult(
        scaler=scaler, pca=pca, kmeans=kmeans, best_k=best_k,
        silhouette=sil, davies_bouldin=dbi, cluster_tier_map=tier_map,
        cluster_profile=profile,
    )

    if save:
        CLUSTERED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(CLUSTERED_DATA_PATH, index=False)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(pca, PCA_MODEL_PATH)
        joblib.dump(kmeans, CLUSTER_MODEL_PATH)
        joblib.dump(
            {"tier_map": tier_map, "profile": profile, "silhouette": sil,
             "davies_bouldin": dbi, "best_k": best_k, "candidates": candidates,
             "feature_columns": CATEGORICAL_FOR_CLUSTERING},
            CLUSTER_PROFILE_PATH,
        )
        logger.info("Saved clustering artifacts to models/ and clustered data to %s", CLUSTERED_DATA_PATH)

    return result_df, result


def assign_cluster_to_new_property(
    feature_row: pd.DataFrame, scaler: StandardScaler, pca: PCA, kmeans: KMeans,
    tier_map: dict[int, str],
) -> tuple[int, str]:
    """Assign a single new property (already engineered) to its market tier."""
    matrix = feature_row[CATEGORICAL_FOR_CLUSTERING].to_numpy(dtype=float)
    scaled = scaler.transform(matrix)
    reduced = pca.transform(scaled)
    cluster_id = int(kmeans.predict(reduced)[0])
    return cluster_id, tier_map.get(cluster_id, "Unclassified")


if __name__ == "__main__":
    run_clustering_pipeline(save=True)
