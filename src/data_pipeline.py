"""Data cleaning, outlier rejection, imputation, and feature engineering.

Pipeline stages (see `build_processed_dataset` for orchestration):
    1. Load + schema validation
    2. Domain-bound filtering (hard physical/geographic limits)
    3. Statistical outlier rejection (IQR and z-score, on price/area/ppsf)
    4. Missing-value imputation (defensive -- the source file is pre-cleaned,
       but production data feeds are never guaranteed to stay that way)
    5. Feature engineering (macro zone, age bins, amenity score, locality
       frequency + K-fold target encoding, categorical ordinal encoding)
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

from src.artifacts import FeatureArtifacts, LocalityTargetEncoder
from src.config import (
    CATEGORICAL_ENCODERS_PATH,
    FEATURE_METADATA_PATH,
    IQR_MULTIPLIER,
    LOCALITY_FREQ_ENCODER_PATH,
    LOCALITY_TARGET_ENCODER_PATH,
    LOG_TARGET_COLUMN,
    MAX_AGE,
    MAX_AREA,
    MAX_BALCONIES,
    MAX_BATHROOMS,
    MAX_BEDROOMS,
    MAX_PRICE,
    MAX_TOTAL_FLOORS,
    MIN_AREA,
    MIN_PRICE,
    MMR_LAT_BOUNDS,
    MMR_LON_BOUNDS,
    OUTLIER_COLUMNS,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    Z_SCORE_THRESHOLD,
    get_logger,
)
from src.zone_mapping import assign_macro_zone

logger = get_logger(__name__)

FURNISHED_ORDER: dict[str, int] = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
AGE_BIN_LABELS: list[str] = ["New / Under-Construction", "0-5 yrs", "5-10 yrs", "10-20 yrs", "20+ yrs"]
AGE_BIN_EDGES: list[float] = [-0.1, 0.0, 5.0, 10.0, 20.0, np.inf]

LOCALITY_TARGET_ENCODING_SMOOTHING: float = 20.0
LOCALITY_TARGET_ENCODING_FOLDS: int = 5

# Features handed to the valuation model. `price_per_sqft` is deliberately
# excluded: it is a deterministic function of the target (price / area), so
# using it as a predictor would leak the label.
MODEL_FEATURE_COLUMNS: list[str] = [
    "area", "bedroom_num", "bathroom_num", "balcony_num", "age", "total_floors",
    "latitude", "longitude", "furnished_ordinal", "property_type_encoded",
    "macro_zone_encoded", "locality_freq_enc", "locality_target_enc",
    "amenity_score", "age_bin_encoded",
]

CATEGORICAL_FOR_CLUSTERING: list[str] = [
    "price_per_sqft", "area", "bedroom_num", "bathroom_num", "amenity_score",
    "age", "macro_zone_encoded", "latitude", "longitude",
]


class SchemaValidationError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw CSV and validate it matches the expected schema."""
    csv_path = path or RAW_DATA_PATH
    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)

    missing_cols = set(RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise SchemaValidationError(f"Raw data is missing required columns: {sorted(missing_cols)}")

    numeric_cols = ["price", "area", "price_per_sqft", "bedroom_num", "bathroom_num",
                     "balcony_num", "age", "total_floors", "latitude", "longitude"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def apply_domain_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows outside hard physical/geographic plausibility bounds."""
    n_before = len(df)
    mask = (
        df["price"].between(MIN_PRICE, MAX_PRICE)
        & df["area"].between(MIN_AREA, MAX_AREA)
        & df["bedroom_num"].between(0, MAX_BEDROOMS)
        & df["bathroom_num"].between(1, MAX_BATHROOMS)
        & df["balcony_num"].between(0, MAX_BALCONIES)
        & df["age"].between(0, MAX_AGE)
        & df["total_floors"].between(1, MAX_TOTAL_FLOORS)
        & df["latitude"].between(*MMR_LAT_BOUNDS)
        & df["longitude"].between(*MMR_LON_BOUNDS)
    )
    filtered = df.loc[mask].copy()
    logger.info("Domain-bound filter: kept %d/%d rows (dropped %d)",
                len(filtered), n_before, n_before - len(filtered))
    return filtered


def reject_outliers_iqr(df: pd.DataFrame, columns: list[str] = OUTLIER_COLUMNS,
                         multiplier: float = IQR_MULTIPLIER) -> pd.DataFrame:
    """Drop rows that are IQR outliers on any of `columns` (evaluated per property_type
    to avoid conflating villas with studio apartments under one global fence)."""
    n_before = len(df)
    keep_mask = pd.Series(True, index=df.index)

    for ptype, group in df.groupby("property_type", observed=True):
        group_mask = pd.Series(True, index=group.index)
        for col in columns:
            q1, q3 = group[col].quantile(0.25), group[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            group_mask &= group[col].between(lower, upper)
        keep_mask.loc[group.index] = group_mask

    filtered = df.loc[keep_mask].copy()
    logger.info("IQR outlier rejection: kept %d/%d rows (dropped %d)",
                len(filtered), n_before, n_before - len(filtered))
    return filtered


def reject_outliers_zscore(df: pd.DataFrame, columns: list[str] = OUTLIER_COLUMNS,
                            threshold: float = Z_SCORE_THRESHOLD) -> pd.DataFrame:
    """Drop rows whose log-transformed values are z-score outliers on `columns`.

    Log-transforming first prevents the heavy right skew in price/area from
    making the z-score test toothless (raw-scale std is dominated by the tail).
    """
    n_before = len(df)
    keep_mask = pd.Series(True, index=df.index)
    for col in columns:
        log_vals = np.log1p(df[col].clip(lower=0))
        z = (log_vals - log_vals.mean()) / log_vals.std(ddof=0)
        keep_mask &= z.abs() <= threshold

    filtered = df.loc[keep_mask].copy()
    logger.info("Z-score outlier rejection: kept %d/%d rows (dropped %d)",
                len(filtered), n_before, n_before - len(filtered))
    return filtered


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns and mode-impute categoricals. Defensive:
    the shipped dataset is pre-cleaned, but production ingestion should not
    assume that holds forever."""
    df = df.copy()
    n_missing_before = int(df.isna().sum().sum())
    if n_missing_before == 0:
        logger.info("No missing values detected; imputation is a no-op.")
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    logger.info("Imputed %d missing values across %d columns", n_missing_before, len(df.columns))
    return df


def _raw_amenity_score(df: pd.DataFrame) -> pd.Series:
    """Unscaled composite luxury-amenity proxy (see `engineer_features` docs
    for why this heuristic exists in the absence of an explicit amenities list).

    Combines: furnishing level, balcony count, bathroom-to-bedroom ratio,
    building height, construction age, and property type -- the closest
    available proxies for a premium, amenity-rich building/unit.
    """
    furnished_component = df["furnished"].map(FURNISHED_ORDER).fillna(0)
    balcony_component = df["balcony_num"].clip(upper=3)
    bath_bed_ratio = (df["bathroom_num"] - df["bedroom_num"]).clip(lower=0, upper=3)
    high_rise_component = (df["total_floors"] >= 15).astype(int) * 2 + \
                           ((df["total_floors"] >= 8) & (df["total_floors"] < 15)).astype(int)
    new_build_component = (df["age"] <= 1).astype(int) * 2 + \
                           ((df["age"] > 1) & (df["age"] <= 5)).astype(int)
    property_type_component = df["property_type"].map({
        "Villa": 3, "Independent House": 2, "Independent Floor": 1,
        "Apartment": 1, "Studio Apartment": 0,
    }).fillna(0)

    return (
        2.0 * furnished_component + 1.5 * balcony_component + 1.5 * bath_bed_ratio
        + 2.0 * high_rise_component + 1.5 * new_build_component + 1.0 * property_type_component
    )


def _scale_amenity_score(raw_score: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """Min-max scale to 0-100 using bounds fitted on the training corpus.

    Using *fixed, persisted* bounds (rather than the batch's own min/max) is
    essential: at inference time `engineer_features` is called on a single
    underwriting candidate, where a batch-relative min/max would collapse to
    a meaningless constant (min == max == the one row's score).
    """
    if max_val > min_val:
        scaled = 100.0 * (raw_score - min_val) / (max_val - min_val)
    else:
        scaled = pd.Series(50.0, index=raw_score.index)
    return scaled.clip(lower=0.0, upper=100.0).round(2)


def engineer_features(df: pd.DataFrame, fit_encoders: bool = True,
                       artifacts: FeatureArtifacts | None = None,
                       locality_encoder: LocalityTargetEncoder | None = None
                       ) -> tuple[pd.DataFrame, FeatureArtifacts, LocalityTargetEncoder]:
    """Add engineered features. When `fit_encoders` is False, `artifacts` and
    `locality_encoder` must be supplied (inference-time transform of new data
    using encoders fitted on the training corpus)."""
    df = df.copy()

    df["price_per_sqft"] = df["price"] / df["area"]
    df[LOG_TARGET_COLUMN] = np.log1p(df[TARGET_COLUMN])

    df["macro_zone"] = assign_macro_zone(df)
    df["age_bin"] = pd.cut(df["age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    df["bath_bed_ratio"] = (df["bathroom_num"] / df["bedroom_num"].replace(0, 1)).round(2)
    df["price_per_bedroom"] = (df["price"] / df["bedroom_num"].replace(0, 1)).round(2)
    raw_amenity = _raw_amenity_score(df)

    if fit_encoders:
        artifacts = FeatureArtifacts()

        property_types = sorted(df["property_type"].unique())
        artifacts.property_type_map = {v: i for i, v in enumerate(property_types)}

        macro_zones = sorted(df["macro_zone"].astype(str).unique())
        artifacts.macro_zone_map = {v: i for i, v in enumerate(macro_zones)}

        freq = df["locality"].value_counts(normalize=True)
        artifacts.locality_freq_map = freq.to_dict()
        artifacts.global_freq_fallback = float(freq.min()) if len(freq) else 0.0
        artifacts.global_log_price_mean = float(df[LOG_TARGET_COLUMN].mean())
        artifacts.amenity_score_min = float(raw_amenity.min())
        artifacts.amenity_score_max = float(raw_amenity.max())

        locality_encoder = LocalityTargetEncoder(
            n_splits=LOCALITY_TARGET_ENCODING_FOLDS,
            smoothing=LOCALITY_TARGET_ENCODING_SMOOTHING,
            random_state=RANDOM_SEED,
        )
        df["locality_target_enc"] = locality_encoder.fit_transform(df, "locality", LOG_TARGET_COLUMN)
    else:
        if artifacts is None or locality_encoder is None:
            raise ValueError("artifacts and locality_encoder are required when fit_encoders=False")
        df["locality_target_enc"] = locality_encoder.transform(df, "locality")

    df["amenity_score"] = _scale_amenity_score(raw_amenity, artifacts.amenity_score_min, artifacts.amenity_score_max)

    df["furnished_ordinal"] = df["furnished"].map(FURNISHED_ORDER).fillna(0).astype(int)
    df["property_type_encoded"] = df["property_type"].map(artifacts.property_type_map).fillna(-1).astype(int)
    df["macro_zone_encoded"] = df["macro_zone"].astype(str).map(artifacts.macro_zone_map).fillna(-1).astype(int)
    df["locality_freq_enc"] = df["locality"].map(artifacts.locality_freq_map).fillna(artifacts.global_freq_fallback)
    df["age_bin_encoded"] = df["age_bin"].cat.codes.astype(int)

    return df, artifacts, locality_encoder


def build_processed_dataset(save: bool = True) -> pd.DataFrame:
    """Run the full cleaning + feature-engineering pipeline end to end."""
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = load_raw_data()
    df = apply_domain_bounds(df)
    df = reject_outliers_iqr(df)
    df = reject_outliers_zscore(df)
    df = impute_missing(df)
    df, artifacts, locality_encoder = engineer_features(df, fit_encoders=True)

    df = df.reset_index(drop=True)
    logger.info("Final processed dataset: %d rows, %d columns", *df.shape)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DATA_PATH, index=False)
        joblib.dump(artifacts, CATEGORICAL_ENCODERS_PATH)
        joblib.dump(locality_encoder, LOCALITY_TARGET_ENCODER_PATH)
        joblib.dump(artifacts.locality_freq_map, LOCALITY_FREQ_ENCODER_PATH)
        joblib.dump(
            {"model_features": MODEL_FEATURE_COLUMNS, "clustering_features": CATEGORICAL_FOR_CLUSTERING},
            FEATURE_METADATA_PATH,
        )
        logger.info("Saved processed dataset to %s and encoders to models/", PROCESSED_DATA_PATH)

    return df


if __name__ == "__main__":
    build_processed_dataset(save=True)
