"""Central configuration: paths, constants, and REIT/valuation assumptions.

All modules import from here so paths and business assumptions live in one
place instead of being duplicated across the pipeline.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = ROOT_DIR / "models"

RAW_DATA_PATH: Final[Path] = RAW_DATA_DIR / "mumbai_house_price_raw.csv"
PROCESSED_DATA_PATH: Final[Path] = PROCESSED_DATA_DIR / "mumbai_house_price_processed.parquet"
CLUSTERED_DATA_PATH: Final[Path] = PROCESSED_DATA_DIR / "mumbai_house_price_clustered.parquet"
SCORED_DATA_PATH: Final[Path] = PROCESSED_DATA_DIR / "mumbai_house_price_scored.parquet"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Serialized artifact paths
VALUATION_MODEL_PATH: Final[Path] = MODELS_DIR / "valuation_model.joblib"
CLUSTER_MODEL_PATH: Final[Path] = MODELS_DIR / "cluster_model.joblib"
SCALER_PATH: Final[Path] = MODELS_DIR / "feature_scaler.joblib"
PCA_MODEL_PATH: Final[Path] = MODELS_DIR / "pca_model.joblib"
LOCALITY_TARGET_ENCODER_PATH: Final[Path] = MODELS_DIR / "locality_target_encoder.joblib"
LOCALITY_FREQ_ENCODER_PATH: Final[Path] = MODELS_DIR / "locality_freq_encoder.joblib"
CATEGORICAL_ENCODERS_PATH: Final[Path] = MODELS_DIR / "categorical_encoders.joblib"
CLUSTER_PROFILE_PATH: Final[Path] = MODELS_DIR / "cluster_profiles.joblib"
FEATURE_METADATA_PATH: Final[Path] = MODELS_DIR / "feature_metadata.joblib"
MODEL_METRICS_PATH: Final[Path] = MODELS_DIR / "model_metrics.joblib"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: Final[int] = 42

# --------------------------------------------------------------------------- #
# Raw schema
# --------------------------------------------------------------------------- #
RAW_COLUMNS: Final[list[str]] = [
    "title", "price", "area", "price_per_sqft", "locality", "city",
    "property_type", "bedroom_num", "bathroom_num", "balcony_num",
    "furnished", "age", "total_floors", "latitude", "longitude",
]

TARGET_COLUMN: Final[str] = "price"
LOG_TARGET_COLUMN: Final[str] = "log_price"

# Plausible bounding box for the Mumbai Metropolitan Region (MMR).
# Rows outside this box are geocoding errors, not real MMR properties.
MMR_LAT_BOUNDS: Final[tuple[float, float]] = (18.75, 19.55)
MMR_LON_BOUNDS: Final[tuple[float, float]] = (72.70, 73.35)

# Hard domain bounds used before statistical outlier rejection.
MIN_PRICE: Final[float] = 1.0e5          # INR 1 lakh
MAX_PRICE: Final[float] = 5.0e9          # INR 500 crore
MIN_AREA: Final[float] = 100.0           # sq. ft.
MAX_AREA: Final[float] = 15000.0         # sq. ft.
MAX_BEDROOMS: Final[int] = 10
MAX_BATHROOMS: Final[int] = 10
MAX_BALCONIES: Final[int] = 6
MAX_AGE: Final[int] = 60
MAX_TOTAL_FLOORS: Final[int] = 60

IQR_MULTIPLIER: Final[float] = 1.5
Z_SCORE_THRESHOLD: Final[float] = 4.0
OUTLIER_COLUMNS: Final[list[str]] = ["price", "area", "price_per_sqft"]

# --------------------------------------------------------------------------- #
# Macro micro-market tiers (spatial zones)
# --------------------------------------------------------------------------- #
ZONE_SOUTH_MUMBAI: Final[str] = "South Mumbai"
ZONE_WESTERN_SUBURBS: Final[str] = "Western Suburbs"
ZONE_CENTRAL_SUBURBS: Final[str] = "Central Suburbs"
ZONE_NAVI_MUMBAI: Final[str] = "Navi Mumbai"
ZONE_THANE_EXTENDED: Final[str] = "Thane & Extended MMR"

MACRO_ZONES: Final[list[str]] = [
    ZONE_SOUTH_MUMBAI,
    ZONE_WESTERN_SUBURBS,
    ZONE_CENTRAL_SUBURBS,
    ZONE_NAVI_MUMBAI,
    ZONE_THANE_EXTENDED,
]

# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
MIN_CLUSTERS: Final[int] = 4
MAX_CLUSTERS: Final[int] = 5
PCA_VARIANCE_RETAINED: Final[float] = 0.90

CLUSTER_TIER_NAMES: Final[list[str]] = [
    "Value / Affordable",
    "High-Growth Suburban",
    "Established Upper-Mid",
    "Ultra-Luxury Core",
    "Emerging Peripheral",
]

# --------------------------------------------------------------------------- #
# Valuation model
# --------------------------------------------------------------------------- #
CV_FOLDS: Final[int] = 5
# Optuna's inner search loop uses fewer folds than the final reported CV
# metric: on modest hardware (2-4 cores), repeatedly spinning up hundreds of
# full-core gradient-boosting thread pools in a tight loop (folds x trials x
# 2 models) has been observed to silently kill the process on Windows before
# it can be caught as a Python exception. Fewer inner folds cuts total fits
# without materially hurting the quality of the hyperparameters found.
OPTUNA_CV_FOLDS: Final[int] = int(os.environ.get("OPTUNA_CV_FOLDS", "3"))
# Overridable via env var so a resource-constrained deploy build (e.g. Render's
# free/starter build machines) can run a lighter tuning budget than a local run.
OPTUNA_TRIALS: Final[int] = int(os.environ.get("OPTUNA_TRIALS", "20"))
OPTUNA_TIMEOUT_SECONDS: Final[int] = int(os.environ.get("OPTUNA_TIMEOUT_SECONDS", "600"))
# Cap threads per individual model fit (rather than defaulting to "use every
# core") so that many sequential fits during hyperparameter search don't
# repeatedly spin up/tear down large OS-level thread pools.
MODEL_N_JOBS: Final[int] = max(1, min(4, (os.cpu_count() or 4) - 1))

# --------------------------------------------------------------------------- #
# REIT / rental-yield assumptions (institutional heuristics, MMR 2025-26 basis)
# --------------------------------------------------------------------------- #
# Gross annual rental yield as a fraction of fair market value, by macro zone.
# Sourced from typical MMR residential gross-yield ranges (2.0% - 4.0%).
BASE_GROSS_YIELD_BY_ZONE: Final[dict[str, float]] = {
    ZONE_SOUTH_MUMBAI: 0.028,
    ZONE_WESTERN_SUBURBS: 0.032,
    ZONE_CENTRAL_SUBURBS: 0.034,
    ZONE_NAVI_MUMBAI: 0.038,
    ZONE_THANE_EXTENDED: 0.040,
}

# Operating expense ratio applied to gross rent to derive NOI (property tax,
# maintenance, insurance, reserve for vacancy/repairs -- excludes debt service).
OPERATING_EXPENSE_RATIO: Final[float] = 0.30
VACANCY_RATE: Final[float] = 0.05

# Simulated long-run price appreciation (mean) and volatility (std) per zone,
# used for the risk-adjusted portfolio return simulation.
APPRECIATION_MEAN_BY_ZONE: Final[dict[str, float]] = {
    ZONE_SOUTH_MUMBAI: 0.055,
    ZONE_WESTERN_SUBURBS: 0.065,
    ZONE_CENTRAL_SUBURBS: 0.070,
    ZONE_NAVI_MUMBAI: 0.085,
    ZONE_THANE_EXTENDED: 0.080,
}
APPRECIATION_VOLATILITY_BY_ZONE: Final[dict[str, float]] = {
    ZONE_SOUTH_MUMBAI: 0.06,
    ZONE_WESTERN_SUBURBS: 0.08,
    ZONE_CENTRAL_SUBURBS: 0.09,
    ZONE_NAVI_MUMBAI: 0.12,
    ZONE_THANE_EXTENDED: 0.11,
}
RISK_FREE_RATE: Final[float] = 0.071  # 10Y Indian G-Sec proxy

N_MONTE_CARLO_PATHS: Final[int] = 5000
MONTE_CARLO_HORIZON_YEARS: Final[int] = 5

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
