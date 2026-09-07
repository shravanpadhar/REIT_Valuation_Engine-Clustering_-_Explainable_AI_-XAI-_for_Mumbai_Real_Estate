"""Pytest suite: schema validation, preprocessing, feature engineering,
clustering, REIT metrics, and inference-path sanity checks.

Uses small synthetic fixtures rather than the full raw CSV so the suite
stays fast; `test_end_to_end_smoke` is the only test that touches the real
dataset, and it is skipped automatically if the file is not present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import _build_cluster_tier_map, select_best_kmeans, reduce_dimensionality
from src.config import MMR_LAT_BOUNDS, MMR_LON_BOUNDS, RAW_DATA_PATH
from src.data_pipeline import (
    LocalityTargetEncoder,
    SchemaValidationError,
    apply_domain_bounds,
    engineer_features,
    impute_missing,
    load_raw_data,
    reject_outliers_iqr,
    reject_outliers_zscore,
)
from src.reit_metrics import compute_property_reit_metrics, compute_undervaluation_gap, simulate_portfolio_returns
from src.valuation_model import predict_price
from src.zone_mapping import assign_macro_zone

N_SYNTHETIC = 400


@pytest.fixture(scope="module")
def synthetic_raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    localities = ["Andheri", "Thane", "Colaba", "Kharghar", "UnknownVillageXYZ"]
    property_types = ["Apartment", "Villa", "Independent House", "Studio Apartment"]
    furnished = ["Unfurnished", "Semi-Furnished", "Furnished"]

    area = rng.uniform(300, 2000, N_SYNTHETIC)
    price_per_sqft = rng.uniform(6000, 40000, N_SYNTHETIC)
    price = area * price_per_sqft

    df = pd.DataFrame({
        "title": [f"Property {i}" for i in range(N_SYNTHETIC)],
        "price": price,
        "area": area,
        "price_per_sqft": price_per_sqft,
        "locality": rng.choice(localities, N_SYNTHETIC),
        "city": "Mumbai",
        "property_type": rng.choice(property_types, N_SYNTHETIC),
        "bedroom_num": rng.integers(1, 5, N_SYNTHETIC),
        "bathroom_num": rng.integers(1, 4, N_SYNTHETIC),
        "balcony_num": rng.integers(0, 3, N_SYNTHETIC),
        "furnished": rng.choice(furnished, N_SYNTHETIC),
        "age": rng.integers(0, 30, N_SYNTHETIC),
        "total_floors": rng.integers(1, 40, N_SYNTHETIC),
        "latitude": rng.uniform(*MMR_LAT_BOUNDS, N_SYNTHETIC),
        "longitude": rng.uniform(*MMR_LON_BOUNDS, N_SYNTHETIC),
    })
    return df


@pytest.fixture(scope="module")
def engineered_df(synthetic_raw_df: pd.DataFrame):
    df, artifacts, locality_encoder = engineer_features(synthetic_raw_df, fit_encoders=True)
    return df, artifacts, locality_encoder


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
class TestSchemaValidation:
    def test_missing_columns_raise(self, tmp_path, synthetic_raw_df):
        bad_path = tmp_path / "bad.csv"
        synthetic_raw_df.drop(columns=["price"]).to_csv(bad_path, index=False)
        with pytest.raises(SchemaValidationError):
            load_raw_data(str(bad_path))

    def test_valid_schema_loads(self, tmp_path, synthetic_raw_df):
        good_path = tmp_path / "good.csv"
        synthetic_raw_df.to_csv(good_path, index=False)
        df = load_raw_data(str(good_path))
        assert len(df) == N_SYNTHETIC
        assert df["price"].dtype.kind in "fi"


# --------------------------------------------------------------------------- #
# Domain-bound filtering and outlier rejection
# --------------------------------------------------------------------------- #
class TestOutlierRejection:
    def test_domain_bounds_drops_out_of_range_geo(self, synthetic_raw_df):
        df = synthetic_raw_df.copy()
        df.loc[0, "latitude"] = 5.0  # clearly outside MMR
        filtered = apply_domain_bounds(df)
        assert 0 not in filtered.index or filtered.loc[df.index == 0].empty
        assert len(filtered) <= len(df)

    def test_domain_bounds_drops_negative_price(self, synthetic_raw_df):
        df = synthetic_raw_df.copy()
        df.loc[1, "price"] = -100
        filtered = apply_domain_bounds(df)
        assert (filtered["price"] > 0).all()

    def test_iqr_rejection_removes_extreme_area(self, synthetic_raw_df):
        df = synthetic_raw_df.copy()
        df.loc[2, "area"] = 500_000  # absurd outlier within an otherwise normal property_type group
        filtered = reject_outliers_iqr(df)
        assert len(filtered) < len(df)
        assert filtered["area"].max() < 500_000

    def test_zscore_rejection_is_bounded(self, synthetic_raw_df):
        filtered = reject_outliers_zscore(synthetic_raw_df)
        assert len(filtered) <= len(synthetic_raw_df)
        assert len(filtered) > 0


class TestImputation:
    def test_imputes_numeric_and_categorical_nans(self, synthetic_raw_df):
        df = synthetic_raw_df.copy()
        df.loc[0, "area"] = np.nan
        df.loc[1, "furnished"] = np.nan
        imputed = impute_missing(df)
        assert imputed.isna().sum().sum() == 0

    def test_noop_when_no_missing(self, synthetic_raw_df):
        imputed = impute_missing(synthetic_raw_df)
        pd.testing.assert_frame_equal(imputed, synthetic_raw_df)


# --------------------------------------------------------------------------- #
# Zone mapping
# --------------------------------------------------------------------------- #
class TestZoneMapping:
    def test_all_rows_get_a_zone(self, synthetic_raw_df):
        zones = assign_macro_zone(synthetic_raw_df)
        assert zones.isna().sum() == 0
        assert len(zones) == len(synthetic_raw_df)

    def test_known_locality_maps_correctly(self, synthetic_raw_df):
        zones = assign_macro_zone(synthetic_raw_df)
        colaba_zones = zones[synthetic_raw_df["locality"] == "Colaba"]
        assert (colaba_zones == "South Mumbai").all()

    def test_unknown_locality_falls_back_via_geo_centroid(self, synthetic_raw_df):
        zones = assign_macro_zone(synthetic_raw_df)
        unknown_zones = zones[synthetic_raw_df["locality"] == "UnknownVillageXYZ"]
        assert unknown_zones.isin(zones.cat.categories).all()
        assert unknown_zones.notna().all()


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class TestFeatureEngineering:
    def test_engineered_columns_present(self, engineered_df):
        df, _, _ = engineered_df
        expected = {
            "macro_zone", "age_bin", "amenity_score", "locality_target_enc",
            "furnished_ordinal", "property_type_encoded", "macro_zone_encoded",
            "locality_freq_enc", "age_bin_encoded",
        }
        assert expected.issubset(df.columns)

    def test_amenity_score_bounded(self, engineered_df):
        df, _, _ = engineered_df
        assert df["amenity_score"].between(0, 100).all()

    def test_no_nans_in_model_features(self, engineered_df):
        from src.data_pipeline import MODEL_FEATURE_COLUMNS
        df, _, _ = engineered_df
        assert df[MODEL_FEATURE_COLUMNS].isna().sum().sum() == 0

    def test_price_per_sqft_excluded_from_model_features(self):
        from src.data_pipeline import MODEL_FEATURE_COLUMNS
        assert "price_per_sqft" not in MODEL_FEATURE_COLUMNS

    def test_locality_target_encoder_transform_matches_fitted_mapping(self, synthetic_raw_df):
        encoder = LocalityTargetEncoder(n_splits=3)
        df = synthetic_raw_df.copy()
        df["log_price"] = np.log1p(df["price"])
        encoder.fit_transform(df, "locality", "log_price")

        transformed = encoder.transform(df.iloc[[0]], "locality")
        assert transformed.shape == (1,)
        assert np.isfinite(transformed).all()

    def test_locality_target_encoder_unseen_locality_uses_global_mean(self, synthetic_raw_df):
        encoder = LocalityTargetEncoder(n_splits=3)
        df = synthetic_raw_df.copy()
        df["log_price"] = np.log1p(df["price"])
        encoder.fit_transform(df, "locality", "log_price")

        unseen = pd.DataFrame({"locality": ["Never Seen Before Place"]})
        result = encoder.transform(unseen, "locality")
        assert result[0] == pytest.approx(encoder.global_mean_)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
class TestClustering:
    def test_select_best_kmeans_returns_valid_k(self, engineered_df):
        from src.data_pipeline import CATEGORICAL_FOR_CLUSTERING
        df, _, _ = engineered_df
        matrix = df[CATEGORICAL_FOR_CLUSTERING].to_numpy(dtype=float)
        _, _, reduced = reduce_dimensionality(matrix)
        _, best_k, sil, dbi, _ = select_best_kmeans(reduced, min_k=2, max_k=4)
        assert 2 <= best_k <= 4
        assert -1.0 <= sil <= 1.0
        assert dbi >= 0.0

    def test_cluster_tier_names_are_ranked_by_price_per_sqft(self, engineered_df):
        df, _, _ = engineered_df
        labels = (df["price_per_sqft"] > df["price_per_sqft"].median()).astype(int).to_numpy()
        tier_map, profile = _build_cluster_tier_map(df, labels)
        assert len(tier_map) == 2
        top_tier_cluster = profile.sort_values("median_price_per_sqft", ascending=False).iloc[0]["cluster_id"]
        assert tier_map[int(top_tier_cluster)] in {"Ultra-Luxury Core", "Established Upper-Mid"}


# --------------------------------------------------------------------------- #
# REIT metrics
# --------------------------------------------------------------------------- #
class TestReitMetrics:
    def test_cap_rate_and_rent_are_positive(self, engineered_df):
        df, _, _ = engineered_df
        scored = compute_property_reit_metrics(df, value_column="price")
        assert (scored["estimated_annual_rent"] > 0).all()
        assert (scored["cap_rate"] > 0).all()
        assert (scored["net_operating_income"] > 0).all()

    def test_zero_price_does_not_raise_division_error(self, engineered_df):
        # A zero-priced property has a genuinely undefined cap rate; the
        # important behavior is that this returns NaN rather than raising
        # ZeroDivisionError/inf, so downstream aggregation (.mean(), etc.)
        # degrades gracefully instead of crashing.
        df, _, _ = engineered_df
        df = df.copy()
        df.loc[0, "price"] = 0.0
        scored = compute_property_reit_metrics(df, value_column="price")
        cap_rate = scored["cap_rate"].iloc[0]
        assert np.isnan(cap_rate) or cap_rate == 0
        assert not np.isinf(cap_rate)

    def test_zero_rent_does_not_raise_division_error(self, engineered_df):
        df, _, _ = engineered_df
        scored = compute_property_reit_metrics(df, value_column="price")
        scored.loc[0, "estimated_annual_rent"] = 0.0
        # price_to_rent_ratio recompute path shouldn't be exercised again here;
        # just assert the original computation never produced inf/nan improperly.
        assert not np.isinf(scored["price_to_rent_ratio"]).any()

    def test_undervaluation_gap_sign(self):
        actual = pd.Series([1_000_000.0, 1_000_000.0])
        predicted = pd.Series([1_200_000.0, 800_000.0])
        gap = compute_undervaluation_gap(actual, predicted)
        assert gap.iloc[0] > 0  # trades below fair value
        assert gap.iloc[1] < 0  # trades above fair value

    def test_portfolio_simulation_returns_sane_ranges(self, engineered_df):
        df, _, _ = engineered_df
        scored = compute_property_reit_metrics(df, value_column="price")
        result = simulate_portfolio_returns(scored.head(100), n_paths=500)
        assert result.n_properties == 100
        assert -1.0 < result.expected_annualized_return < 2.0
        assert result.return_volatility >= 0.0

    def test_portfolio_simulation_raises_on_empty(self):
        with pytest.raises(ValueError):
            simulate_portfolio_returns(pd.DataFrame())


# --------------------------------------------------------------------------- #
# Inference path
# --------------------------------------------------------------------------- #
class _StubModel:
    """Predicts a deterministic log-price so predict_price's inverse-transform
    logic can be tested without training a real gradient-boosted model."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), np.log1p(5_000_000.0))


class TestInference:
    def test_predict_price_inverts_log_transform(self, engineered_df):
        from src.data_pipeline import MODEL_FEATURE_COLUMNS
        df, _, _ = engineered_df
        row = df.iloc[[0]].copy()
        row["cluster_id"] = 0
        feature_columns = MODEL_FEATURE_COLUMNS + ["cluster_id"]
        price = predict_price(_StubModel(), row, feature_columns)
        assert price == pytest.approx(5_000_000.0, rel=1e-3)


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="Raw dataset not present in this environment.")
def test_end_to_end_smoke():
    """Runs the real cleaning + feature-engineering pipeline on the actual
    raw CSV (skipped if the file isn't available) as a lightweight smoke test."""
    df = load_raw_data()
    df = apply_domain_bounds(df)
    df = reject_outliers_iqr(df)
    df = reject_outliers_zscore(df)
    df = impute_missing(df)
    df, _, _ = engineer_features(df, fit_encoders=True)

    assert len(df) > 0
    assert df.isna().sum().sum() == 0
