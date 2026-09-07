"""Streamlit dashboard: REIT Valuation Engine for Mumbai Metropolitan Region.

Three tabs for institutional underwriting/portfolio teams:
    1. Market Explorer & Cluster Maps
    2. Single Property Underwriter (valuation + SHAP explanation)
    3. REIT Portfolio Screener (cap-rate / undervaluation filtering)

Run locally with:  streamlit run app/app.py
Deployed on Render.com as a web service (see render.yaml at repo root).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import assign_cluster_to_new_property
from src.config import (
    CATEGORICAL_ENCODERS_PATH,
    CLUSTER_MODEL_PATH,
    CLUSTER_PROFILE_PATH,
    CLUSTERED_DATA_PATH,
    LOCALITY_TARGET_ENCODER_PATH,
    PCA_MODEL_PATH,
    RISK_FREE_RATE,
    SCALER_PATH,
    VALUATION_MODEL_PATH,
)
from src.data_pipeline import CATEGORICAL_FOR_CLUSTERING, FURNISHED_ORDER, engineer_features
from src.explainability import build_shap_explainer, compute_shap_values, plot_shap_waterfall
from src.reit_metrics import compute_property_reit_metrics, compute_undervaluation_gap, simulate_portfolio_returns
from src.valuation_model import TRAINING_FEATURE_COLUMNS, predict_price

st.set_page_config(page_title="REIT Valuation Engine | MMR", layout="wide", page_icon="\U0001F3E2")

ARTIFACTS_MISSING_MSG = (
    "Model artifacts not found. Run the training pipeline first:\n\n"
    "    python -m src.data_pipeline\n"
    "    python -m src.clustering\n"
    "    python -m src.valuation_model\n"
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts() -> dict:
    import joblib

    required = [VALUATION_MODEL_PATH, CLUSTER_MODEL_PATH, SCALER_PATH, PCA_MODEL_PATH,
                CATEGORICAL_ENCODERS_PATH, LOCALITY_TARGET_ENCODER_PATH, CLUSTER_PROFILE_PATH]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(ARTIFACTS_MISSING_MSG)

    valuation_bundle = joblib.load(VALUATION_MODEL_PATH)
    cluster_profile_bundle = joblib.load(CLUSTER_PROFILE_PATH)

    return {
        "model": valuation_bundle["model"],
        "model_name": valuation_bundle["model_name"],
        "feature_columns": valuation_bundle["feature_columns"],
        "kmeans": joblib.load(CLUSTER_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "pca": joblib.load(PCA_MODEL_PATH),
        "feature_artifacts": joblib.load(CATEGORICAL_ENCODERS_PATH),
        "locality_encoder": joblib.load(LOCALITY_TARGET_ENCODER_PATH),
        "tier_map": cluster_profile_bundle["tier_map"],
        "cluster_profile": cluster_profile_bundle["profile"],
        "silhouette": cluster_profile_bundle["silhouette"],
        "davies_bouldin": cluster_profile_bundle["davies_bouldin"],
        "best_k": cluster_profile_bundle["best_k"],
    }


@st.cache_data(show_spinner="Loading property universe...")
def load_clustered_data() -> pd.DataFrame:
    if not CLUSTERED_DATA_PATH.exists():
        raise FileNotFoundError(ARTIFACTS_MISSING_MSG)
    return pd.read_parquet(CLUSTERED_DATA_PATH)


@st.cache_data(show_spinner="Scoring property universe (predicted fair value)...")
def build_scored_universe(_model, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    df = load_clustered_data().copy()
    X = df[list(feature_columns)].to_numpy(dtype=float)
    df["predicted_fair_value"] = np.expm1(_model.predict(X))
    df["undervaluation_gap"] = compute_undervaluation_gap(df["price"], df["predicted_fair_value"])
    df = compute_property_reit_metrics(df, value_column="price")
    return df


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    return build_shap_explainer(_model)


@st.cache_data(show_spinner=False)
def locality_lookup_table() -> pd.DataFrame:
    df = load_clustered_data()
    return (
        df.groupby("locality")
        .agg(median_ppsf=("price_per_sqft", "median"), lat=("latitude", "mean"), lon=("longitude", "mean"))
        .reset_index()
    )


def estimate_preliminary_price_per_sqft(locality: str, lookup: pd.DataFrame, global_median: float) -> float:
    row = lookup.loc[lookup["locality"] == locality]
    if not row.empty:
        return float(row["median_ppsf"].iloc[0])
    return global_median


def build_underwriting_row(
    *, locality: str, property_type: str, area: float, bedroom_num: int, bathroom_num: int,
    balcony_num: int, furnished: str, age: int, total_floors: int, latitude: float, longitude: float,
    artifacts: dict, lookup: pd.DataFrame, global_median_ppsf: float,
) -> pd.DataFrame:
    """Assemble a single-row raw-schema DataFrame for a hypothetical property
    and run it through the same feature-engineering path as training data.

    NOTE on breaking the valuation/clustering circularity: the trained
    valuation model conditions on `cluster_id`, and cluster assignment uses
    `price_per_sqft` -- which is unknown for a property we haven't valued
    yet. We resolve this the same way a human underwriter would: use the
    locality's historical median price/sq.ft as a *preliminary* comparable
    value purely to place the property in the correct micro-market cluster,
    then let the valuation model produce the actual fair-value estimate
    conditioned on that cluster.
    """
    preliminary_ppsf = estimate_preliminary_price_per_sqft(locality, lookup, global_median_ppsf)
    preliminary_price = preliminary_ppsf * area

    raw_row = pd.DataFrame([{
        "title": "Underwriting Candidate", "price": preliminary_price, "area": area,
        "price_per_sqft": preliminary_ppsf, "locality": locality, "city": "Mumbai",
        "property_type": property_type, "bedroom_num": bedroom_num, "bathroom_num": bathroom_num,
        "balcony_num": balcony_num, "furnished": furnished, "age": age, "total_floors": total_floors,
        "latitude": latitude, "longitude": longitude,
    }])

    engineered, _, _ = engineer_features(
        raw_row, fit_encoders=False,
        artifacts=artifacts["feature_artifacts"], locality_encoder=artifacts["locality_encoder"],
    )

    cluster_id, tier_name = assign_cluster_to_new_property(
        engineered, artifacts["scaler"], artifacts["pca"], artifacts["kmeans"], artifacts["tier_map"],
    )
    engineered["cluster_id"] = cluster_id
    engineered["market_tier"] = tier_name
    return engineered


def render_market_explorer_tab(data: pd.DataFrame, artifacts: dict) -> None:
    st.subheader("Micro-Market Segmentation Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Market Tiers (k)", artifacts["best_k"])
    c2.metric("Silhouette Score", f"{artifacts['silhouette']:.3f}")
    c3.metric("Davies-Bouldin Index", f"{artifacts['davies_bouldin']:.3f}")

    st.dataframe(
        artifacts["cluster_profile"].style.format({
            "mean_price": "{:,.0f}", "median_price_per_sqft": "{:,.0f}",
            "mean_area": "{:,.0f}", "mean_amenity_score": "{:.1f}",
        }),
        use_container_width=True,
    )

    st.divider()
    zones = sorted(data["macro_zone"].astype(str).unique())
    tiers = sorted(data["market_tier"].astype(str).unique())
    col_a, col_b = st.columns(2)
    selected_zones = col_a.multiselect("Filter by macro zone", zones, default=zones)
    selected_tiers = col_b.multiselect("Filter by market tier", tiers, default=tiers)

    filtered = data[data["macro_zone"].astype(str).isin(selected_zones) & data["market_tier"].astype(str).isin(selected_tiers)]
    st.caption(f"{len(filtered):,} properties match the current filter")

    plot_sample = filtered.sample(min(8000, len(filtered)), random_state=42) if len(filtered) else filtered

    map_col, pca_col = st.columns(2)
    with map_col:
        st.markdown("**Geographic Cluster Map**")
        fig_map = px.scatter(
            plot_sample, x="longitude", y="latitude", color="market_tier",
            hover_data=["locality", "price_per_sqft"], opacity=0.6, height=520,
        )
        fig_map.update_layout(legend_title_text="Market Tier")
        st.plotly_chart(fig_map, use_container_width=True)

    with pca_col:
        st.markdown("**PCA Projection of Feature Space**")
        fig_pca = px.scatter(
            plot_sample, x="pca_1", y="pca_2", color="market_tier",
            hover_data=["locality", "price_per_sqft"], opacity=0.6, height=520,
        )
        fig_pca.update_layout(legend_title_text="Market Tier")
        st.plotly_chart(fig_pca, use_container_width=True)


def render_underwriter_tab(artifacts: dict) -> None:
    st.subheader("Single Property Underwriter")

    lookup = locality_lookup_table()
    global_median_ppsf = float(lookup["median_ppsf"].median())
    localities = sorted(lookup["locality"].unique())
    property_types = sorted(artifacts["feature_artifacts"].property_type_map.keys())
    furnished_options = list(FURNISHED_ORDER.keys())

    with st.form("underwriting_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            locality = st.selectbox("Locality", localities, index=localities.index("Andheri") if "Andheri" in localities else 0)
            property_type = st.selectbox("Property Type", property_types)
            furnished = st.selectbox("Furnishing", furnished_options, index=0)
        with col2:
            area = st.number_input("Area (sq. ft.)", min_value=150.0, max_value=10000.0, value=750.0, step=25.0)
            bedroom_num = st.slider("Bedrooms (BHK)", 0, 8, 2)
            bathroom_num = st.slider("Bathrooms", 1, 8, 2)
        with col3:
            balcony_num = st.slider("Balconies", 0, 5, 1)
            age = st.slider("Property Age (years)", 0, 50, 3)
            total_floors = st.slider("Building Floor / Height", 1, 55, 8)

        with st.expander("Advanced: fine-tune coordinates"):
            default_row = lookup.loc[lookup["locality"] == locality]
            default_lat = float(default_row["lat"].iloc[0]) if not default_row.empty else 19.10
            default_lon = float(default_row["lon"].iloc[0]) if not default_row.empty else 72.90
            latitude = st.number_input("Latitude", value=default_lat, format="%.6f")
            longitude = st.number_input("Longitude", value=default_lon, format="%.6f")

        submitted = st.form_submit_button("Underwrite Property", type="primary")

    if not submitted:
        st.info("Fill in the property details above and click **Underwrite Property**.")
        return

    engineered_row = build_underwriting_row(
        locality=locality, property_type=property_type, area=area, bedroom_num=bedroom_num,
        bathroom_num=bathroom_num, balcony_num=balcony_num, furnished=furnished, age=age,
        total_floors=total_floors, latitude=latitude, longitude=longitude,
        artifacts=artifacts, lookup=lookup, global_median_ppsf=global_median_ppsf,
    )

    predicted_price = predict_price(artifacts["model"], engineered_row, artifacts["feature_columns"])
    engineered_row["price"] = predicted_price
    scored_row = compute_property_reit_metrics(engineered_row, value_column="price")

    st.divider()
    st.markdown(f"#### Market Tier: **{engineered_row['market_tier'].iloc[0]}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Fair Value", f"₹{predicted_price:,.0f}")
    m2.metric("Implied ₹/sq.ft.", f"₹{predicted_price / area:,.0f}")
    m3.metric("Est. Monthly Rent", f"₹{scored_row['estimated_monthly_rent'].iloc[0]:,.0f}")
    m4.metric("Simulated Cap Rate", f"{scored_row['cap_rate'].iloc[0] * 100:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Net Operating Income (annual)", f"₹{scored_row['net_operating_income'].iloc[0]:,.0f}")
    m6.metric("Price-to-Rent Ratio", f"{scored_row['price_to_rent_ratio'].iloc[0]:.1f}x")
    m7.metric("Yield Spread vs. Risk-Free", f"{scored_row['yield_spread_vs_riskfree'].iloc[0] * 100:+.2f} pp")
    m8.metric("Risk-Free Rate (10Y G-Sec)", f"{RISK_FREE_RATE * 100:.2f}%")

    st.divider()
    st.markdown("#### Why this valuation? (SHAP local explanation)")
    explainer = get_shap_explainer(artifacts["model"])
    X_row = engineered_row[artifacts["feature_columns"]]
    try:
        fig = plot_shap_waterfall(explainer, X_row)
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Base value and contributions are in log-price space (the model's native "
            "training target). Bars pushing right increase the valuation above the "
            "cluster/portfolio baseline; bars pushing left decrease it."
        )
    except Exception as exc:  # noqa: BLE001 - surface explainability failures without crashing the app
        st.warning(f"SHAP waterfall could not be rendered: {exc}")


def render_screener_tab(scored_universe: pd.DataFrame) -> None:
    st.subheader("REIT Portfolio Screener")

    col1, col2, col3, col4 = st.columns(4)
    zones = sorted(scored_universe["macro_zone"].astype(str).unique())
    tiers = sorted(scored_universe["market_tier"].astype(str).unique())
    selected_zones = col1.multiselect("Macro Zone", zones, default=zones, key="screener_zones")
    selected_tiers = col2.multiselect("Market Tier", tiers, default=tiers, key="screener_tiers")
    min_cap_rate_pct = col3.slider("Min. Cap Rate (%)", 0.0, 10.0, 3.0, step=0.1)
    min_gap_pct = col4.slider("Min. Undervaluation Gap (%)", -20.0, 50.0, 0.0, step=1.0)

    mask = (
        scored_universe["macro_zone"].astype(str).isin(selected_zones)
        & scored_universe["market_tier"].astype(str).isin(selected_tiers)
        & (scored_universe["cap_rate"].fillna(-np.inf) >= min_cap_rate_pct / 100.0)
        & (scored_universe["undervaluation_gap"].fillna(-np.inf) >= min_gap_pct / 100.0)
    )
    screened = scored_universe.loc[mask].copy()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Candidates Found", f"{len(screened):,}")
    s2.metric("Avg. Cap Rate", f"{screened['cap_rate'].mean() * 100:.2f}%" if len(screened) else "n/a")
    s3.metric("Avg. Undervaluation Gap", f"{screened['undervaluation_gap'].mean() * 100:.1f}%" if len(screened) else "n/a")
    s4.metric("Avg. Fair Value", f"₹{screened['predicted_fair_value'].mean():,.0f}" if len(screened) else "n/a")

    display_cols = [
        "locality", "macro_zone", "market_tier", "property_type", "area", "price",
        "predicted_fair_value", "undervaluation_gap", "cap_rate", "estimated_monthly_rent",
        "price_to_rent_ratio",
    ]
    st.dataframe(
        screened[display_cols].sort_values("undervaluation_gap", ascending=False).head(500).style.format({
            "price": "{:,.0f}", "predicted_fair_value": "{:,.0f}", "undervaluation_gap": "{:+.1%}",
            "cap_rate": "{:.2%}", "estimated_monthly_rent": "{:,.0f}", "price_to_rent_ratio": "{:.1f}",
            "area": "{:,.0f}",
        }),
        use_container_width=True, height=420,
    )

    st.divider()
    st.markdown("#### Portfolio Risk-Adjusted Return Simulation")
    st.caption("Monte Carlo simulation over the currently screened candidate set (5-year horizon, "
               "zone-specific appreciation drift/volatility plus simulated rental cash flow).")

    if st.button("Run Portfolio Simulation", type="primary", disabled=screened.empty):
        with st.spinner("Simulating portfolio paths..."):
            sim_sample = screened.sample(min(2000, len(screened)), random_state=42)
            result = simulate_portfolio_returns(sim_sample)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Expected Annualized Return", f"{result.expected_annualized_return * 100:.2f}%")
        r2.metric("Return Volatility", f"{result.return_volatility * 100:.2f}%")
        r3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        r4.metric("5% Value-at-Risk (total return)", f"{result.value_at_risk_95 * 100:.2f}%")
    elif screened.empty:
        st.info("No candidates match the current filters -- widen the cap rate or undervaluation gap.")


def main() -> None:
    st.title("\U0001F3E2 REIT Valuation Engine")
    st.caption("Clustering & Explainable AI for Mumbai Metropolitan Region Real Estate")

    try:
        artifacts = load_artifacts()
        clustered_data = load_clustered_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return

    scored_universe = build_scored_universe(artifacts["model"], tuple(artifacts["feature_columns"]))

    tab1, tab2, tab3 = st.tabs([
        "\U0001F4CD Market Explorer & Cluster Maps",
        "\U0001F4CA Single Property Underwriter",
        "\U0001F50D REIT Portfolio Screener",
    ])
    with tab1:
        render_market_explorer_tab(clustered_data, artifacts)
    with tab2:
        render_underwriter_tab(artifacts)
    with tab3:
        render_screener_tab(scored_universe)


if __name__ == "__main__":
    main()
