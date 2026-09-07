"""Explainable AI: SHAP (global + local) and LIME (local) explanations for
the valuation model, for institutional decision-makers who need to see
*why* a property was valued the way it was, not just the number.
"""
from __future__ import annotations

from typing import Any

import lime.lime_tabular
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")  # headless rendering for Streamlit / batch scripts

from src.config import get_logger

logger = get_logger(__name__)


def build_shap_explainer(model: Any) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for a fitted LightGBM/XGBoost regressor.
    Works directly on log-price predictions (the model's native output space)."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return SHAP values array of shape (n_samples, n_features)."""
    values = explainer.shap_values(X)
    return np.asarray(values)


def plot_shap_summary(
    shap_values: np.ndarray, X: pd.DataFrame, max_display: int = 15
) -> plt.Figure:
    """Global feature-importance summary (beeswarm) plot."""
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, max_display: int = 15) -> plt.Figure:
    """Global mean(|SHAP|) bar chart -- simpler alternative to the beeswarm."""
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_waterfall(
    explainer: shap.TreeExplainer, X_row: pd.DataFrame, max_display: int = 12
) -> plt.Figure:
    """Local explanation for a single property: which features pushed the
    valuation above/below the model's baseline (expected log-price)."""
    if len(X_row) != 1:
        raise ValueError("plot_shap_waterfall expects a single-row DataFrame.")

    explanation = explainer(X_row)
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray, X: pd.DataFrame, feature_name: str, interaction_feature: str | None = "auto"
) -> plt.Figure:
    """Dependence plot: how SHAP contribution for one feature varies with its value
    (optionally colored by an interacting feature)."""
    fig = plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature_name, shap_values, X, interaction_index=interaction_feature, show=False, ax=plt.gca()
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def summarize_local_shap(
    explainer: shap.TreeExplainer, X_row: pd.DataFrame, top_n: int = 8
) -> pd.DataFrame:
    """Tabular (non-plot) local explanation: top contributing features, their
    values, and signed SHAP contribution -- useful for a Streamlit dataframe
    or a plain-text underwriting note."""
    explanation = explainer(X_row)
    values = explanation.values[0]
    base_value = explanation.base_values[0]
    feature_names = X_row.columns

    contrib = pd.DataFrame({
        "feature": feature_names,
        "value": X_row.iloc[0].to_numpy(),
        "shap_contribution": values,
    })
    contrib["abs_contribution"] = contrib["shap_contribution"].abs()
    contrib = contrib.sort_values("abs_contribution", ascending=False).head(top_n)
    contrib.attrs["base_value_log_price"] = float(base_value)
    return contrib.drop(columns="abs_contribution").reset_index(drop=True)


def build_lime_explainer(
    X_train: pd.DataFrame, categorical_feature_names: list[str] | None = None
) -> lime.lime_tabular.LimeTabularExplainer:
    """Build a LIME tabular explainer over the training feature distribution."""
    categorical_feature_names = categorical_feature_names or []
    categorical_idx = [X_train.columns.get_loc(c) for c in categorical_feature_names if c in X_train.columns]

    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=list(X_train.columns),
        categorical_features=categorical_idx,
        mode="regression",
        discretize_continuous=True,
        verbose=False,
    )


def explain_instance_lime(
    explainer: lime.lime_tabular.LimeTabularExplainer, model: Any, instance: pd.DataFrame, num_features: int = 10
) -> lime.explanation.Explanation:
    """LIME local explanation for a single property. `model.predict` must
    accept a 2D numpy array and return log-price predictions."""
    if len(instance) != 1:
        raise ValueError("explain_instance_lime expects a single-row DataFrame.")

    return explainer.explain_instance(
        data_row=instance.to_numpy(dtype=float)[0],
        predict_fn=lambda arr: model.predict(arr),
        num_features=num_features,
    )
