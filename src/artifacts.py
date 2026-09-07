"""Dataclasses and encoder classes that get serialized via joblib.

These live in their own module -- never executed as `__main__` -- so their
pickled `__module__` reference stays stable regardless of which entrypoint
(`python -m src.data_pipeline`, `python -m src.valuation_model`, or the
Streamlit app) created or loads them. Defining a class inside a script that
is sometimes run directly (`python -m src.foo` sets that module's
`__name__` to `"__main__"`) bakes `__main__.ClassName` into the pickle,
which then fails to unpickle from any other entrypoint with
`AttributeError: Can't get attribute 'ClassName' on <module '__main__'>`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class FeatureArtifacts:
    """Bundle of fitted encoders needed to transform new/unseen properties."""

    property_type_map: dict[str, int] = field(default_factory=dict)
    macro_zone_map: dict[str, int] = field(default_factory=dict)
    locality_freq_map: dict[str, float] = field(default_factory=dict)
    global_freq_fallback: float = 0.0
    global_log_price_mean: float = 0.0
    amenity_score_min: float = 0.0
    amenity_score_max: float = 1.0


class LocalityTargetEncoder:
    """K-fold, smoothed mean-target encoder for the high-cardinality `locality` column.

    Fitting produces out-of-fold encoded values for the training set (so the
    model never sees a row's own target baked into its encoded feature), plus
    a full-data locality -> encoded-value mapping used at inference time for
    localities seen during training. Unseen localities fall back to the
    global target mean.
    """

    def __init__(self, n_splits: int = 5, smoothing: float = 20.0, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_: float = 0.0
        self.mapping_: dict[str, float] = {}

    @staticmethod
    def _smoothed_means(df: pd.DataFrame, col: str, target: str, global_mean: float,
                         smoothing: float) -> pd.Series:
        agg = df.groupby(col)[target].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        return smoothed

    def fit_transform(self, df: pd.DataFrame, col: str, target: str) -> np.ndarray:
        self.global_mean_ = float(df[target].mean())
        out_of_fold = np.full(len(df), self.global_mean_, dtype=float)

        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, holdout_idx in kfold.split(df):
            train_fold = df.iloc[train_idx]
            fold_means = self._smoothed_means(train_fold, col, target, self.global_mean_, self.smoothing)
            holdout_vals = df.iloc[holdout_idx][col].map(fold_means).fillna(self.global_mean_)
            out_of_fold[holdout_idx] = holdout_vals.to_numpy()

        self.mapping_ = self._smoothed_means(df, col, target, self.global_mean_, self.smoothing).to_dict()
        return out_of_fold

    def transform(self, df: pd.DataFrame, col: str) -> np.ndarray:
        return df[col].map(self.mapping_).fillna(self.global_mean_).to_numpy()


@dataclass
class ModelEvaluation:
    """Original-scale (INR) regression metrics for one trained model."""

    model_name: str
    rmse: float
    mae: float
    r2: float
    mape: float
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_rmse_mean: float = 0.0
    cv_rmse_std: float = 0.0
