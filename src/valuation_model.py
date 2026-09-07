"""Supervised fair-market valuation engine.

Trains LightGBM and XGBoost regressors on the log-transformed price target,
tunes both with Optuna over 5-fold cross-validation, evaluates on a held-out
test split (RMSE, MAE, R^2, MAPE -- all reported in original INR scale), and
persists the better-performing model via joblib.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from src.artifacts import ModelEvaluation
from src.config import (
    CLUSTERED_DATA_PATH,
    CV_FOLDS,
    LOG_TARGET_COLUMN,
    MODEL_METRICS_PATH,
    MODEL_N_JOBS,
    OPTUNA_CV_FOLDS,
    OPTUNA_TIMEOUT_SECONDS,
    OPTUNA_TRIALS,
    RANDOM_SEED,
    TARGET_COLUMN,
    VALUATION_MODEL_PATH,
    get_logger,
)
from src.data_pipeline import MODEL_FEATURE_COLUMNS

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAINING_FEATURE_COLUMNS: list[str] = MODEL_FEATURE_COLUMNS + ["cluster_id"]


def _rmse_log_cv(model_ctor, params: dict[str, Any], X: np.ndarray, y_log: np.ndarray,
                  n_splits: int = OPTUNA_CV_FOLDS) -> float:
    """Mean CV RMSE in log space (Optuna objective -- cheap to evaluate).

    Uses a lighter fold count than the final reported cross-validation metric
    (see OPTUNA_CV_FOLDS docs in src.config) purely to bound how many model
    fits the hyperparameter search performs.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, val_idx in kfold.split(X):
        model = model_ctor(**params)
        model.fit(X[train_idx], y_log[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(float(np.sqrt(np.mean((preds - y_log[val_idx]) ** 2))))
    return float(np.mean(scores))


def _lgbm_ctor(**params: Any) -> LGBMRegressor:
    return LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1, n_jobs=MODEL_N_JOBS, **params)


def _xgb_ctor(**params: Any) -> XGBRegressor:
    return XGBRegressor(random_state=RANDOM_SEED, verbosity=0, tree_method="hist",
                         n_jobs=MODEL_N_JOBS, **params)


def tune_lightgbm(X: np.ndarray, y_log: np.ndarray, n_trials: int = OPTUNA_TRIALS) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        try:
            return _rmse_log_cv(_lgbm_ctor, params, X, y_log)
        except Exception as exc:  # noqa: BLE001 - isolate a bad trial instead of killing the whole study
            logger.warning("LightGBM trial failed (%s); pruning.", exc)
            raise optuna.TrialPruned() from exc

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT_SECONDS, show_progress_bar=False)
    logger.info("LightGBM Optuna best CV log-RMSE=%.5f | params=%s", study.best_value, study.best_params)
    return study.best_params


def tune_xgboost(X: np.ndarray, y_log: np.ndarray, n_trials: int = OPTUNA_TRIALS) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        try:
            return _rmse_log_cv(_xgb_ctor, params, X, y_log)
        except Exception as exc:  # noqa: BLE001 - isolate a bad trial instead of killing the whole study
            logger.warning("XGBoost trial failed (%s); pruning.", exc)
            raise optuna.TrialPruned() from exc

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT_SECONDS, show_progress_bar=False)
    logger.info("XGBoost Optuna best CV log-RMSE=%.5f | params=%s", study.best_value, study.best_params)
    return study.best_params


def evaluate_on_holdout(model, X_test: np.ndarray, y_test_actual: np.ndarray, model_name: str,
                         best_params: dict[str, Any], cv_rmse_mean: float, cv_rmse_std: float) -> ModelEvaluation:
    """Predict in log space, invert to INR, and score against actual prices."""
    log_preds = model.predict(X_test)
    price_preds = np.expm1(log_preds)
    price_preds = np.clip(price_preds, a_min=1.0, a_max=None)

    rmse = float(np.sqrt(np.mean((price_preds - y_test_actual) ** 2)))
    mae = float(mean_absolute_error(y_test_actual, price_preds))
    r2 = float(r2_score(y_test_actual, price_preds))
    mape = float(mean_absolute_percentage_error(y_test_actual, price_preds))

    logger.info("%s holdout -> RMSE=%.0f MAE=%.0f R2=%.4f MAPE=%.4f", model_name, rmse, mae, r2, mape)
    return ModelEvaluation(
        model_name=model_name, rmse=rmse, mae=mae, r2=r2, mape=mape,
        best_params=best_params, cv_rmse_mean=cv_rmse_mean, cv_rmse_std=cv_rmse_std,
    )


def train_valuation_models(
    df: pd.DataFrame | None = None, n_trials: int = OPTUNA_TRIALS, save: bool = True
) -> tuple[Any, dict[str, ModelEvaluation]]:
    """Train, tune, and evaluate LightGBM + XGBoost; persist the better model."""
    if df is None:
        df = pd.read_parquet(CLUSTERED_DATA_PATH)

    X = df[TRAINING_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_log = df[LOG_TARGET_COLUMN].to_numpy(dtype=float)
    y_actual = df[TARGET_COLUMN].to_numpy(dtype=float)

    X_train, X_test, y_log_train, y_log_test, y_actual_train, y_actual_test = train_test_split(
        X, y_log, y_actual, test_size=0.2, random_state=RANDOM_SEED
    )

    logger.info("Training on %d rows, holding out %d rows for evaluation", len(X_train), len(X_test))

    lgbm_params = tune_lightgbm(X_train, y_log_train, n_trials=n_trials)
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    lgbm_cv_scores = [
        float(np.sqrt(np.mean((
            _lgbm_ctor(**lgbm_params).fit(X_train[tr], y_log_train[tr]).predict(X_train[va]) - y_log_train[va]
        ) ** 2)))
        for tr, va in kfold.split(X_train)
    ]
    lgbm_model = _lgbm_ctor(**lgbm_params).fit(X_train, y_log_train)
    lgbm_eval = evaluate_on_holdout(
        lgbm_model, X_test, y_actual_test, "LightGBM", lgbm_params,
        float(np.mean(lgbm_cv_scores)), float(np.std(lgbm_cv_scores)),
    )

    xgb_params = tune_xgboost(X_train, y_log_train, n_trials=n_trials)
    xgb_cv_scores = [
        float(np.sqrt(np.mean((
            _xgb_ctor(**xgb_params).fit(X_train[tr], y_log_train[tr]).predict(X_train[va]) - y_log_train[va]
        ) ** 2)))
        for tr, va in kfold.split(X_train)
    ]
    xgb_model = _xgb_ctor(**xgb_params).fit(X_train, y_log_train)
    xgb_eval = evaluate_on_holdout(
        xgb_model, X_test, y_actual_test, "XGBoost", xgb_params,
        float(np.mean(xgb_cv_scores)), float(np.std(xgb_cv_scores)),
    )

    evaluations = {"LightGBM": lgbm_eval, "XGBoost": xgb_eval}
    best_name = min(evaluations, key=lambda k: evaluations[k].rmse)
    best_model = lgbm_model if best_name == "LightGBM" else xgb_model
    logger.info("Best model: %s (RMSE=%.0f, R2=%.4f)", best_name, evaluations[best_name].rmse,
                evaluations[best_name].r2)

    # Refit the winning model on the full dataset (train + holdout) for deployment,
    # keeping the holdout-derived metrics above as the reported generalization estimate.
    best_params = lgbm_params if best_name == "LightGBM" else xgb_params
    ctor = _lgbm_ctor if best_name == "LightGBM" else _xgb_ctor
    final_model = ctor(**best_params).fit(X, y_log)

    if save:
        joblib.dump(
            {"model": final_model, "model_name": best_name, "feature_columns": TRAINING_FEATURE_COLUMNS},
            VALUATION_MODEL_PATH,
        )
        joblib.dump(evaluations, MODEL_METRICS_PATH)
        logger.info("Saved best model (%s) to %s", best_name, VALUATION_MODEL_PATH)

    return final_model, evaluations


def predict_price(model, feature_row: pd.DataFrame, feature_columns: list[str] = TRAINING_FEATURE_COLUMNS) -> float:
    """Predict fair-market price (INR) for a single-row feature DataFrame."""
    X = feature_row[feature_columns].to_numpy(dtype=float)
    log_pred = model.predict(X)[0]
    return float(np.expm1(log_pred))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the REIT valuation engine's regressors.")
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                         help="Optuna trials per model (overrides OPTUNA_TRIALS env var).")
    args = parser.parse_args()

    train_valuation_models(n_trials=args.trials, save=True)
