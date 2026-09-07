"""REIT-oriented quantitative metrics: NOI, cap rate, yield spread, and a
Monte Carlo risk-adjusted portfolio return simulation.

All formulas are institutional heuristics calibrated to typical MMR
residential-rental benchmarks (see `src.config` for the underlying
assumptions) -- they are designed for relative screening/ranking across a
portfolio, not as a substitute for a bottom-up appraisal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    APPRECIATION_MEAN_BY_ZONE,
    APPRECIATION_VOLATILITY_BY_ZONE,
    BASE_GROSS_YIELD_BY_ZONE,
    MONTE_CARLO_HORIZON_YEARS,
    N_MONTE_CARLO_PATHS,
    OPERATING_EXPENSE_RATIO,
    RANDOM_SEED,
    RISK_FREE_RATE,
    VACANCY_RATE,
    ZONE_THANE_EXTENDED,
    get_logger,
)

logger = get_logger(__name__)

FURNISHED_RENT_ADJUSTMENT: dict[str, float] = {"Unfurnished": -0.03, "Semi-Furnished": 0.0, "Furnished": 0.05}
_MIN_ADJUSTMENT, _MAX_ADJUSTMENT = 0.75, 1.35
_EPSILON = 1e-9


@dataclass
class PortfolioSimulationResult:
    """Monte Carlo risk-adjusted return summary for a property subset."""

    n_properties: int
    n_paths: int
    horizon_years: int
    expected_annualized_return: float
    return_volatility: float
    sharpe_ratio: float
    value_at_risk_95: float
    expected_total_return: float


def _rent_adjustment_factor(amenity_score: pd.Series, furnished: pd.Series) -> pd.Series:
    furnished_component = furnished.map(FURNISHED_RENT_ADJUSTMENT).fillna(0.0)
    amenity_component = 0.002 * (amenity_score - 50.0)
    factor = 1.0 + furnished_component + amenity_component
    return factor.clip(lower=_MIN_ADJUSTMENT, upper=_MAX_ADJUSTMENT)


def compute_property_reit_metrics(df: pd.DataFrame, value_column: str = "price") -> pd.DataFrame:
    """Append rent, NOI, cap-rate, price-to-rent, and yield-spread columns.

    `value_column` lets callers pass either the observed transaction price or
    a model-predicted fair value as the basis for the yield calculations.
    """
    df = df.copy()
    zone_yield = df["macro_zone"].astype(str).map(BASE_GROSS_YIELD_BY_ZONE).fillna(
        BASE_GROSS_YIELD_BY_ZONE[ZONE_THANE_EXTENDED]
    )
    adjustment = _rent_adjustment_factor(df["amenity_score"], df["furnished"])

    property_value = df[value_column].clip(lower=_EPSILON)
    df["estimated_annual_rent"] = (property_value * zone_yield * adjustment).round(0)
    df["estimated_monthly_rent"] = (df["estimated_annual_rent"] / 12.0).round(0)

    df["net_operating_income"] = (
        df["estimated_annual_rent"] * (1 - OPERATING_EXPENSE_RATIO) * (1 - VACANCY_RATE)
    ).round(0)

    df["cap_rate"] = np.where(
        property_value > _EPSILON, df["net_operating_income"] / property_value, np.nan
    )
    df["price_to_rent_ratio"] = np.where(
        df["estimated_annual_rent"] > _EPSILON, property_value / df["estimated_annual_rent"], np.nan
    )
    df["yield_spread_vs_riskfree"] = df["cap_rate"] - RISK_FREE_RATE

    return df


def compute_undervaluation_gap(actual_price: pd.Series, predicted_fair_value: pd.Series) -> pd.Series:
    """Positive gap = property trades below the model's fair-value estimate
    (a candidate acquisition target); negative = trades above fair value."""
    safe_actual = actual_price.clip(lower=_EPSILON)
    return ((predicted_fair_value - actual_price) / safe_actual).round(4)


def simulate_portfolio_returns(
    df: pd.DataFrame,
    horizon_years: int = MONTE_CARLO_HORIZON_YEARS,
    n_paths: int = N_MONTE_CARLO_PATHS,
    random_seed: int = RANDOM_SEED,
) -> PortfolioSimulationResult:
    """Monte Carlo simulation of equal-weighted portfolio total return
    (price appreciation + cumulative rental income), combining each
    property's zone-specific appreciation mean/volatility with its cap rate.

    Each property's terminal value multiple is drawn from a lognormal
    distribution parameterized by its zone's assumed annual appreciation
    mean/volatility; annual rental cash flow is assumed to grow with price.
    """
    if df.empty:
        raise ValueError("Cannot simulate returns for an empty property selection.")

    rng = np.random.default_rng(random_seed)
    zones = df["macro_zone"].astype(str).to_numpy()
    cap_rates = df["cap_rate"].fillna(df["cap_rate"].median()).to_numpy()
    n_properties = len(df)

    mu = np.array([APPRECIATION_MEAN_BY_ZONE.get(z, 0.07) for z in zones])
    sigma = np.array([APPRECIATION_VOLATILITY_BY_ZONE.get(z, 0.10) for z in zones])

    # Lognormal terminal appreciation multiple per property per path.
    # Shapes: (n_paths, n_properties)
    drift = (mu - 0.5 * sigma**2) * horizon_years
    diffusion = sigma * np.sqrt(horizon_years) * rng.standard_normal((n_paths, n_properties))
    price_multiple = np.exp(drift + diffusion)

    price_return = price_multiple - 1.0
    rental_return = cap_rates * horizon_years  # simple cumulative income yield over the horizon
    total_return_per_property = price_return + rental_return

    portfolio_return_per_path = total_return_per_property.mean(axis=1)
    annualized_return_per_path = (1.0 + portfolio_return_per_path) ** (1.0 / horizon_years) - 1.0

    expected_annualized = float(np.mean(annualized_return_per_path))
    volatility = float(np.std(annualized_return_per_path))
    sharpe = float((expected_annualized - RISK_FREE_RATE) / volatility) if volatility > _EPSILON else 0.0
    var_95 = float(-np.percentile(portfolio_return_per_path, 5))

    logger.info(
        "Portfolio MC sim (%d properties, %d paths, %dY): E[ann. return]=%.2f%%, vol=%.2f%%, Sharpe=%.2f",
        n_properties, n_paths, horizon_years, 100 * expected_annualized, 100 * volatility, sharpe,
    )

    return PortfolioSimulationResult(
        n_properties=n_properties, n_paths=n_paths, horizon_years=horizon_years,
        expected_annualized_return=expected_annualized, return_volatility=volatility,
        sharpe_ratio=sharpe, value_at_risk_95=var_95,
        expected_total_return=float(np.mean(portfolio_return_per_path)),
    )


def screen_portfolio(
    df: pd.DataFrame, min_cap_rate: float = 0.0, min_undervaluation_gap: float = 0.0,
) -> pd.DataFrame:
    """Filter a scored property universe for REIT acquisition candidates."""
    mask = pd.Series(True, index=df.index)
    if "cap_rate" in df.columns:
        mask &= df["cap_rate"].fillna(-np.inf) >= min_cap_rate
    if "undervaluation_gap" in df.columns:
        mask &= df["undervaluation_gap"].fillna(-np.inf) >= min_undervaluation_gap
    return df.loc[mask].copy()
