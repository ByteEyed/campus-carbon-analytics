"""
Forecasting Evaluation Metrics
==============================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Implements academically defensible evaluation metrics for time-series forecasting:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Coverage Probability for Prediction Intervals
- Residual Analysis & Baseline Relative Improvement
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastMetrics:
    """Container for forecast accuracy and uncertainty evaluation metrics."""

    mae: float
    rmse: float
    mape: float
    coverage_pct: float | None = None
    mean_residual: float | None = None
    std_residual: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert metrics to dictionary."""
        return asdict(self)


def _validate_arrays(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to clean 1D float numpy arrays and validate dimensions."""
    y = np.asarray(actual, dtype=float).ravel()
    y_hat = np.asarray(predicted, dtype=float).ravel()

    if len(y) == 0:
        raise ValueError("Actual series cannot be empty.")
    if len(y_hat) == 0:
        raise ValueError("Predicted series cannot be empty.")
    if len(y) != len(y_hat):
        raise ValueError(
            f"Dimension mismatch: actual has {len(y)} elements, "
            f"predicted has {len(y_hat)} elements."
        )

    if np.isnan(y).any() or np.isnan(y_hat).any():
        raise ValueError("Input arrays must not contain NaN values.")
    if np.isinf(y).any() or np.isinf(y_hat).any():
        raise ValueError("Input arrays must not contain Infinite values.")

    return y, y_hat


def calculate_mae(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Formula:
        MAE = (1 / n) * sum(|actual - predicted|)
    """
    y, y_hat = _validate_arrays(actual, predicted)
    return round(float(np.mean(np.abs(y - y_hat))), 4)


def calculate_rmse(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Formula:
        RMSE = sqrt((1 / n) * sum((actual - predicted)^2))
    """
    y, y_hat = _validate_arrays(actual, predicted)
    return round(float(np.sqrt(np.mean((y - y_hat) ** 2))), 4)


def calculate_mape(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Formula:
        MAPE = (100 / n) * sum(|(actual - predicted) / actual|)

    Raises
    ------
    ValueError
        If actual series contains zero values (to prevent division by zero).
    """
    y, y_hat = _validate_arrays(actual, predicted)
    if (y == 0.0).any():
        raise ValueError("Cannot calculate MAPE when actual values contain zero (division by zero).")
    return round(float(np.mean(np.abs((y - y_hat) / y)) * 100.0), 4)


def calculate_coverage_probability(
    actual: Sequence[float] | np.ndarray | pd.Series,
    lower_bound: Sequence[float] | np.ndarray | pd.Series,
    upper_bound: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """
    Calculate the empirical coverage probability of prediction intervals.

    Formula:
        Coverage (%) = (100 / n) * sum(lower_bound <= actual <= upper_bound)
    """
    y = np.asarray(actual, dtype=float).ravel()
    lb = np.asarray(lower_bound, dtype=float).ravel()
    ub = np.asarray(upper_bound, dtype=float).ravel()

    if len(y) != len(lb) or len(y) != len(ub):
        raise ValueError("Dimensions of actual, lower_bound, and upper_bound must match.")

    within_bounds = (y >= lb) & (y <= ub)
    return round(float(np.mean(within_bounds) * 100.0), 2)


def calculate_residuals(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Calculate point-by-point forecast residuals.

    Formula:
        residuals = actual - predicted
    """
    y, y_hat = _validate_arrays(actual, predicted)
    return y - y_hat


def calculate_improvement(baseline_metric: float, model_metric: float) -> float:
    """
    Calculate percentage improvement of a primary model relative to a baseline model.

    Formula:
        Improvement (%) = ((baseline_metric - model_metric) / baseline_metric) * 100

    Positive value indicates model outperforms baseline (lower error).
    """
    if baseline_metric <= 0:
        raise ValueError("Baseline metric must be strictly positive.")
    return round(((baseline_metric - model_metric) / baseline_metric) * 100.0, 2)


def evaluate_forecast(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
    lower_bound: Sequence[float] | np.ndarray | pd.Series | None = None,
    upper_bound: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> ForecastMetrics:
    """
    Compute comprehensive accuracy and uncertainty evaluation metrics.

    Parameters
    ----------
    actual : Sequence[float]
        True observed test series values.
    predicted : Sequence[float]
        Point forecast predictions.
    lower_bound : Sequence[float] | None, default None
        Lower bound of prediction interval.
    upper_bound : Sequence[float] | None, default None
        Upper bound of prediction interval.

    Returns
    -------
    ForecastMetrics
        Structured container with MAE, RMSE, MAPE, coverage %, and residual stats.
    """
    y, y_hat = _validate_arrays(actual, predicted)
    mae = calculate_mae(y, y_hat)
    rmse = calculate_rmse(y, y_hat)
    mape = calculate_mape(y, y_hat)

    residuals = y - y_hat
    mean_res = round(float(np.mean(residuals)), 4)
    std_res = round(float(np.std(residuals, ddof=1)), 4) if len(residuals) > 1 else 0.0

    coverage = None
    if lower_bound is not None and upper_bound is not None:
        coverage = calculate_coverage_probability(y, lower_bound, upper_bound)

    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        coverage_pct=coverage,
        mean_residual=mean_res,
        std_residual=std_res,
    )
