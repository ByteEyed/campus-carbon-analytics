"""
Forecasting Baseline Models
===========================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 10: Forecasting Baseline Implementation
Implements:
1. NaiveBaseline: Standard naive flat forecast (y_{T+h} = y_T).
2. SeasonalNaiveBaseline: Seasonal repetition forecast (y_t = y_{t-m}, default m=12).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
import pandas as pd


def _validate_input_series(
    series: Sequence[float] | np.ndarray | pd.Series,
    min_length: int = 1,
) -> np.ndarray:
    """
    Validate that input series is 1D, numeric, non-empty, and free of NaN/Inf.

    Parameters
    ----------
    series : Sequence[float] | np.ndarray | pd.Series
        Input time-series data.
    min_length : int, default 1
        Minimum number of observations required.

    Returns
    -------
    np.ndarray
        Clean 1D float array.

    Raises
    ------
    ValueError
        If series is empty, has fewer than min_length observations, or contains NaN/Inf.
    """
    arr = np.asarray(series, dtype=float).ravel()
    if len(arr) == 0:
        raise ValueError("Time series cannot be empty.")
    if len(arr) < min_length:
        raise ValueError(
            f"Series length ({len(arr)}) is less than required minimum ({min_length})."
        )
    if np.isnan(arr).any():
        raise ValueError("Series contains NaN values. Clean or impute before fitting.")
    if np.isinf(arr).any():
        raise ValueError("Series contains Infinite values.")
    return arr


class BaseBaseline(ABC):
    """Abstract base class for time-series forecasting baselines."""

    @abstractmethod
    def fit(self, series: Sequence[float] | np.ndarray | pd.Series) -> BaseBaseline:
        """Fit the baseline model to historical training observations."""
        pass

    @abstractmethod
    def predict(self, horizon: int = 12) -> np.ndarray:
        """
        Generate point forecasts across the forecast horizon.

        Parameters
        ----------
        horizon : int, default 12
            Number of future time steps to forecast.

        Returns
        -------
        np.ndarray
            1D array of forecasted values of length `horizon`.
        """
        pass


class NaiveBaseline(BaseBaseline):
    """
    Standard Naive Baseline Forecaster (y_{T+h} = y_T).

    Takes the final available historical observation from the training period
    and projects it flatly across all future forecast horizon steps.

    Notes
    -----
    - Formula: y_{T+h} = y_T for all h = 1, 2, ..., H.
    - Captures no trend and no seasonality.
    - Establishes the lower benchmark for time-series predictability.
    """

    def __init__(self) -> None:
        self.last_value_: float | None = None
        self.is_fitted_: bool = False

    def fit(self, series: Sequence[float] | np.ndarray | pd.Series) -> NaiveBaseline:
        """
        Fit standard naive model by recording the final historical observation.

        Parameters
        ----------
        series : Sequence[float] | np.ndarray | pd.Series
            Historical training time series. Must contain at least 1 observation.
        """
        y = _validate_input_series(series, min_length=1)
        self.last_value_ = float(y[-1])
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int = 12) -> np.ndarray:
        """
        Generate flat point forecasts equal to the last historical observation.

        Parameters
        ----------
        horizon : int, default 12
            Number of steps to project forward. Must be a positive integer.

        Returns
        -------
        np.ndarray
            Array of shape (horizon,) where every entry equals y_T.
        """
        if not self.is_fitted_ or self.last_value_ is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer, got {horizon}.")

        return np.full(shape=(horizon,), fill_value=self.last_value_, dtype=float)


class SeasonalNaiveBaseline(BaseBaseline):
    """
    Seasonal Naive Baseline Forecaster (y_t = y_{t-m}).

    Projects future values using the observed values from the corresponding
    month in the previous seasonal cycle. Relies on an explicit seasonal period
    m (default m=12 for monthly institutional academic data).

    Parameters
    ----------
    seasonal_period : int, default 12
        Length of the seasonal cycle (e.g. 12 for monthly academic calendar).
        Must be >= 1.

    Notes
    -----
    - Formula: y_{T+h} = y_{T - m + ((h - 1) % m)} for h = 1, 2, ..., H.
    - For a 12-month horizon (h=12) with m=12, the forecast for each future month
      equals the observation from the exact same calendar month in the previous year.
    - Accurately captures seasonal variations (semester breaks, summer recess,
      peak cooling months), but ignores secular growth trends.
    """

    def __init__(self, seasonal_period: int = 12) -> None:
        if seasonal_period < 1:
            raise ValueError(
                f"Seasonal period must be a positive integer >= 1, got {seasonal_period}."
            )
        self.seasonal_period: int = seasonal_period
        self.history_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, series: Sequence[float] | np.ndarray | pd.Series) -> SeasonalNaiveBaseline:
        """
        Fit seasonal naive model by storing historical observations.

        Requires at least one full seasonal period (len(series) >= seasonal_period).

        Parameters
        ----------
        series : Sequence[float] | np.ndarray | pd.Series
            Historical training time series.
        """
        y = _validate_input_series(series, min_length=self.seasonal_period)
        self.history_ = y
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int = 12) -> np.ndarray:
        """
        Generate seasonal naive point forecasts.

        Parameters
        ----------
        horizon : int, default 12
            Number of future months to forecast.

        Returns
        -------
        np.ndarray
            Array of shape (horizon,) containing seasonal lag predictions.
        """
        if not self.is_fitted_ or self.history_ is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer, got {horizon}.")

        m = self.seasonal_period
        last_cycle = self.history_[-m:]

        # For step h in 1..horizon, index into the last seasonal cycle
        preds = np.array([last_cycle[(h - 1) % m] for h in range(1, horizon + 1)], dtype=float)
        return preds
