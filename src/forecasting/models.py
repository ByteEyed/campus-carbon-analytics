"""
Time-Series Forecasting Models
==============================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Implements:
1. SeasonalNaiveForecaster: Baseline seasonal model (y_t = y_{t-m}) with
   residual-based 95% prediction intervals.
2. HoltWintersForecaster: Exponential Smoothing (additive trend, additive seasonality)
   with state-space simulation-based 95% prediction intervals.
3. ForecastResult: Structured, immutable container for point forecasts and intervals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters.results import HoltWintersResults


@dataclass(frozen=True)
class ForecastResult:
    """
    Immutable container for forecast predictions and uncertainty intervals.

    Parameters
    ----------
    point_forecast : np.ndarray
        Array of point forecasts for the horizon.
    lower_bound : np.ndarray
        Lower bound of the prediction interval (non-negative).
    upper_bound : np.ndarray
        Upper bound of the prediction interval.
    confidence_level : float
        Nominal confidence level for the prediction interval (e.g., 0.95).
    model_name : str
        Name/identifier of the model that generated this forecast.
    residuals : np.ndarray | None
        In-sample training residuals, if available.
    """

    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float = 0.95
    model_name: str = ""
    residuals: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate array shapes, values, and interval consistency."""
        pt = np.asarray(self.point_forecast, dtype=float).ravel()
        lb = np.asarray(self.lower_bound, dtype=float).ravel()
        ub = np.asarray(self.upper_bound, dtype=float).ravel()

        if len(pt) == 0:
            raise ValueError("Point forecast cannot be empty.")
        if len(pt) != len(lb) or len(pt) != len(ub):
            raise ValueError(
                f"Dimension mismatch in ForecastResult: point({len(pt)}), "
                f"lower({len(lb)}), upper({len(ub)})."
            )

        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(
                f"Confidence level must be between 0 and 1 exclusive, got {self.confidence_level}."
            )

        if (lb < 0.0).any():
            raise ValueError("Lower bound must be non-negative for physical emissions.")

        if (lb > ub).any():
            raise ValueError("Lower bound cannot exceed upper bound.")

        # Reassign 1D flattened float arrays to frozen dataclass
        object.__setattr__(self, "point_forecast", pt)
        object.__setattr__(self, "lower_bound", lb)
        object.__setattr__(self, "upper_bound", ub)
        if self.residuals is not None:
            res = np.asarray(self.residuals, dtype=float).ravel()
            object.__setattr__(self, "residuals", res)

    @property
    def mean(self) -> np.ndarray:
        """Alias for point forecast matching statsmodels convention."""
        return self.point_forecast

    @property
    def mean_ci_lower(self) -> np.ndarray:
        """Alias for lower bound matching statsmodels convention."""
        return self.lower_bound

    @property
    def mean_ci_upper(self) -> np.ndarray:
        """Alias for upper bound matching statsmodels convention."""
        return self.upper_bound

    def summary_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return DataFrame matching statsmodels summary_frame convention.

        Columns: ['mean', 'mean_ci_lower', 'mean_ci_upper']
        """
        return pd.DataFrame({
            "mean": self.point_forecast,
            "mean_ci_lower": self.lower_bound,
            "mean_ci_upper": self.upper_bound,
        })

    def to_dataframe(
        self,
        dates: Sequence[str] | pd.DatetimeIndex | pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Convert forecast and prediction intervals into a pandas DataFrame.

        Parameters
        ----------
        dates : Sequence[str] | pd.DatetimeIndex | pd.Series | None, default None
            Optional sequence of dates for index/column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['point_forecast', 'lower_bound', 'upper_bound']
            (and 'date' if provided).
        """
        data: dict[str, Any] = {
            "point_forecast": self.point_forecast,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }
        if dates is not None:
            if len(dates) != len(self.point_forecast):
                raise ValueError(
                    f"Length of dates ({len(dates)}) must match forecast horizon ({len(self.point_forecast)})."
                )
            df = pd.DataFrame(data)
            df.insert(0, "date", list(dates))
            return df

        return pd.DataFrame(data)


def _validate_input_series(
    series: Sequence[float] | np.ndarray | pd.Series,
    min_length: int = 1,
) -> np.ndarray:
    """Validate that input series is 1D, numeric, non-empty, and free of NaN/Inf."""
    arr = np.asarray(series, dtype=float).ravel()
    if len(arr) < min_length:
        raise ValueError(
            f"Series length ({len(arr)}) is less than required minimum ({min_length})."
        )
    if np.isnan(arr).any():
        raise ValueError("Series contains NaN values. Clean or impute before fitting.")
    if np.isinf(arr).any():
        raise ValueError("Series contains Infinite values.")
    return arr


class BaseForecaster(ABC):
    """Abstract base class for emissions forecasters."""

    @abstractmethod
    def fit(self, series: Sequence[float] | np.ndarray | pd.Series) -> BaseForecaster:
        """Fit the forecasting model to historical time-series data."""
        pass

    @abstractmethod
    def predict(
        self,
        steps: int = 12,
        confidence_level: float = 0.95,
    ) -> ForecastResult:
        """Generate point forecasts and prediction intervals for future steps."""
        pass


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive Baseline Model: y_{T+h} = y_{T+h-m(k+1)}.

    Parameters
    ----------
    seasonal_period : int, default 12
        The length of the seasonal cycle (e.g., 12 for monthly academic data).
    """

    def __init__(self, seasonal_period: int = 12) -> None:
        if seasonal_period < 1:
            raise ValueError(f"Seasonal period must be >= 1, got {seasonal_period}.")
        self.seasonal_period: int = seasonal_period
        self._history: np.ndarray | None = None
        self._residual_std: float | None = None
        self._residuals: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether model has been fitted."""
        return self._history is not None

    def fit(self, series: Sequence[float] | np.ndarray | pd.Series) -> SeasonalNaiveForecaster:
        """
        Fit Seasonal Naive model.

        Requires at least one full seasonal cycle (len(series) >= seasonal_period).
        Computes in-sample seasonal residuals (y_t - y_{t-m}) for prediction intervals.
        """
        y = _validate_input_series(series, min_length=self.seasonal_period)
        self._history = y

        m = self.seasonal_period
        if len(y) > m:
            # Seasonal differences: e_t = y_t - y_{t-m}
            diffs = y[m:] - y[:-m]
            self._residuals = diffs
            # Sample standard deviation of seasonal jumps
            self._residual_std = float(np.std(diffs, ddof=1 if len(diffs) > 1 else 0))
            if self._residual_std == 0.0:
                self._residual_std = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
        else:
            self._residuals = np.array([])
            self._residual_std = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0

        return self

    def predict(
        self,
        steps: int = 12,
        confidence_level: float = 0.95,
    ) -> ForecastResult:
        """
        Generate seasonal naive point forecasts and prediction intervals.

        Point forecast:
            y_{T+h} = y_{T - m + ((h - 1) % m)}

        Standard error at horizon h:
            SE(h) = s_e * sqrt(floor((h - 1) / m) + 1)
        """
        if not self.is_fitted or self._history is None or self._residual_std is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        if steps <= 0:
            raise ValueError(f"Steps must be a positive integer, got {steps}.")

        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"Confidence level must be between 0 and 1 exclusive, got {confidence_level}."
            )

        m = self.seasonal_period
        last_cycle = self._history[-m:]

        # Point predictions
        point_preds = np.array([last_cycle[i % m] for i in range(steps)], dtype=float)

        # Standard error scales with number of seasonal cycles ahead
        h_indices = np.arange(1, steps + 1)
        cycle_multipliers = np.sqrt(np.floor((h_indices - 1) / m) + 1.0)
        se_h = self._residual_std * cycle_multipliers

        # Normal distribution critical value z
        alpha = 1.0 - confidence_level
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))

        margin = z * se_h
        lower_bound = np.maximum(0.0, point_preds - margin)
        upper_bound = point_preds + margin

        return ForecastResult(
            point_forecast=point_preds,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            model_name="SeasonalNaive",
            residuals=self._residuals,
        )


# Re-export canonical HoltWintersForecaster from holt_winters.py lazily via PEP 562
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.forecasting.holt_winters import HoltWintersForecaster


def __getattr__(name: str) -> Any:
    """Lazy import fallback for HoltWintersForecaster to prevent circular import."""
    if name == "HoltWintersForecaster":
        from src.forecasting.holt_winters import HoltWintersForecaster
        return HoltWintersForecaster
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ForecastResult",
    "BaseForecaster",
    "SeasonalNaiveForecaster",
    "HoltWintersForecaster",
]

