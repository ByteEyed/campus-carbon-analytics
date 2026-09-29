"""
Primary Forecasting Model: Holt-Winters Exponential Smoothing
============================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 11: Primary Forecasting Model Implementation
Model: Exponential Smoothing (Holt-Winters) with Additive Trend and Additive Seasonality.

Key Characteristics:
- Level, Trend (additive), and Seasonality (additive, m=12).
- Designed for short, strongly seasonal time series with secular drift (~2.5% annual growth).
- Generates point forecasts and 95% state-space simulated prediction intervals.
- Enforces physical non-negativity and interval sandwiching: lower <= point <= upper.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters.results import HoltWintersResults

from src.forecasting.baselines import _validate_input_series
from src.forecasting.metrics import (
    calculate_improvement,
    calculate_mae,
    calculate_mape,
    calculate_rmse,
)

logger = logging.getLogger("campus_carbon.forecasting.holt_winters")

# Canonical prediction container from models.py
from src.forecasting.models import ForecastResult

# Backward-compatibility alias for Phase 11 evaluations and tests
HoltWintersPrediction = ForecastResult


class HoltWintersForecaster:
    """
    Holt-Winters Exponential Smoothing Forecaster.

    Configured with additive trend and additive seasonality (m=12)
    per Phase 11 specification. Uncertainty is quantified using state-space
    simulated prediction intervals (Hyndman & Athanasopoulos).

    Parameters
    ----------
    trend : str, default "add"
        Type of trend component ('add').
    seasonal : str, default "add"
        Type of seasonal component ('add').
    seasonal_periods : int, default 12
        Length of seasonal cycle (m=12 for monthly academic calendar).
    damped_trend : bool, default False
        Whether to dampen the trend.
    initialization_method : str, default "estimated"
        Initialization method ('estimated', 'heuristic').
    random_state : int | None, default 42
        Random seed for state-space simulation intervals.
    simulation_repetitions : int, default 2000
        Number of Monte Carlo paths generated to derive prediction intervals.
    """

    def __init__(
        self,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 12,
        damped_trend: bool = False,
        initialization_method: str = "estimated",
        random_state: int | None = 42,
        simulation_repetitions: int = 2000,
    ) -> None:
        self.trend: str | None = trend
        self.seasonal: str | None = seasonal
        self.seasonal_periods: int = seasonal_periods
        self.damped_trend: bool = damped_trend
        self.initialization_method: str = initialization_method
        self.random_state: int | None = random_state
        self.simulation_repetitions: int = simulation_repetitions

        self._history: np.ndarray | None = None
        self._fitted_model: HoltWintersResults | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether forecaster has been fitted."""
        return self._fitted_model is not None

    @property
    def fitted_model(self) -> HoltWintersResults | None:
        """Access underlying fitted statsmodels HoltWintersResults instance."""
        return self._fitted_model

    @property
    def model_summary(self) -> str:
        """Summary text from fitted statsmodels object."""
        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before accessing model_summary.")
        return str(self._fitted_model.summary())

    @property
    def params(self) -> dict[str, float]:
        """Estimated smoothing parameters (alpha, beta, gamma)."""
        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before accessing params.")
        p = self._fitted_model.params
        return {
            "smoothing_level": float(p.get("smoothing_level", np.nan)),
            "smoothing_trend": float(p.get("smoothing_trend", np.nan)),
            "smoothing_seasonal": float(p.get("smoothing_seasonal", np.nan)),
        }

    @property
    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values."""
        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before accessing fitted_values.")
        return np.asarray(self._fitted_model.fittedvalues, dtype=float)

    @property
    def residuals(self) -> np.ndarray:
        """In-sample residuals (actual - fitted)."""
        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before accessing residuals.")
        return np.asarray(self._fitted_model.resid, dtype=float)

    def fit(self, series: Sequence[float] | np.ndarray | pd.Series) -> HoltWintersForecaster:
        """
        Fit Holt-Winters Exponential Smoothing model.

        Requires at least 2 full seasonal cycles (len(series) >= 2 * seasonal_periods).

        Parameters
        ----------
        series : Sequence[float] | np.ndarray | pd.Series
            Historical training series.

        Returns
        -------
        HoltWintersForecaster
            Fitted forecaster instance.

        Raises
        ------
        ValueError
            If series contains fewer than 2 * seasonal_periods observations,
            or contains NaNs/Infs.
        """
        min_len = 2 * self.seasonal_periods
        arr = np.asarray(series).ravel()
        if len(arr) == 0:
            raise ValueError("Time series cannot be empty.")
        if len(arr) < min_len:
            raise ValueError(
                f"Insufficient history: Series length ({len(arr)}) is less than "
                f"required minimum ({min_len}) for m={self.seasonal_periods} seasonal Holt-Winters."
            )
        y = _validate_input_series(series, min_length=min_len)
        self._history = y

        model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            initialization_method=self.initialization_method,
        )
        self._fitted_model = model.fit()
        return self

    def forecast(self, steps: int = 12) -> np.ndarray:
        """
        Generate point forecasts across the forecast horizon.

        Parameters
        ----------
        steps : int, default 12
            Number of future time steps to forecast.

        Returns
        -------
        np.ndarray
            Array of point forecasts of shape (steps,).
        """
        if self._fitted_model is None or self._history is None:
            raise RuntimeError("Model must be fitted before forecast() is called.")
        if steps <= 0:
            raise ValueError(f"Steps must be a positive integer, got {steps}.")

        return np.asarray(self._fitted_model.forecast(steps), dtype=float)

    def predict(
        self,
        horizon: int = 12,
        confidence_level: float = 0.95,
        steps: int | None = None,
    ) -> ForecastResult:
        """
        Generate point forecasts and 95% prediction intervals via state-space simulation.

        Parameters
        ----------
        horizon : int, default 12
            Forecast horizon length.
        confidence_level : float, default 0.95
            Coverage probability for prediction intervals (0 < confidence_level < 1).
        steps : int | None, default None
            Alias for horizon to maintain API compatibility with BaseForecaster.

        Returns
        -------
        ForecastResult
            Container with point forecasts, lower bound (>= 0), and upper bound.
        """
        if steps is not None:
            horizon = steps

        if self._fitted_model is None or self._history is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer, got {horizon}.")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"Confidence level must be between 0 and 1 exclusive, got {confidence_level}."
            )

        # 1. Point forecast
        pt = self.forecast(steps=horizon)

        # 2. State-space simulation for prediction intervals
        rng = np.random.default_rng(self.random_state)
        sim = self._fitted_model.simulate(
            nsimulations=horizon,
            repetitions=self.simulation_repetitions,
            error="add",
            rng=rng,
        )

        alpha = 1.0 - confidence_level
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0

        lower = np.percentile(sim, lower_pct, axis=1)
        upper = np.percentile(sim, upper_pct, axis=1)

        # Enforce physical non-negativity and interval sandwiching
        lower = np.maximum(0.0, lower)
        lower = np.minimum(lower, pt)
        upper = np.maximum(upper, pt)

        return ForecastResult(
            point_forecast=pt,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            model_name="HoltWintersAdditive",
            residuals=self.residuals,
        )

    def get_prediction(self, steps: int = 12, alpha: float = 0.05) -> ForecastResult:
        """
        Compatibility method matching statsmodels get_prediction interface.

        Parameters
        ----------
        steps : int, default 12
            Forecast horizon.
        alpha : float, default 0.05
            Significance level (default 0.05 for 95% confidence).
        """
        return self.predict(horizon=steps, confidence_level=1.0 - alpha)


def evaluate_primary_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, ForecastResult], dict[str, Any]]:
    """
    Fit Holt-Winters on training set and evaluate against test set and Phase 10 baselines.

    Parameters
    ----------
    train_df : pd.DataFrame
        24-month training set.
    test_df : pd.DataFrame
        12-month test set.
    targets : list[str] | None, default None
        List of target column names. Defaults to total, electricity, travel.

    Returns
    -------
    summary_df : pd.DataFrame
        Evaluation comparison table.
    predictions : dict[str, ForecastResult]
        Prediction objects per target.
    error_analysis : dict[str, Any]
        Diagnostic statistics (mean residuals, largest errors, seasonal breakdown).
    """
    from src.forecasting.baselines import NaiveBaseline, SeasonalNaiveBaseline

    if targets is None:
        targets = ["total_emissions_kg", "electricity_emissions_kg", "travel_emissions_kg"]

    target_labels = {
        "total_emissions_kg": "Total Campus Emissions (kgCO2e)",
        "electricity_emissions_kg": "Electricity Emissions (kgCO2e)",
        "travel_emissions_kg": "Travel Emissions (kgCO2e)",
    }

    horizon = len(test_df)
    summary_rows: list[dict[str, Any]] = []
    predictions: dict[str, ForecastResult] = {}
    error_analysis: dict[str, Any] = {}

    for target in targets:
        y_train = train_df[target].values
        y_test = test_df[target].values

        # 1. Phase 10 Baselines (for direct fair comparison)
        naive_model = NaiveBaseline().fit(y_train)
        naive_pred = naive_model.predict(horizon)
        mae_naive = calculate_mae(y_test, naive_pred)
        rmse_naive = calculate_rmse(y_test, naive_pred)
        mape_naive = calculate_mape(y_test, naive_pred)

        s_naive_model = SeasonalNaiveBaseline(seasonal_period=12).fit(y_train)
        s_naive_pred = s_naive_model.predict(horizon)
        mae_sn = calculate_mae(y_test, s_naive_pred)
        rmse_sn = calculate_rmse(y_test, s_naive_pred)
        mape_sn = calculate_mape(y_test, s_naive_pred)

        # 2. Phase 11 Primary Model: Holt-Winters Additive
        hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12).fit(y_train)
        hw_pred = hw.predict(horizon=horizon, confidence_level=0.95)
        predictions[target] = hw_pred

        mae_hw = calculate_mae(y_test, hw_pred.point_forecast)
        rmse_hw = calculate_rmse(y_test, hw_pred.point_forecast)
        mape_hw = calculate_mape(y_test, hw_pred.point_forecast)

        # Relative improvement over Seasonal Naive baseline
        mae_imp_sn = calculate_improvement(mae_sn, mae_hw)
        rmse_imp_sn = calculate_improvement(rmse_sn, rmse_hw)

        # Error & bias analysis
        residuals_sn = y_test - s_naive_pred
        residuals_hw = y_test - hw_pred.point_forecast
        abs_errors = np.abs(residuals_hw)
        max_idx = int(np.argmax(abs_errors))
        max_month = str(test_df["date"].iloc[max_idx])

        # Coverage probability
        covered = (y_test >= hw_pred.lower_bound) & (y_test <= hw_pred.upper_bound)
        cov_pct = round(float(np.mean(covered) * 100.0), 2)

        error_analysis[target] = {
            "params": hw.params,
            "sn_mean_residual_kg": round(float(np.mean(residuals_sn)), 2),
            "hw_mean_residual_kg": round(float(np.mean(residuals_hw)), 2),
            "bias_reduction_pct": round(
                float((abs(np.mean(residuals_sn)) - abs(np.mean(residuals_hw))) / abs(np.mean(residuals_sn)) * 100.0),
                2,
            ),
            "coverage_pct": cov_pct,
            "largest_error_month": max_month,
            "largest_abs_error_kg": round(float(abs_errors[max_idx]), 2),
            "actual_at_largest_error": round(float(y_test[max_idx]), 2),
            "forecast_at_largest_error": round(float(hw_pred.point_forecast[max_idx]), 2),
        }

        # Tabular summary rows
        summary_rows.append({
            "target": target,
            "target_label": target_labels.get(target, target),
            "model": "Standard Naive (Flat)",
            "mae_kg": mae_naive,
            "rmse_kg": rmse_naive,
            "mape_pct": mape_naive,
            "coverage_95_pct": np.nan,
            "mae_improvement_vs_seasonal_naive_pct": np.nan,
        })
        summary_rows.append({
            "target": target,
            "target_label": target_labels.get(target, target),
            "model": "Seasonal Naive (m=12)",
            "mae_kg": mae_sn,
            "rmse_kg": rmse_sn,
            "mape_pct": mape_sn,
            "coverage_95_pct": np.nan,
            "mae_improvement_vs_seasonal_naive_pct": 0.0,
        })
        summary_rows.append({
            "target": target,
            "target_label": target_labels.get(target, target),
            "model": "Holt-Winters (Additive Trend & Season)",
            "mae_kg": mae_hw,
            "rmse_kg": rmse_hw,
            "mape_pct": mape_hw,
            "coverage_95_pct": cov_pct,
            "mae_improvement_vs_seasonal_naive_pct": mae_imp_sn,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, predictions, error_analysis


def generate_primary_comparison_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: dict[str, ForecastResult],
    output_path: str | Path = "reports/figures/11_primary_vs_baseline_comparison.png",
) -> Path:
    """
    Generate 3-panel publication comparison of Primary Holt-Winters model vs Actuals and 95% PI.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training observations.
    test_df : pd.DataFrame
        Testing observations.
    predictions : dict[str, ForecastResult]
        Holt-Winters predictions per target.
    output_path : str | Path
        Path to output PNG image.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    targets = list(predictions.keys())
    target_labels = {
        "total_emissions_kg": "Total Campus Emissions (MTCO2e)",
        "electricity_emissions_kg": "Electricity Emissions (MTCO2e)",
        "travel_emissions_kg": "Travel Emissions (MTCO2e)",
    }

    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 11), sharex=False)
    fig.suptitle(
        "Phase 11: Primary Forecasting Model Evaluation (Holt-Winters Additive + 95% PI)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    train_dates = pd.to_datetime(train_df["date"])
    test_dates = pd.to_datetime(test_df["date"])

    for ax, target in zip(axes, targets):
        y_train = train_df[target].values / 1000.0  # Scale to MT
        y_test = test_df[target].values / 1000.0

        hw_pred = predictions[target]
        hw_pt = hw_pred.point_forecast / 1000.0
        hw_lb = hw_pred.lower_bound / 1000.0
        hw_ub = hw_pred.upper_bound / 1000.0

        # Training history
        ax.plot(
            train_dates,
            y_train,
            color="#2b5c8f",
            linewidth=2,
            marker="o",
            markersize=3.5,
            label="Training Data (2023-2024)",
        )
        # Actual Test
        ax.plot(
            test_dates,
            y_test,
            color="#1b9e77",
            linewidth=2.2,
            marker="s",
            markersize=4.5,
            label="Actual Test (2025)",
        )
        # Holt-Winters forecast
        ax.plot(
            test_dates,
            hw_pt,
            color="#d95f02",
            linewidth=2.2,
            linestyle="--",
            marker="^",
            markersize=4.5,
            label="Holt-Winters Primary Forecast",
        )
        # 95% Prediction Interval
        ax.fill_between(
            test_dates,
            hw_lb,
            hw_ub,
            color="#d95f02",
            alpha=0.18,
            label="95% Prediction Interval",
        )

        # Split marker
        ax.axvline(x=train_dates.iloc[-1], color="#757575", linestyle="--", linewidth=1.2, alpha=0.7)

        title_str = target_labels.get(target, target)
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("MTCO2e", fontweight="bold")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_p, dpi=300)
    plt.close(fig)
    logger.info("Saved primary model comparison plot to %s", out_p)
    return out_p


def run_primary_experiment() -> tuple[pd.DataFrame, dict[str, ForecastResult], dict[str, Any]]:
    """
    Execute complete Phase 11 primary model evaluation pipeline.

    1. Loads and aggregates monthly emissions from data/processed/campus_emissions.csv.
    2. Executes chronological 24m train / 12m test split.
    3. Fits Holt-Winters (additive trend, additive seasonality m=12).
    4. Evaluates performance against Phase 10 Standard Naive and Seasonal Naive baselines.
    5. Exports summary table, 2025 predictions with 95% PI, and comparison plot.
    """
    from src.forecasting.experiment import (
        load_and_aggregate_monthly_emissions,
        split_chronological,
    )

    print("=" * 75)
    print("PHASE 11: PRIMARY FORECASTING MODEL EVALUATION (HOLT-WINTERS)")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("=" * 75)

    # 1. Load data
    monthly_df = load_and_aggregate_monthly_emissions("data/processed/campus_emissions.csv")
    train_df, test_df = split_chronological(monthly_df, train_months=24, test_months=12)

    print(f"Historical Data : {len(monthly_df)} months ({monthly_df['date'].iloc[0]} to {monthly_df['date'].iloc[-1]})")
    print(f"Training Period : {len(train_df)} months ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
    print(f"Testing Period  : {len(test_df)} months ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")
    print(f"Forecast Horizon: {len(test_df)} months (Jan 2025 - Dec 2025)")
    print("-" * 75)

    # 2. Evaluate
    summary_df, predictions, error_analysis = evaluate_primary_model(train_df, test_df)

    display_cols = [
        "target",
        "model",
        "mae_kg",
        "rmse_kg",
        "mape_pct",
        "coverage_95_pct",
        "mae_improvement_vs_seasonal_naive_pct",
    ]
    print("\n--- MODEL PERFORMANCE COMPARISON TABLE ---")
    print(summary_df[display_cols].to_string(index=False))

    # 3. Export Summary CSV
    csv_summary_path = Path("data/processed/primary_model_evaluation_summary.csv")
    csv_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_summary_path, index=False)
    print(f"\n[Artifact] Exported summary CSV : {csv_summary_path.resolve()}")

    # 4. Build and Export Detailed 2025 Predictions CSV
    pred_records = []
    test_dates = test_df["date"].values
    for i, date_str in enumerate(test_dates):
        row = {"date": date_str}
        for target in ["total_emissions_kg", "electricity_emissions_kg", "travel_emissions_kg"]:
            short_tgt = target.replace("_emissions_kg", "")
            actual_val = float(test_df[target].iloc[i])
            pred_obj = predictions[target]
            pt = float(pred_obj.point_forecast[i])
            lb = float(pred_obj.lower_bound[i])
            ub = float(pred_obj.upper_bound[i])
            res = actual_val - pt
            row[f"{short_tgt}_actual_kg"] = round(actual_val, 2)
            row[f"{short_tgt}_pred_kg"] = round(pt, 2)
            row[f"{short_tgt}_lower_95_kg"] = round(lb, 2)
            row[f"{short_tgt}_upper_95_kg"] = round(ub, 2)
            row[f"{short_tgt}_residual_kg"] = round(res, 2)
        pred_records.append(row)

    pred_df = pd.DataFrame(pred_records)
    csv_pred_path = Path("data/processed/primary_model_predictions_2025.csv")
    pred_df.to_csv(csv_pred_path, index=False)
    print(f"[Artifact] Exported predictions CSV : {csv_pred_path.resolve()}")

    # 5. Generate and Export Comparison Visualization
    fig_path = Path("reports/figures/11_primary_vs_baseline_comparison.png")
    generate_primary_comparison_plot(train_df, test_df, predictions, output_path=fig_path)
    print(f"[Artifact] Exported comparison plot: {fig_path.resolve()}")

    # 6. Print Diagnostic Analysis
    print("\n--- ERROR AND BIAS DIAGNOSTICS ---")
    for tgt, diag in error_analysis.items():
        print(f"\nTarget: {tgt}")
        print(f"  Estimated Parameters : {diag['params']}")
        print(f"  Seasonal Naive Bias  : {diag['sn_mean_residual_kg']:+,.2f} kgCO2e")
        print(f"  Holt-Winters Bias    : {diag['hw_mean_residual_kg']:+,.2f} kgCO2e")
        print(f"  Bias Reduction       : {diag['bias_reduction_pct']:.2f}%")
        print(f"  95% PI Coverage      : {diag['coverage_pct']:.1f}%")
        print(
            f"  Max Error Month      : {diag['largest_error_month']} "
            f"(Error: {diag['largest_abs_error_kg']:,.2f} kgCO2e, "
            f"Actual: {diag['actual_at_largest_error']:,.2f}, "
            f"Forecast: {diag['forecast_at_largest_error']:,.2f})"
        )

    print("=" * 75)
    return summary_df, predictions, error_analysis


if __name__ == "__main__":
    run_primary_experiment()

