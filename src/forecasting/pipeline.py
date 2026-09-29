"""
Forecasting Pipeline & Experiment Execution
===========================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Executes time-series forecasting experiment per docs/FORECASTING_METHOD.md:
1. Strict chronological train/test split (24 months train, 12 months test).
2. Seasonal Naive baseline evaluation.
3. Holt-Winters Exponential Smoothing evaluation with 95% prediction intervals.
4. Error analysis: MAE, RMSE, MAPE, coverage probability, and relative improvement.
5. Future 12-month projections trained on full 36-month dataset.
6. Publication-quality figure generation (saved to reports/figures/).
7. Clean CSV exports for dashboard and reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecasting.metrics import (
    ForecastMetrics,
    calculate_improvement,
    evaluate_forecast,
)
from src.forecasting.models import (
    ForecastResult,
    HoltWintersForecaster,
    SeasonalNaiveForecaster,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TARGETS: list[str] = [
    "total_emissions_kg",
    "electricity_emissions_kg",
    "travel_emissions_kg",
]

TARGET_LABELS: dict[str, str] = {
    "total_emissions_kg": "Total Campus Emissions (kgCO2e)",
    "electricity_emissions_kg": "Electricity Emissions (kgCO2e)",
    "travel_emissions_kg": "Travel Emissions (kgCO2e)",
}


def prepare_monthly_emissions(
    data_path: str | Path = "data/processed/campus_emissions.csv",
) -> pd.DataFrame:
    """
    Load validated campus emissions data and aggregate to campus-wide monthly totals.

    Parameters
    ----------
    data_path : str | Path
        Path to processed campus emissions CSV file.

    Returns
    -------
    pd.DataFrame
        Monthly aggregated emissions with columns:
        ['date', 'electricity_emissions_kg', 'travel_emissions_kg',
         'waste_emissions_kg', 'procurement_emissions_kg', 'total_emissions_kg']
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Emissions dataset not found at {path.resolve()}")

    df = pd.read_csv(path)
    required_cols = [
        "date",
        "electricity_emissions_kg",
        "travel_emissions_kg",
        "waste_emissions_kg",
        "procurement_emissions_kg",
        "total_emissions_kg",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # Aggregate by date summing across all buildings
    emissions_cols = [c for c in required_cols if c != "date"]
    monthly_df = df.groupby("date", as_index=False)[emissions_cols].sum()

    # Ensure chronological order
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)
    monthly_df["date"] = monthly_df["date"].dt.strftime("%Y-%m-%d")

    # Verify no missing months
    dates = pd.to_datetime(monthly_df["date"])
    expected_range = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq="MS")
    if len(dates) != len(expected_range):
        raise ValueError(
            f"Monthly time series has gaps: expected {len(expected_range)} months, "
            f"found {len(dates)} records."
        )

    logger.info(
        "Loaded and aggregated monthly emissions: %d months from %s to %s.",
        len(monthly_df),
        monthly_df["date"].iloc[0],
        monthly_df["date"].iloc[-1],
    )
    return monthly_df


def split_chronological_data(
    monthly_df: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological time-series train/test split.

    Explicitly avoids random splitting to prevent temporal leakage and respect
    autocorrelation in monthly academic cycles.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly aggregated emissions dataframe.
    train_months : int, default 24
        Number of historical months for training.
    test_months : int, default 12
        Number of historical months for testing / evaluation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    total_required = train_months + test_months
    if len(monthly_df) < total_required:
        raise ValueError(
            f"Dataset length ({len(monthly_df)}) is insufficient for "
            f"train ({train_months}) + test ({test_months}) = {total_required} months."
        )

    train_df = monthly_df.iloc[:train_months].copy().reset_index(drop=True)
    test_df = monthly_df.iloc[train_months : total_required].copy().reset_index(drop=True)

    logger.info(
        "Strict chronological split: Train=%d months (%s to %s), Test=%d months (%s to %s).",
        len(train_df),
        train_df["date"].iloc[0],
        train_df["date"].iloc[-1],
        len(test_df),
        test_df["date"].iloc[0],
        test_df["date"].iloc[-1],
    )
    return train_df, test_df


def run_forecast_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, ForecastResult]], dict[str, dict[str, ForecastMetrics]]]:
    """
    Fit baseline and primary forecasting models, evaluate accuracy and coverage.

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
        Tabular comparison of MAE, RMSE, MAPE, Coverage %, and relative improvements.
    forecast_results : dict[str, dict[str, ForecastResult]]
        Nested dictionary target -> model_name -> ForecastResult.
    metrics_results : dict[str, dict[str, ForecastMetrics]]
        Nested dictionary target -> model_name -> ForecastMetrics.
    """
    if targets is None:
        targets = list(DEFAULT_TARGETS)

    forecast_results: dict[str, dict[str, ForecastResult]] = {}
    metrics_results: dict[str, dict[str, ForecastMetrics]] = {}
    summary_rows: list[dict[str, Any]] = []

    steps = len(test_df)

    for target in targets:
        if target not in train_df.columns or target not in test_df.columns:
            raise KeyError(f"Target column '{target}' not present in both train and test sets.")

        y_train = train_df[target].values
        y_test = test_df[target].values

        # 1. Seasonal Naive baseline
        naive_model = SeasonalNaiveForecaster(seasonal_period=12).fit(y_train)
        naive_fc = naive_model.predict(steps=steps, confidence_level=0.95)
        naive_metrics = evaluate_forecast(
            actual=y_test,
            predicted=naive_fc.point_forecast,
            lower_bound=naive_fc.lower_bound,
            upper_bound=naive_fc.upper_bound,
        )

        # 2. Holt-Winters Additive model
        hw_model = HoltWintersForecaster(
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            random_state=42,
            simulation_repetitions=2000,
        ).fit(y_train)
        hw_fc = hw_model.predict(steps=steps, confidence_level=0.95)
        hw_metrics = evaluate_forecast(
            actual=y_test,
            predicted=hw_fc.point_forecast,
            lower_bound=hw_fc.lower_bound,
            upper_bound=hw_fc.upper_bound,
        )

        # 3. Relative improvement of Holt-Winters over Seasonal Naive
        mae_imp = calculate_improvement(naive_metrics.mae, hw_metrics.mae)
        rmse_imp = calculate_improvement(naive_metrics.rmse, hw_metrics.rmse)

        forecast_results[target] = {
            "Seasonal Naive": naive_fc,
            "Holt-Winters": hw_fc,
        }
        metrics_results[target] = {
            "Seasonal Naive": naive_metrics,
            "Holt-Winters": hw_metrics,
        }

        summary_rows.append({
            "target": target,
            "target_label": TARGET_LABELS.get(target, target),
            "model": "Seasonal Naive",
            "mae_kg": naive_metrics.mae,
            "rmse_kg": naive_metrics.rmse,
            "mape_pct": naive_metrics.mape,
            "coverage_pct": naive_metrics.coverage_pct,
            "mae_improvement_pct": 0.0,
            "rmse_improvement_pct": 0.0,
        })
        summary_rows.append({
            "target": target,
            "target_label": TARGET_LABELS.get(target, target),
            "model": "Holt-Winters (Additive)",
            "mae_kg": hw_metrics.mae,
            "rmse_kg": hw_metrics.rmse,
            "mape_pct": hw_metrics.mape,
            "coverage_pct": hw_metrics.coverage_pct,
            "mae_improvement_pct": mae_imp,
            "rmse_improvement_pct": rmse_imp,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, forecast_results, metrics_results


def generate_future_projections(
    monthly_df: pd.DataFrame,
    steps: int = 12,
    targets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, ForecastResult]]:
    """
    Retrain Holt-Winters model on all 36 months of historical data and forecast next 12 months.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Complete 36-month monthly emissions data.
    steps : int, default 12
        Forecast horizon steps (12 months = 2026).
    targets : list[str] | None, default None
        List of targets to forecast.

    Returns
    -------
    projections_df : pd.DataFrame
        Forecast table with future dates, point forecasts, and 95% PI bounds.
    projections_dict : dict[str, ForecastResult]
        Dictionary mapping target -> ForecastResult.
    """
    if targets is None:
        targets = list(DEFAULT_TARGETS)

    last_date = pd.to_datetime(monthly_df["date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq="MS",
    ).strftime("%Y-%m-%d")

    projections_dict: dict[str, ForecastResult] = {}
    rows: list[dict[str, Any]] = []

    for target in targets:
        series = monthly_df[target].values
        hw_model = HoltWintersForecaster(
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            random_state=42,
            simulation_repetitions=2000,
        ).fit(series)
        fc = hw_model.predict(steps=steps, confidence_level=0.95)
        projections_dict[target] = fc

        for d, pt, lb, ub in zip(future_dates, fc.point_forecast, fc.lower_bound, fc.upper_bound):
            rows.append({
                "date": d,
                "target": target,
                "target_label": TARGET_LABELS.get(target, target),
                "point_forecast": round(float(pt), 4),
                "lower_bound_95": round(float(lb), 4),
                "upper_bound_95": round(float(ub), 4),
            })

    projections_df = pd.DataFrame(rows)
    return projections_df, projections_dict


def export_forecast_results(
    monthly_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    test_results: dict[str, dict[str, ForecastResult]],
    projections_df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> tuple[Path, Path]:
    """
    Export forecast evaluation summary and complete time-series forecasts to CSV.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Complete historical monthly emissions.
    summary_df : pd.DataFrame
        Test evaluation metrics summary.
    test_results : dict[str, dict[str, ForecastResult]]
        Results on the 12-month test set.
    projections_df : pd.DataFrame
        Future 12-month projections.
    output_dir : str | Path, default 'data/processed'
        Output directory path.

    Returns
    -------
    tuple[Path, Path]
        Paths to (summary_csv, full_forecasts_csv).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "forecast_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Build comprehensive master forecast table covering 2023-01 to 2026-12 (48 months)
    records: list[dict[str, Any]] = []
    train_dates = monthly_df["date"].iloc[:24].tolist()
    test_dates = monthly_df["date"].iloc[24:36].tolist()
    future_dates = projections_df["date"].unique().tolist()

    for target in test_results.keys():
        naive_fc = test_results[target]["Seasonal Naive"]
        hw_fc = test_results[target]["Holt-Winters"]

        # Train records (24 months)
        for d in train_dates:
            val = float(monthly_df.loc[monthly_df["date"] == d, target].iloc[0])
            records.append({
                "date": d,
                "target": target,
                "split_type": "train",
                "actual": round(val, 4),
                "naive_forecast": np.nan,
                "hw_forecast": np.nan,
                "hw_lower_95": np.nan,
                "hw_upper_95": np.nan,
                "residual": np.nan,
            })

        # Test records (12 months)
        for i, d in enumerate(test_dates):
            val = float(monthly_df.loc[monthly_df["date"] == d, target].iloc[0])
            n_pt = float(naive_fc.point_forecast[i])
            hw_pt = float(hw_fc.point_forecast[i])
            hw_lb = float(hw_fc.lower_bound[i])
            hw_ub = float(hw_fc.upper_bound[i])
            res = val - hw_pt
            records.append({
                "date": d,
                "target": target,
                "split_type": "test",
                "actual": round(val, 4),
                "naive_forecast": round(n_pt, 4),
                "hw_forecast": round(hw_pt, 4),
                "hw_lower_95": round(hw_lb, 4),
                "hw_upper_95": round(hw_ub, 4),
                "residual": round(res, 4),
            })

        # Future projection records (12 months)
        target_proj = projections_df[projections_df["target"] == target]
        for _, row in target_proj.iterrows():
            records.append({
                "date": row["date"],
                "target": target,
                "split_type": "future_forecast",
                "actual": np.nan,
                "naive_forecast": np.nan,
                "hw_forecast": row["point_forecast"],
                "hw_lower_95": row["lower_bound_95"],
                "hw_upper_95": row["upper_bound_95"],
                "residual": np.nan,
            })

    forecasts_df = pd.DataFrame(records)
    forecasts_path = out_dir / "emissions_forecasts.csv"
    forecasts_df.to_csv(forecasts_path, index=False)

    logger.info("Exported evaluation summary to %s", summary_path)
    logger.info("Exported complete forecast series to %s", forecasts_path)
    return summary_path, forecasts_path


def generate_forecast_visualizations(
    monthly_df: pd.DataFrame,
    test_results: dict[str, dict[str, ForecastResult]],
    projections_dict: dict[str, ForecastResult],
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    """
    Generate publication-quality figures matching the styling of reports/figures/:
    - 07_forecast_baseline_comparison.png
    - 08_forecast_residuals_analysis.png
    - 09_future_emissions_projections.png

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Complete historical monthly emissions.
    test_results : dict[str, dict[str, ForecastResult]]
        Evaluation test results per target.
    projections_dict : dict[str, ForecastResult]
        Future 12-month projections per target.
    output_dir : str | Path, default 'reports/figures'
        Directory to save generated PNG images.

    Returns
    -------
    list[Path]
        Paths of generated figure files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure plot parameters consistent with existing charts
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

    generated_paths: list[Path] = []
    targets = list(test_results.keys())

    train_dates = pd.to_datetime(monthly_df["date"].iloc[:24])
    test_dates = pd.to_datetime(monthly_df["date"].iloc[24:36])
    last_date = pd.to_datetime(monthly_df["date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq="MS",
    )

    # -------------------------------------------------------------
    # Figure 07: Forecast Baseline Comparison (Train / Test / Naive / HW + 95% PI)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(
        "Campus Emissions Forecasting: Holt-Winters vs. Seasonal Naive Baseline (2025 Test Set)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for ax, target in zip(axes, targets):
        y_train = monthly_df.iloc[:24][target].values
        y_test = monthly_df.iloc[24:36][target].values

        naive_pt = test_results[target]["Seasonal Naive"].point_forecast
        hw_res = test_results[target]["Holt-Winters"]
        hw_pt = hw_res.point_forecast
        hw_lb = hw_res.lower_bound
        hw_ub = hw_res.upper_bound

        # Convert to MT for clean visualization if target is total, else keep kg or label
        unit_scale = 1000.0  # Convert kg to MTCO2e for readability
        unit_label = "MTCO2e"

        # Historical Training
        ax.plot(
            train_dates,
            y_train / unit_scale,
            color="#2b5c8f",
            linewidth=2,
            marker="o",
            markersize=3.5,
            label="Training Data (2023-2024)",
        )
        # Actual Test
        ax.plot(
            test_dates,
            y_test / unit_scale,
            color="#1b9e77",
            linewidth=2.2,
            marker="s",
            markersize=4.5,
            label="Actual Test (2025)",
        )
        # Seasonal Naive
        ax.plot(
            test_dates,
            naive_pt / unit_scale,
            color="#e7298a",
            linewidth=1.8,
            linestyle=":",
            marker="x",
            markersize=5,
            label="Seasonal Naive Baseline",
        )
        # Holt-Winters
        ax.plot(
            test_dates,
            hw_pt / unit_scale,
            color="#d95f02",
            linewidth=2.2,
            linestyle="--",
            marker="^",
            markersize=4.5,
            label="Holt-Winters Forecast",
        )
        # Shaded 95% Prediction Interval
        ax.fill_between(
            test_dates,
            hw_lb / unit_scale,
            hw_ub / unit_scale,
            color="#d95f02",
            alpha=0.18,
            label="Holt-Winters 95% Prediction Interval",
        )

        # Vertical divider for chronological split
        ax.axvline(x=train_dates.iloc[-1], color="#757575", linestyle="--", linewidth=1.2, alpha=0.7)

        title_str = TARGET_LABELS.get(target, target).replace("(kgCO2e)", f"({unit_label})")
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(unit_label, fontweight="semibold")

        # Format x ticks using DateFormatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig07_path = out_dir / "07_forecast_baseline_comparison.png"
    plt.savefig(fig07_path, dpi=300)
    plt.close(fig)
    generated_paths.append(fig07_path)
    logger.info("Saved figure: %s", fig07_path)

    # -------------------------------------------------------------
    # Figure 08: Forecast Residuals Analysis (Time-series and Distribution)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(
        "Forecast Error & Residual Analysis (2025 Test Period: Holt-Winters Additive)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for i, target in enumerate(targets):
        y_test = monthly_df.iloc[24:36][target].values
        hw_pt = test_results[target]["Holt-Winters"].point_forecast
        residuals = y_test - hw_pt
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals, ddof=1))

        # Time series of residuals (Left column)
        ax_ts = axes[i, 0]
        ax_ts.plot(test_dates, residuals, marker="o", color="#d95f02", linewidth=1.8)
        ax_ts.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
        ax_ts.axhline(mean_res, color="blue", linestyle="--", linewidth=1.2, label=f"Mean Error: {mean_res:+,.0f} kg")
        ax_ts.fill_between(
            test_dates,
            mean_res - 1.96 * std_res,
            mean_res + 1.96 * std_res,
            color="blue",
            alpha=0.08,
            label=f"±1.96 SD Band ({std_res:,.0f} kg)",
        )
        ax_ts.set_title(f"{TARGET_LABELS.get(target, target)} — Residual Sequence", fontsize=10.5)
        ax_ts.set_ylabel("Error (kgCO2e)", fontweight="semibold")
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_ts.get_xticklabels(), rotation=30, ha="right")
        ax_ts.legend(loc="upper right", frameon=True)

        # Histogram & Distribution of residuals (Right column)
        ax_hist = axes[i, 1]
        n_bins = 6
        ax_hist.hist(residuals, bins=n_bins, color="#2b5c8f", alpha=0.7, edgecolor="white", density=True)
        # Normal fit curve for reference
        norm_x = np.linspace(min(residuals) - std_res, max(residuals) + std_res, 100)
        norm_y = (1.0 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((norm_x - mean_res) / std_res) ** 2)
        ax_hist.plot(norm_x, norm_y, color="crimson", linewidth=1.8, label="Fitted Normal")
        ax_hist.axvline(0, color="black", linestyle="-", linewidth=1.0)
        ax_hist.set_title(f"{TARGET_LABELS.get(target, target)} — Residual Distribution", fontsize=10.5)
        ax_hist.set_xlabel("Residual (kgCO2e)", fontweight="semibold")
        ax_hist.set_ylabel("Density", fontweight="semibold")
        ax_hist.legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig08_path = out_dir / "08_forecast_residuals_analysis.png"
    plt.savefig(fig08_path, dpi=300)
    plt.close(fig)
    generated_paths.append(fig08_path)
    logger.info("Saved figure: %s", fig08_path)

    # -------------------------------------------------------------
    # Figure 09: Future 2026 Emissions Projections with 95% Confidence Band
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(
        "Campus Carbon Emissions: 3-Year Historical Trend & Future Projections (2026)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    all_hist_dates = pd.to_datetime(monthly_df["date"])
    for ax, target in zip(axes, targets):
        hist_y = monthly_df[target].values
        fc = projections_dict[target]
        fut_pt = fc.point_forecast
        fut_lb = fc.lower_bound
        fut_ub = fc.upper_bound

        unit_scale = 1000.0
        unit_label = "MTCO2e"

        # Historical (2023-2025)
        ax.plot(
            all_hist_dates,
            hist_y / unit_scale,
            color="#2b5c8f",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Observed History (2023-2025)",
        )

        # Future Forecast (2026)
        # Connect last historical point to first projection for visual continuity
        plot_dates = pd.concat(
            [pd.Series([all_hist_dates.iloc[-1]]), pd.Series(future_dates)],
            ignore_index=True,
        )
        plot_pt = np.insert(fut_pt / unit_scale, 0, hist_y[-1] / unit_scale)
        plot_lb = np.insert(fut_lb / unit_scale, 0, hist_y[-1] / unit_scale)
        plot_ub = np.insert(fut_ub / unit_scale, 0, hist_y[-1] / unit_scale)

        ax.plot(
            plot_dates,
            plot_pt,
            color="#d95f02",
            linewidth=2.2,
            linestyle="--",
            marker="^",
            markersize=4.5,
            label="2026 Holt-Winters Projection",
        )
        ax.fill_between(
            plot_dates,
            plot_lb,
            plot_ub,
            color="#d95f02",
            alpha=0.2,
            label="95% Prediction Interval",
        )

        # Vertical separator
        ax.axvline(x=all_hist_dates.iloc[-1], color="#757575", linestyle="--", linewidth=1.2, alpha=0.7)

        title_str = TARGET_LABELS.get(target, target).replace("(kgCO2e)", f"({unit_label})")
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(unit_label, fontweight="semibold")

        # Ticks: every 4 months across the 48 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig09_path = out_dir / "09_future_emissions_projections.png"
    plt.savefig(fig09_path, dpi=300)
    plt.close(fig)
    generated_paths.append(fig09_path)
    logger.info("Saved figure: %s", fig09_path)

    return generated_paths


def run_pipeline() -> None:
    """Execute complete forecasting workflow and log results."""
    print("=" * 70)
    print("CAMPUS CARBON EMISSIONS FORECASTING PIPELINE")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("Methodology: docs/FORECASTING_METHOD.md")
    print("=" * 70)

    # 1. Load and aggregate monthly campus emissions
    monthly_df = prepare_monthly_emissions()

    # 2. Strict chronological split
    train_df, test_df = split_chronological_data(monthly_df, train_months=24, test_months=12)

    # 3. Model evaluation on test set
    summary_df, test_results, _ = run_forecast_experiment(train_df, test_df)

    print("\n--- FORECAST EVALUATION SUMMARY (2025 Test Set) ---")
    display_cols = [
        "target",
        "model",
        "mae_kg",
        "rmse_kg",
        "mape_pct",
        "coverage_pct",
        "mae_improvement_pct",
    ]
    print(summary_df[display_cols].to_string(index=False))

    # 4. Generate future projections for 2026
    projections_df, projections_dict = generate_future_projections(monthly_df, steps=12)

    # 5. Export results
    summary_path, forecasts_path = export_forecast_results(
        monthly_df=monthly_df,
        summary_df=summary_df,
        test_results=test_results,
        projections_df=projections_df,
    )

    # 6. Generate visualizations
    fig_paths = generate_forecast_visualizations(
        monthly_df=monthly_df,
        test_results=test_results,
        projections_dict=projections_dict,
    )

    print("\n--- ARTIFACTS GENERATED ---")
    print(f"Summary Table:  {summary_path.resolve()}")
    print(f"Forecast CSV:   {forecasts_path.resolve()}")
    for p in fig_paths:
        print(f"Figure:         {p.resolve()}")
    print("=" * 70)
    print("Forecasting pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
