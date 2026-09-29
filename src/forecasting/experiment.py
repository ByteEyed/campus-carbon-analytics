"""
Forecasting Baseline Experiment Pipeline
========================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 10: Forecasting Baseline Experiment
Implements:
1. Strict chronological train/test split (24 months train: 2023-01 to 2024-12,
   12 months test: 2025-01 to 2025-12).
2. Leakage-controlled evaluation of NaiveBaseline vs. SeasonalNaiveBaseline.
3. Multi-target evaluation for:
   - Total Campus Emissions (kgCO2e)
   - Electricity Emissions (kgCO2e)
   - Travel Emissions (kgCO2e)
4. Evaluation via MAE, RMSE, MAPE, and relative seasonal improvement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecasting.baselines import NaiveBaseline, SeasonalNaiveBaseline
from src.forecasting.metrics import (
    calculate_improvement,
    calculate_mae,
    calculate_mape,
    calculate_rmse,
)

logger = logging.getLogger("campus_carbon.forecasting.experiment")

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


def load_and_aggregate_monthly_emissions(
    data_path: str | Path = "data/processed/campus_emissions.csv",
) -> pd.DataFrame:
    """
    Load validated campus emissions data and aggregate to campus-wide monthly totals.

    Calculates campus totals across all facilities per date chronologically.
    Does not compute global stats over future periods to prevent data leakage.

    Parameters
    ----------
    data_path : str | Path
        Path to processed campus emissions CSV.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted monthly DataFrame with columns:
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

    # Aggregate by date summing across all facilities
    emissions_cols = [c for c in required_cols if c != "date"]
    monthly_df = df.groupby("date", as_index=False)[emissions_cols].sum()

    # Enforce strict chronological order
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)

    # Validate monotonicity and continuity
    if not monthly_df["date"].is_monotonic_increasing:
        raise ValueError("Dates in aggregated monthly emissions must be strictly increasing.")

    # Convert back to ISO string format YYYY-MM-DD
    monthly_df["date"] = monthly_df["date"].dt.strftime("%Y-%m-%d")
    return monthly_df


def split_chronological(
    monthly_df: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute strict chronological train/test split.

    Strictly separates historical training observations from test observations.
    Random shuffling is explicitly prohibited to prevent lookahead bias and temporal leakage.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Chronologically sorted monthly DataFrame.
    train_months : int, default 24
        Number of historical months for training (Months 1–24: Jan 2023 – Dec 2024).
    test_months : int, default 12
        Number of historical months for testing (Months 25–36: Jan 2025 – Dec 2025).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)

    Raises
    ------
    ValueError
        If dataset length is insufficient or dates are unordered.
    """
    total_needed = train_months + test_months
    if len(monthly_df) < total_needed:
        raise ValueError(
            f"Dataset length ({len(monthly_df)}) is insufficient for "
            f"train ({train_months}) + test ({test_months}) = {total_needed} months."
        )

    # Verify chronological ordering prior to split
    dates = pd.to_datetime(monthly_df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("Cannot split non-monotonic time series.")

    train_df = monthly_df.iloc[:train_months].copy().reset_index(drop=True)
    test_df = monthly_df.iloc[train_months : total_needed].copy().reset_index(drop=True)

    # Verification: test set must be strictly in the future of train set
    train_max_date = pd.to_datetime(train_df["date"].iloc[-1])
    test_min_date = pd.to_datetime(test_df["date"].iloc[0])
    if train_max_date >= test_min_date:
        raise ValueError(
            f"Temporal leakage detected: train end date ({train_max_date}) "
            f"is not strictly before test start date ({test_min_date})."
        )

    return train_df, test_df


def run_baseline_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
    seasonal_period: int = 12,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """
    Execute baseline forecasting experiment comparing Standard Naive vs Seasonal Naive.

    Controls for data leakage by strictly passing only train_df to baseline fit methods.

    Parameters
    ----------
    train_df : pd.DataFrame
        24-month training set.
    test_df : pd.DataFrame
        12-month test set.
    targets : list[str] | None, default None
        List of target column names. Defaults to Total, Electricity, Travel.
    seasonal_period : int, default 12
        Seasonal lag period for SeasonalNaiveBaseline (m=12).

    Returns
    -------
    summary_df : pd.DataFrame
        Evaluation metrics summary (MAE, RMSE, MAPE, relative improvements).
    predictions : dict[str, dict[str, np.ndarray]]
        Point forecasts for each target and baseline model.
    """
    if targets is None:
        targets = list(DEFAULT_TARGETS)

    horizon = len(test_df)
    summary_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for target in targets:
        if target not in train_df.columns:
            raise KeyError(f"Target column '{target}' missing from train_df.")
        if target not in test_df.columns:
            raise KeyError(f"Target column '{target}' missing from test_df.")

        # Extract 1D training and test series
        y_train = train_df[target].values
        y_test = test_df[target].values

        # -------------------------------------------------------------
        # 1. Baseline 1: Standard Naive (y_{T+h} = y_T)
        # -------------------------------------------------------------
        naive_model = NaiveBaseline().fit(y_train)
        naive_preds = naive_model.predict(horizon=horizon)

        mae_naive = calculate_mae(y_test, naive_preds)
        rmse_naive = calculate_rmse(y_test, naive_preds)
        mape_naive = calculate_mape(y_test, naive_preds)

        # -------------------------------------------------------------
        # 2. Baseline 2: Seasonal Naive (y_t = y_{t-m}, m=12)
        # -------------------------------------------------------------
        s_naive_model = SeasonalNaiveBaseline(seasonal_period=seasonal_period).fit(y_train)
        s_naive_preds = s_naive_model.predict(horizon=horizon)

        mae_s_naive = calculate_mae(y_test, s_naive_preds)
        rmse_s_naive = calculate_rmse(y_test, s_naive_preds)
        mape_s_naive = calculate_mape(y_test, s_naive_preds)

        # -------------------------------------------------------------
        # 3. Seasonal Improvement over Standard Naive
        # -------------------------------------------------------------
        mae_imp = calculate_improvement(mae_naive, mae_s_naive)
        rmse_imp = calculate_improvement(rmse_naive, rmse_s_naive)

        predictions[target] = {
            "Standard Naive": naive_preds,
            "Seasonal Naive": s_naive_preds,
        }

        summary_rows.append({
            "target": target,
            "target_label": TARGET_LABELS.get(target, target),
            "model": "Standard Naive (Flat)",
            "mae_kg": mae_naive,
            "rmse_kg": rmse_naive,
            "mape_pct": mape_naive,
            "mae_improvement_pct": 0.0,
            "rmse_improvement_pct": 0.0,
        })
        summary_rows.append({
            "target": target,
            "target_label": TARGET_LABELS.get(target, target),
            "model": "Seasonal Naive (m=12)",
            "mae_kg": mae_s_naive,
            "rmse_kg": rmse_s_naive,
            "mape_pct": mape_s_naive,
            "mae_improvement_pct": mae_imp,
            "rmse_improvement_pct": rmse_imp,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, predictions


def generate_baseline_comparison_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    output_path: str | Path = "reports/figures/10_baseline_model_comparison.png",
) -> Path:
    """
    Generate publication-quality comparison chart of Standard Naive vs Seasonal Naive.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set with date and target columns.
    test_df : pd.DataFrame
        Test set with date and target columns.
    predictions : dict[str, dict[str, np.ndarray]]
        Dictionary of point predictions per target and model.
    output_path : str | Path, default 'reports/figures/10_baseline_model_comparison.png'
        Destination file path.

    Returns
    -------
    Path
        Path to saved PNG image.
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
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 10), sharex=False)
    fig.suptitle(
        "Phase 10: Forecasting Baseline Comparison (Standard Naive vs. Seasonal Naive)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    train_dates = pd.to_datetime(train_df["date"])
    test_dates = pd.to_datetime(test_df["date"])

    for ax, target in zip(axes, targets):
        y_train = train_df[target].values / 1000.0  # Scale to MTCO2e
        y_test = test_df[target].values / 1000.0
        naive_pt = predictions[target]["Standard Naive"] / 1000.0
        s_naive_pt = predictions[target]["Seasonal Naive"] / 1000.0

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
        # Standard Naive (Flat line)
        ax.plot(
            test_dates,
            naive_pt,
            color="#d95f02",
            linewidth=1.8,
            linestyle="--",
            label="Standard Naive (Flat)",
        )
        # Seasonal Naive (Seasonal repetition)
        ax.plot(
            test_dates,
            s_naive_pt,
            color="#7570b3",
            linewidth=2.2,
            linestyle=":",
            marker="^",
            markersize=4.5,
            label="Seasonal Naive (m=12)",
        )

        # Split marker
        ax.axvline(x=train_dates.iloc[-1], color="#757575", linestyle="--", linewidth=1.2, alpha=0.7)

        title_str = TARGET_LABELS.get(target, target).replace("(kgCO2e)", "(MTCO2e)")
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("MTCO2e", fontweight="bold")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_p, dpi=300)
    plt.close(fig)
    logger.info("Saved baseline comparison plot to %s", out_p)
    return out_p


def run_experiment() -> tuple[pd.DataFrame, Path]:
    """Execute complete Phase 10 baseline experiment pipeline."""
    print("=" * 70)
    print("PHASE 10: FORECASTING BASELINE EVALUATION")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("=" * 70)

    # 1. Load data
    monthly_df = load_and_aggregate_monthly_emissions()

    # 2. Chronological split
    train_df, test_df = split_chronological(monthly_df, train_months=24, test_months=12)
    print(f"Historical Data : {len(monthly_df)} months ({monthly_df['date'].iloc[0]} to {monthly_df['date'].iloc[-1]})")
    print(f"Training Period : {len(train_df)} months ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
    print(f"Testing Period  : {len(test_df)} months ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")
    print(f"Forecast Horizon: {len(test_df)} months")
    print("-" * 70)

    # 3. Run baseline experiment
    summary_df, predictions = run_baseline_experiment(train_df, test_df)

    display_cols = [
        "target",
        "model",
        "mae_kg",
        "rmse_kg",
        "mape_pct",
        "mae_improvement_pct",
    ]
    print("\n--- BASELINE EVALUATION METRICS TABLE ---")
    print(summary_df[display_cols].to_string(index=False))

    # 4. Generate comparison figure
    fig_path = generate_baseline_comparison_plot(train_df, test_df, predictions)
    print(f"\nGenerated visualization: {fig_path.resolve()}")

    # 5. Export summary CSV
    csv_path = Path("data/processed/baseline_evaluation_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"Exported summary table : {csv_path.resolve()}")
    print("=" * 70)

    return summary_df, fig_path


if __name__ == "__main__":
    run_experiment()
