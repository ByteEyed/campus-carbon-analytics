"""
Campus Carbon Analytics - Dashboard Data Adapter
================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 17: Dashboard Data Ingestion & Caching Layer
Decouples the Streamlit presentation interface from raw filesystem operations.
Ingests, validates schemas, and caches precomputed Phase 16 pipeline artifacts.
Strictly read-only: executes zero model fitting, simulation, or optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger("campus_carbon.dashboard.adapter")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUN_DIR = REPO_ROOT / "data" / "processed" / "pipeline_run"
DEFAULT_METRICS_PATH = REPO_ROOT / "data" / "processed" / "primary_model_evaluation_summary.csv"

REQUIRED_FILES: list[str] = [
    "pipeline_summary.json",
    "campus_emissions.csv",
    "cleaned_activity.csv",
    "historical_monthly_emissions.csv",
    "forecast_projections.csv",
    "uncertainty_variance_decomposition.csv",
    "uncertainty_monte_carlo_simulation.csv",
    "scenario_evaluations.csv",
    "portfolio_optimization_all_32.csv",
]

# Required columns per artifact table to ensure schema integrity
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "campus_emissions": [
        "date",
        "building",
        "electricity_emissions_kg",
        "travel_emissions_kg",
        "waste_emissions_kg",
        "procurement_emissions_kg",
        "total_emissions_kg",
        "total_emissions_mt",
    ],
    "historical_monthly": [
        "date",
        "electricity_emissions_kg",
        "travel_emissions_kg",
        "waste_emissions_kg",
        "procurement_emissions_kg",
        "total_emissions_kg",
        "total_emissions_mt",
    ],
    "forecast_projections": [
        "date",
        "target",
        "point_forecast",
        "lower_bound_95",
        "upper_bound_95",
    ],
    "uncertainty_decomp": [
        "target",
        "forecast_variance_pct",
        "activity_variance_pct",
        "ef_variance_pct",
        "total_variance_kg2",
    ],
    "uncertainty_sim": [
        "date",
        "point_forecast",
        "mean",
        "std",
        "ci_lower_95",
        "ci_upper_95",
    ],
    "scenario_evaluations": [
        "intervention_id",
        "intervention_name",
        "implementation_cost_inr",
        "baseline_emissions_kg",
        "scenario_emissions_kg",
        "absolute_reduction_kg",
        "percentage_reduction",
        "cost_per_kg_reduced",
    ],
    "portfolio_all_32": [
        "portfolio_id",
        "selected_interventions",
        "num_interventions",
        "total_cost_inr",
        "total_absolute_reduction_kg",
        "percentage_reduction",
        "is_feasible",
    ],
}


class DashboardDataError(Exception):
    """Exception raised when required pipeline artifacts are missing or malformed."""


def _validate_table_schema(name: str, df: pd.DataFrame) -> None:
    """Validate that DataFrame is non-empty and contains all required columns."""
    if not isinstance(df, pd.DataFrame):
        raise DashboardDataError(f"Expected DataFrame for '{name}', got {type(df).__name__}")
    if df.empty:
        raise DashboardDataError(f"Artifact table '{name}' is empty (0 records).")

    expected_cols = REQUIRED_COLUMNS.get(name)
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise DashboardDataError(
                f"Artifact table '{name}' is missing required column(s): {missing}. "
                f"Present columns: {list(df.columns)}"
            )


def _load_data_internal(run_dir: Path) -> dict[str, Any]:
    """Internal synchronous loading routine executed under caching."""
    if not run_dir.exists() or not run_dir.is_dir():
        raise DashboardDataError(
            f"Pipeline artifacts directory not found: '{run_dir.resolve()}'. "
            "Please execute the Phase 16 integrated pipeline (python -m src.pipeline) "
            "before launching the executive dashboard."
        )

    # 1. Verify existence of all required files
    missing_files = [fn for fn in REQUIRED_FILES if not (run_dir / fn).exists()]
    if missing_files:
        raise DashboardDataError(
            f"Directory '{run_dir.resolve()}' is missing required artifact(s): {missing_files}. "
            "Please ensure Phase 16 execution completed successfully with artifact export."
        )

    # 2. Ingest JSON summary
    summary_path = run_dir / "pipeline_summary.json"
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    except Exception as exc:
        raise DashboardDataError(f"Failed to parse '{summary_path.name}': {exc}") from exc

    for req_key in ["config", "validation", "carbon_accounting", "optimal_portfolio"]:
        if req_key not in summary_data:
            raise DashboardDataError(
                f"JSON summary is missing required top-level section: '{req_key}'"
            )

    # 3. Ingest CSV DataFrames
    try:
        campus_emissions_df = pd.read_csv(run_dir / "campus_emissions.csv")
        cleaned_activity_df = pd.read_csv(run_dir / "cleaned_activity.csv")
        historical_monthly_df = pd.read_csv(run_dir / "historical_monthly_emissions.csv")
        forecast_projections_df = pd.read_csv(run_dir / "forecast_projections.csv")
        uncertainty_decomp_df = pd.read_csv(run_dir / "uncertainty_variance_decomposition.csv")
        uncertainty_sim_df = pd.read_csv(run_dir / "uncertainty_monte_carlo_simulation.csv")
        scenario_evaluations_df = pd.read_csv(run_dir / "scenario_evaluations.csv")
        portfolio_all_32_df = pd.read_csv(run_dir / "portfolio_optimization_all_32.csv")
    except Exception as exc:
        raise DashboardDataError(f"Error reading CSV artifact from '{run_dir}': {exc}") from exc

    # 4. Schema validation
    _validate_table_schema("campus_emissions", campus_emissions_df)
    _validate_table_schema("historical_monthly", historical_monthly_df)
    _validate_table_schema("forecast_projections", forecast_projections_df)
    _validate_table_schema("uncertainty_decomp", uncertainty_decomp_df)
    _validate_table_schema("uncertainty_sim", uncertainty_sim_df)
    _validate_table_schema("scenario_evaluations", scenario_evaluations_df)
    _validate_table_schema("portfolio_all_32", portfolio_all_32_df)

    # 5. Ingest optional model performance metrics if present
    forecast_metrics_df: pd.DataFrame | None = None
    if DEFAULT_METRICS_PATH.exists():
        try:
            forecast_metrics_df = pd.read_csv(DEFAULT_METRICS_PATH)
        except Exception:
            forecast_metrics_df = None

    logger.info("Successfully loaded and validated all Phase 16 dashboard artifacts from %s", run_dir)

    return {
        "summary": summary_data,
        "campus_emissions": campus_emissions_df,
        "cleaned_activity": cleaned_activity_df,
        "historical_monthly": historical_monthly_df,
        "forecast_projections": forecast_projections_df,
        "uncertainty_decomp": uncertainty_decomp_df,
        "uncertainty_sim": uncertainty_sim_df,
        "scenario_evaluations": scenario_evaluations_df,
        "portfolio_all_32": portfolio_all_32_df,
        "forecast_metrics": forecast_metrics_df,
    }


@st.cache_data(show_spinner=False)
def _cached_load_dashboard_data(run_dir_str: str) -> dict[str, Any]:
    """Cached loader utilizing Streamlit's in-memory data cache."""
    return _load_data_internal(Path(run_dir_str))


def load_dashboard_data(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Public entry point for ingesting precomputed Phase 16 dashboard artifacts.

    Parameters
    ----------
    run_dir : str | Path, default "data/processed/pipeline_run"
        Path to directory holding Phase 16 exported CSVs and JSON summary.
    use_cache : bool, default True
        Whether to route through Streamlit's in-memory cache manager.

    Returns
    -------
    dict[str, Any]
        Dictionary containing verified DataFrames and summary dictionary.

    Raises
    ------
    DashboardDataError
        If directory or required artifacts are missing, empty, or schema-invalid.
    """
    path_obj = Path(run_dir)
    if not path_obj.is_absolute() and not path_obj.exists():
        alt_path = REPO_ROOT / path_obj
        if alt_path.exists():
            path_obj = alt_path

    if use_cache:
        return _cached_load_dashboard_data(str(path_obj.resolve()))
    return _load_data_internal(path_obj)
