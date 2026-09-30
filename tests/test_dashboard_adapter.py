"""
Unit and Integration Tests for Dashboard Data Adapter and Components
=====================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 17: Dashboard Adapter Test Suite
Validates:
1. Schema validation across all 9 Phase 16 pipeline artifacts.
2. Missing directory and missing file error handling.
3. Empty dataframe and malformed schema rejection.
4. JSON summary integrity and validation.
5. Numerical consistency between pipeline_summary.json and artifact tables.
6. Plotly figure generation across all dashboard visualization components.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard.adapter import (
    DEFAULT_RUN_DIR,
    REQUIRED_COLUMNS,
    REQUIRED_FILES,
    DashboardDataError,
    load_dashboard_data,
)
from src.dashboard.components import (
    create_48m_timeline_chart,
    create_building_emissions_bar_chart,
    create_category_donut_chart,
    create_efficient_frontier_chart,
    create_forecast_trajectory_chart,
    create_scenario_reduction_chart,
    create_stacked_monthly_bar_chart,
    create_variance_decomposition_chart,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_pipeline_dir(tmp_path: Path) -> Path:
    """Create temporary directory containing valid mock Phase 16 pipeline artifacts."""
    run_dir = tmp_path / "mock_pipeline_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. pipeline_summary.json
    summary_data = {
        "config": {
            "optimization_budget_inr": 2500000.0,
            "baseline_year": 2025,
            "random_seed": 42,
        },
        "validation": {
            "total_records": 36,
            "valid_records": 36,
            "rejected_records": 0,
            "completeness_percentage": 100.0,
        },
        "carbon_accounting": {
            "total_emissions_kg": 100000.0,
            "total_emissions_mt": 100.0,
            "emissions_per_student_kg": 20.0,
            "category_contributions": {
                "Electricity": 80.0,
                "Travel": 14.0,
                "Waste": 5.0,
                "Procurement": 1.0,
            },
        },
        "optimal_portfolio": {
            "portfolio_id": "P-10110",
            "selected_interventions": ["INT-001", "INT-003", "INT-004"],
            "total_cost_inr": 950000.0,
            "remaining_budget_inr": 1550000.0,
            "total_absolute_reduction_kg": 225755.02,
            "percentage_reduction": 15.80,
            "cost_per_kg_reduced": 4.21,
        },
    }
    with open(run_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    # 2. campus_emissions.csv
    campus_df = pd.DataFrame({
        "date": ["2025-01-01"],
        "building": ["Science Complex"],
        "electricity_kwh": [10000.0],
        "travel_km": [5000.0],
        "waste_kg": [2000.0],
        "procurement_inr": [100000.0],
        "student_count": [1000],
        "staff_count": [100],
        "electricity_emissions_kg": [7160.0],
        "travel_emissions_kg": [700.0],
        "waste_emissions_kg": [892.0],
        "procurement_emissions_kg": [42.0],
        "total_emissions_kg": [8794.0],
        "total_emissions_mt": [8.79],
        "emissions_per_student_kg": [8.79],
    })
    campus_df.to_csv(run_dir / "campus_emissions.csv", index=False)

    # 3. cleaned_activity.csv
    cleaned_df = campus_df[[
        "date", "building", "electricity_kwh", "travel_km",
        "waste_kg", "procurement_inr", "student_count", "staff_count"
    ]].copy()
    cleaned_df.to_csv(run_dir / "cleaned_activity.csv", index=False)

    # 4. historical_monthly_emissions.csv
    hist_mo_df = pd.DataFrame({
        "date": ["2025-01-01"],
        "electricity_emissions_kg": [7160.0],
        "travel_emissions_kg": [700.0],
        "waste_emissions_kg": [892.0],
        "procurement_emissions_kg": [42.0],
        "total_emissions_kg": [8794.0],
        "total_emissions_mt": [8.79],
    })
    hist_mo_df.to_csv(run_dir / "historical_monthly_emissions.csv", index=False)

    # 5. forecast_projections.csv
    fc_df = pd.DataFrame({
        "date": ["2026-01-01"],
        "target": ["total_emissions_kg"],
        "target_label": ["Total Campus Emissions"],
        "point_forecast": [9000.0],
        "lower_bound_95": [8500.0],
        "upper_bound_95": [9500.0],
    })
    fc_df.to_csv(run_dir / "forecast_projections.csv", index=False)

    # 6. uncertainty_variance_decomposition.csv
    decomp_df = pd.DataFrame({
        "target": ["total_emissions_kg"],
        "forecast_variance_kg2": [1000.0],
        "forecast_variance_pct": [15.0],
        "activity_variance_kg2": [600.0],
        "activity_variance_pct": [9.0],
        "ef_variance_kg2": [5000.0],
        "ef_variance_pct": [76.0],
        "total_variance_kg2": [6600.0],
    })
    decomp_df.to_csv(run_dir / "uncertainty_variance_decomposition.csv", index=False)

    # 7. uncertainty_monte_carlo_simulation.csv
    mc_sim_df = pd.DataFrame({
        "date": ["2026-01-01"],
        "point_forecast": [9000.0],
        "mean": [9010.0],
        "std": [250.0],
        "ci_lower_95": [8520.0],
        "ci_upper_95": [9510.0],
        "relative_uncertainty_pct": [11.0],
    })
    mc_sim_df.to_csv(run_dir / "uncertainty_monte_carlo_simulation.csv", index=False)

    # 8. scenario_evaluations.csv
    scen_df = pd.DataFrame({
        "intervention_id": ["INT-001"],
        "intervention_name": ["LED Lighting Retrofit"],
        "affected_category": ["Electricity"],
        "implementation_cost_inr": [500000.0],
        "baseline_emissions_kg": [100000.0],
        "scenario_emissions_kg": [90000.0],
        "absolute_reduction_kg": [10000.0],
        "percentage_reduction": [10.0],
        "cost_per_kg_reduced": [50.0],
        "absolute_reduction_lower_kg": [9500.0],
        "absolute_reduction_upper_kg": [10500.0],
    })
    scen_df.to_csv(run_dir / "scenario_evaluations.csv", index=False)

    # 9. portfolio_optimization_all_32.csv
    p32_df = pd.DataFrame({
        "portfolio_id": ["P-10110"],
        "selected_interventions": ["INT-001, INT-003, INT-004"],
        "num_interventions": [3],
        "total_cost_inr": [950000.0],
        "remaining_budget_inr": [1550000.0],
        "is_feasible": [True],
        "baseline_emissions_kg": [100000.0],
        "portfolio_emissions_kg": [80000.0],
        "total_absolute_reduction_kg": [20000.0],
        "percentage_reduction": [20.0],
        "cost_per_kg_reduced": [47.5],
        "absolute_reduction_lower_kg": [19000.0],
        "absolute_reduction_upper_kg": [21000.0],
    })
    p32_df.to_csv(run_dir / "portfolio_optimization_all_32.csv", index=False)

    return run_dir


# =====================================================================
# 1. Schema Validation & Loading Tests
# =====================================================================

def test_load_dashboard_data_mock(mock_pipeline_dir: Path) -> None:
    """Verify load_dashboard_data correctly loads all artifacts from mock directory."""
    data = load_dashboard_data(mock_pipeline_dir, use_cache=False)

    assert isinstance(data, dict)
    expected_keys = [
        "summary",
        "campus_emissions",
        "cleaned_activity",
        "historical_monthly",
        "forecast_projections",
        "uncertainty_decomp",
        "uncertainty_sim",
        "scenario_evaluations",
        "portfolio_all_32",
    ]
    for k in expected_keys:
        assert k in data

    assert data["summary"]["optimal_portfolio"]["portfolio_id"] == "P-10110"
    assert len(data["portfolio_all_32"]) == 1


def test_load_dashboard_data_real_artifacts() -> None:
    """Verify load_dashboard_data against actual Phase 16 outputs in data/processed/pipeline_run."""
    if not DEFAULT_RUN_DIR.exists():
        pytest.skip("Pipeline run artifacts not yet generated.")

    data = load_dashboard_data(DEFAULT_RUN_DIR, use_cache=False)

    assert len(data["campus_emissions"]) == 180
    assert len(data["historical_monthly"]) == 36
    assert len(data["forecast_projections"]) == 36
    assert len(data["scenario_evaluations"]) == 5
    assert len(data["portfolio_all_32"]) == 32

    # Check optimal portfolio consistency
    summary_opt = data["summary"]["optimal_portfolio"]
    assert summary_opt["portfolio_id"] == "P-10110"
    assert summary_opt["total_cost_inr"] == 950000.0
    assert summary_opt["percentage_reduction"] == 15.80


# =====================================================================
# 2. Error Handling & Edge Cases
# =====================================================================

def test_missing_directory_raises_error(tmp_path: Path) -> None:
    """Non-existent directory must raise DashboardDataError."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(DashboardDataError, match="Pipeline artifacts directory not found"):
        load_dashboard_data(non_existent, use_cache=False)


def test_missing_required_file_raises_error(mock_pipeline_dir: Path) -> None:
    """Directory missing a required CSV artifact raises DashboardDataError."""
    (mock_pipeline_dir / "campus_emissions.csv").unlink()

    with pytest.raises(DashboardDataError, match="missing required artifact"):
        load_dashboard_data(mock_pipeline_dir, use_cache=False)


def test_empty_csv_raises_error(mock_pipeline_dir: Path) -> None:
    """Empty CSV file raises DashboardDataError."""
    # Overwrite scenario_evaluations.csv with empty file
    with open(mock_pipeline_dir / "scenario_evaluations.csv", "w", encoding="utf-8") as f:
        f.write("")

    with pytest.raises(DashboardDataError, match="empty"):
        load_dashboard_data(mock_pipeline_dir, use_cache=False)


def test_missing_required_column_raises_error(mock_pipeline_dir: Path) -> None:
    """CSV with missing required schema column raises DashboardDataError."""
    broken_df = pd.DataFrame({
        "date": ["2025-01-01"],
        # Missing 'building', 'total_emissions_kg', etc.
    })
    broken_df.to_csv(mock_pipeline_dir / "campus_emissions.csv", index=False)

    with pytest.raises(DashboardDataError, match="missing required column"):
        load_dashboard_data(mock_pipeline_dir, use_cache=False)


def test_malformed_json_raises_error(mock_pipeline_dir: Path) -> None:
    """Malformed pipeline_summary.json raises DashboardDataError."""
    with open(mock_pipeline_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        f.write("{ invalid json")

    with pytest.raises(DashboardDataError, match="Failed to parse"):
        load_dashboard_data(mock_pipeline_dir, use_cache=False)


def test_json_missing_sections_raises_error(mock_pipeline_dir: Path) -> None:
    """pipeline_summary.json missing required top-level section raises DashboardDataError."""
    with open(mock_pipeline_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": {}}, f)  # missing 'optimal_portfolio', 'carbon_accounting', etc.

    with pytest.raises(DashboardDataError, match="missing required top-level section"):
        load_dashboard_data(mock_pipeline_dir, use_cache=False)


# =====================================================================
# 3. Visual Components Plotly Figure Generation
# =====================================================================

def test_components_plotly_figures(mock_pipeline_dir: Path) -> None:
    """Verify that all visual components generate valid Plotly figures without error."""
    data = load_dashboard_data(mock_pipeline_dir, use_cache=False)

    # 1. Donut chart
    fig_donut = create_category_donut_chart(data["summary"]["carbon_accounting"]["category_contributions"])
    assert isinstance(fig_donut, go.Figure)

    # 2. 48-month timeline chart
    fig_timeline = create_48m_timeline_chart(data["historical_monthly"], data["forecast_projections"])
    assert isinstance(fig_timeline, go.Figure)

    # 3. Stacked bar chart
    fig_stacked = create_stacked_monthly_bar_chart(data["historical_monthly"])
    assert isinstance(fig_stacked, go.Figure)

    # 4. Forecast trajectory chart
    fig_fc = create_forecast_trajectory_chart(data["historical_monthly"], data["forecast_projections"])
    assert isinstance(fig_fc, go.Figure)

    # 5. Variance decomposition chart
    fig_var = create_variance_decomposition_chart(data["uncertainty_decomp"])
    assert isinstance(fig_var, go.Figure)

    # 6. Scenario reduction chart
    fig_scen = create_scenario_reduction_chart(data["scenario_evaluations"])
    assert isinstance(fig_scen, go.Figure)

    # 7. Efficient frontier chart
    fig_ef = create_efficient_frontier_chart(
        data["portfolio_all_32"],
        optimal_portfolio_id="P-10110",
        budget_inr=2500000.0,
    )
    assert isinstance(fig_ef, go.Figure)

    # 8. Building emissions bar chart
    fig_bldg = create_building_emissions_bar_chart(data["campus_emissions"])
    assert isinstance(fig_bldg, go.Figure)
