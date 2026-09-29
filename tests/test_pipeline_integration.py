"""
Integration Tests for End-to-End Analytics Pipeline
===================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 16: End-to-End Pipeline Integration Test Suite
Validates:
1. Complete happy-path pipeline execution across all stages.
2. Input contract validation (empty dataframe, missing required columns).
3. Output contract validation (non-negativity, interval ordering, powerset completeness).
4. Missing input files handling (FileNotFoundError).
5. Malformed activity data handling (negative values, missing values).
6. Missing emission factors handling.
7. Insufficient time-series history (< 36 months).
8. Invalid budget rejection (ValueError for budget < 0).
9. Deterministic repeated execution (identical results with identical random seed).
10. Upstream failure propagation (no silent suppression of errors).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import EmissionFactorRegistry
from src.pipeline import (
    PipelineContractError,
    PipelineResult,
    run_pipeline,
    validate_emissions_dataframe,
    validate_forecast_projections,
    validate_input_dataframe,
    validate_time_series_continuity,
)
from src.pipeline_config import PipelineConfig


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_36m_activity_df() -> pd.DataFrame:
    """Generate 36 contiguous monthly records (2023-01 to 2025-12) for a single facility."""
    dates = pd.date_range(start="2023-01-01", periods=36, freq="MS").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates,
        "building": ["Science & Engineering Complex"] * 36,
        "electricity_kwh": [10000.0 + (i * 50.0) for i in range(36)],
        "travel_km": [5000.0 + (i * 20.0) for i in range(36)],
        "waste_kg": [2000.0 + (i * 10.0) for i in range(36)],
        "procurement_inr": [300000.0 + (i * 1000.0) for i in range(36)],
        "student_count": [1200] * 36,
        "staff_count": [150] * 36,
    })


@pytest.fixture
def temp_pipeline_env(tmp_path: Path, synthetic_36m_activity_df: pd.DataFrame) -> PipelineConfig:
    """Create temporary directory containing valid input files and configuration."""
    act_path = tmp_path / "test_activity.csv"
    synthetic_36m_activity_df.to_csv(act_path, index=False)

    ef_path = tmp_path / "test_factors.csv"
    ef_df = pd.DataFrame([
        {
            "category": "Electricity",
            "activity_type": "electricity_kwh",
            "unit": "kgCO2e/kWh",
            "factor": 0.716,
            "source": "CEA CO2 Baseline Database v20.0",
            "version": "2024",
        },
        {
            "category": "Travel",
            "activity_type": "travel_km",
            "unit": "kgCO2e/km",
            "factor": 0.145,
            "source": "India GHG Program Transport Tool",
            "version": "2023",
        },
        {
            "category": "Waste",
            "activity_type": "waste_kg",
            "unit": "kgCO2e/kg",
            "factor": 0.446,
            "source": "UK DESNZ / DEFRA Conversion Factors",
            "version": "2024",
        },
        {
            "category": "Procurement",
            "activity_type": "procurement_inr",
            "unit": "kgCO2e/INR",
            "factor": 0.00035,
            "source": "Exiobase v3.8.2 EEIO India Model",
            "version": "2023",
        },
    ])
    ef_df.to_csv(ef_path, index=False)

    out_dir = tmp_path / "pipeline_output"

    return PipelineConfig(
        activity_data_path=act_path,
        emission_factors_path=ef_path,
        output_dir=out_dir,
        optimization_budget_inr=1_000_000.0,
        random_seed=42,
        train_months=24,
        forecast_horizon_months=12,
        uncertainty_simulations=20,  # Fast for unit tests
        baseline_year=2025,
        fail_fast=True,
        export_artifacts_on_finish=True,
    )


# =====================================================================
# 1. Complete Happy-Path Pipeline Test
# =====================================================================

def test_complete_happy_path_pipeline(temp_pipeline_env: PipelineConfig) -> None:
    """Verify that run_pipeline executes smoothly through all 9 stages to completion."""
    res = run_pipeline(temp_pipeline_env)

    assert isinstance(res, PipelineResult)

    # 1. Validation Report
    assert res.validation_report.total_records == 36
    assert res.validation_report.valid_records == 36
    assert res.validation_report.rejected_records == 0
    assert res.validation_report.completeness_percentage == 100.0

    # 2. Emissions & Accounting
    assert res.accounting_summary.total_emissions_kg > 0.0
    assert len(res.historical_monthly_emissions) == 36

    # 3. Forecast Projections
    assert len(res.forecast_projections) == 36  # 3 targets * 12 months
    assert set(res.forecast_projections["target"].unique()) == {
        "total_emissions_kg",
        "electricity_emissions_kg",
        "travel_emissions_kg",
    }

    # 4. Uncertainty
    assert len(res.uncertainty_decomposition) == 3
    assert len(res.uncertainty_simulation) == 12

    # 5. Scenarios
    assert len(res.scenario_evaluations) == 5

    # 6. Optimization
    assert len(res.all_portfolios) == 32
    assert res.optimal_portfolio.is_feasible is True
    assert res.optimal_portfolio.total_cost_inr <= temp_pipeline_env.optimization_budget_inr

    # 7. Artifact Export
    out_dir = temp_pipeline_env.output_dir
    assert out_dir.exists()
    assert (out_dir / "cleaned_activity.csv").exists()
    assert (out_dir / "campus_emissions.csv").exists()
    assert (out_dir / "historical_monthly_emissions.csv").exists()
    assert (out_dir / "forecast_projections.csv").exists()
    assert (out_dir / "uncertainty_variance_decomposition.csv").exists()
    assert (out_dir / "uncertainty_monte_carlo_simulation.csv").exists()
    assert (out_dir / "scenario_evaluations.csv").exists()
    assert (out_dir / "portfolio_optimization_all_32.csv").exists()
    assert (out_dir / "pipeline_summary.json").exists()


# =====================================================================
# 2. Input Contract Validation Tests
# =====================================================================

def test_input_contract_empty_dataframe() -> None:
    """Passing an empty dataframe must fail input contract validation."""
    with pytest.raises(PipelineContractError, match="cannot be empty"):
        validate_input_dataframe(pd.DataFrame())


def test_input_contract_missing_columns() -> None:
    """Missing required activity columns must raise PipelineContractError."""
    broken_df = pd.DataFrame({"date": ["2025-01-01"], "electricity_kwh": [100.0]})
    with pytest.raises(PipelineContractError, match="missing required column"):
        validate_input_dataframe(broken_df)


# =====================================================================
# 3. Output Contract Validation Tests
# =====================================================================

def test_output_contract_forecast_sandwiching() -> None:
    """Forecast validator must enforce lower <= point <= upper."""
    # Valid forecast
    valid_fc = pd.DataFrame({
        "date": ["2026-01-01"],
        "target": ["total_emissions_kg"],
        "point_forecast": [1000.0],
        "lower_bound_95": [900.0],
        "upper_bound_95": [1100.0],
    })
    validate_forecast_projections(valid_fc)  # should not raise

    # Broken sandwiching: lower > point
    broken_fc = pd.DataFrame({
        "date": ["2026-01-01"],
        "target": ["total_emissions_kg"],
        "point_forecast": [1000.0],
        "lower_bound_95": [1050.0],
        "upper_bound_95": [1100.0],
    })
    with pytest.raises(PipelineContractError, match="lower bound .* exceeds point forecast"):
        validate_forecast_projections(broken_fc)


def test_output_contract_negative_emissions() -> None:
    """Emissions dataframe validator must reject negative emission values."""
    broken_em = pd.DataFrame({
        "date": ["2025-01-01"],
        "electricity_emissions_kg": [100.0],
        "travel_emissions_kg": [50.0],
        "waste_kg": [10.0],
        "waste_emissions_kg": [10.0],
        "procurement_emissions_kg": [10.0],
        "total_emissions_kg": [-20.0],  # Negative
        "total_emissions_mt": [-0.02],
    })
    with pytest.raises(PipelineContractError, match="contains negative emissions"):
        validate_emissions_dataframe(broken_em)


# =====================================================================
# 4. Missing Input Files
# =====================================================================

def test_missing_activity_file_raises_filenotfound(tmp_path: Path) -> None:
    """Non-existent activity file path raises FileNotFoundError."""
    cfg = PipelineConfig(
        activity_data_path=tmp_path / "non_existent_activity.csv",
        emission_factors_path=tmp_path / "non_existent_ef.csv",
    )
    with pytest.raises(FileNotFoundError, match="Activity data file not found"):
        run_pipeline(cfg)


def test_missing_factors_file_raises_filenotfound(
    temp_pipeline_env: PipelineConfig,
    tmp_path: Path,
) -> None:
    """Non-existent emission factors path raises FileNotFoundError."""
    cfg = PipelineConfig(
        activity_data_path=temp_pipeline_env.activity_data_path,
        emission_factors_path=tmp_path / "non_existent_factors.csv",
    )
    with pytest.raises(FileNotFoundError, match="Emission factors file not found"):
        run_pipeline(cfg)


# =====================================================================
# 5. Malformed Activity Data
# =====================================================================

def test_malformed_activity_data_rejected(
    temp_pipeline_env: PipelineConfig,
    synthetic_36m_activity_df: pd.DataFrame,
) -> None:
    """Malformed activity data (negative values) causes pipeline rejection under fail_fast=True."""
    corrupted_df = synthetic_36m_activity_df.copy()
    corrupted_df.loc[5, "electricity_kwh"] = -999.0

    with pytest.raises(PipelineContractError, match="validation failed"):
        run_pipeline(temp_pipeline_env, raw_activity_df=corrupted_df)


# =====================================================================
# 6. Missing Emission Factors
# =====================================================================

def test_missing_emission_factor_raises(
    temp_pipeline_env: PipelineConfig,
    synthetic_36m_activity_df: pd.DataFrame,
) -> None:
    """Registry missing an activity factor raises ValueError during carbon accounting."""
    incomplete_factors = pd.DataFrame([
        {
            "category": "Travel",
            "activity_type": "travel_km",
            "unit": "kgCO2e/km",
            "factor": 0.145,
            "source": "India GHG Program Transport Tool",
            "version": "2023",
        }
        # Electricity, Waste, and Procurement missing!
    ])
    incomplete_reg = EmissionFactorRegistry.from_dataframe(incomplete_factors)

    with pytest.raises((ValueError, KeyError), match="is missing or not registered|No factor configured"):
        run_pipeline(
            temp_pipeline_env,
            raw_activity_df=synthetic_36m_activity_df,
            registry=incomplete_reg,
        )


# =====================================================================
# 7. Insufficient History (< 36 Months)
# =====================================================================

def test_insufficient_forecast_history_raises() -> None:
    """Fewer than 36 months must fail continuity contract."""
    short_df = pd.DataFrame({
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "total_emissions_kg": [100.0, 110.0, 105.0],
    })
    with pytest.raises(PipelineContractError, match="Insufficient historical timeline"):
        validate_time_series_continuity(short_df, required_history_months=36)


def test_timeline_gap_raises() -> None:
    """Timeline with missing months must raise PipelineContractError."""
    gapped_df = pd.DataFrame({
        "date": [f"2023-{m:02d}-01" for m in range(1, 13)]
        + [f"2024-{m:02d}-01" for m in range(1, 13)]
        # Skip 2025-01, 2025-02
        + [f"2025-{m:02d}-01" for m in range(3, 13)],
        "total_emissions_kg": [100.0] * 34,
    })
    with pytest.raises(PipelineContractError, match="Monthly time series has temporal gaps"):
        validate_time_series_continuity(gapped_df, required_history_months=34)


# =====================================================================
# 8. Invalid Budget Rejection
# =====================================================================

def test_invalid_negative_budget_raises() -> None:
    """Negative optimization budget must raise ValueError during config validation."""
    with pytest.raises(ValueError, match="optimization_budget_inr cannot be negative"):
        PipelineConfig(optimization_budget_inr=-1000.0)


# =====================================================================
# 9. Deterministic Repeated Execution
# =====================================================================

def test_deterministic_repeated_execution(temp_pipeline_env: PipelineConfig) -> None:
    """Two executions with the same seed must produce identical optimal portfolios and forecasts."""
    res1 = run_pipeline(temp_pipeline_env)
    res2 = run_pipeline(temp_pipeline_env)

    # Optimal portfolio identity and metrics
    assert res1.optimal_portfolio.portfolio_id == res2.optimal_portfolio.portfolio_id
    assert res1.optimal_portfolio.total_cost_inr == res2.optimal_portfolio.total_cost_inr
    assert res1.optimal_portfolio.total_absolute_reduction_kg == res2.optimal_portfolio.total_absolute_reduction_kg
    assert res1.optimal_portfolio.percentage_reduction == res2.optimal_portfolio.percentage_reduction

    # Point forecasts identity
    pd.testing.assert_frame_equal(res1.forecast_projections, res2.forecast_projections)

    # Accounting summary identity
    assert res1.accounting_summary.total_emissions_kg == res2.accounting_summary.total_emissions_kg


# =====================================================================
# 10. Upstream Failure Propagation
# =====================================================================

def test_upstream_failure_propagation(
    temp_pipeline_env: PipelineConfig,
    synthetic_36m_activity_df: pd.DataFrame,
) -> None:
    """Ensure exceptions raised in upstream steps bubble up unmodified."""
    corrupted_df = synthetic_36m_activity_df.copy()
    corrupted_df["electricity_kwh"] = np.nan  # NaN in required activity column

    with pytest.raises(PipelineContractError):
        run_pipeline(temp_pipeline_env, raw_activity_df=corrupted_df)
