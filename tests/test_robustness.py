"""
Robustness, Security, and Edge-Case Test Suite
==============================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 18: Robustness, Security & Responsible AI
Validates system resilience against:
1. Emission factor under-reporting (zero/negative factors rejection in registry).
2. Negative intervention capital costs exploit rejection across all intervention types.
3. Physical boundary conservation: scenario emissions clamping under extreme over-reduction.
4. Path traversal attacks and sensitive directory protection in PipelineConfig.
5. Corrupted, non-numeric, NaN, infinite, and negative activity data rejection.
6. Fault tolerance and graceful failure in dashboard adapter on corrupted artifacts.
7. Mathematical stability and boundary checks on optimization budgets and simulations.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import (
    CATEGORY_ELECTRICITY,
    CATEGORY_PROCUREMENT,
    CATEGORY_TRAVEL,
    CATEGORY_WASTE,
    EmissionFactor,
    EmissionFactorRegistry,
    calculate_campus_emissions,
    calculate_emissions,
)
from src.dashboard.adapter import DashboardDataError, load_dashboard_data
from src.pipeline_config import PipelineConfig
from src.scenarios.evaluation import evaluate_intervention
from src.scenarios.interventions import (
    ACOptimization,
    LEDLightingRetrofit,
    LowCarbonTransport,
    RooftopSolarInstallation,
    WasteSegregation,
)
from src.scenarios.optimization import evaluate_portfolio


# =====================================================================
# 1. Emission Factor Security & Zero/Negative Rejection
# =====================================================================

def test_registry_rejects_zero_emission_factor() -> None:
    """Ensure EmissionFactorRegistry strictly rejects zero emission factors (EF = 0.0)."""
    registry = EmissionFactorRegistry.default()
    zero_factor = EmissionFactor(
        category="Electricity",
        activity_type="electricity_kwh",
        unit="kgCO2e/kWh",
        factor=0.0,
        source="Unverified Zero Tariff",
        version="2026",
    )
    with pytest.raises(ValueError, match="must be strictly positive"):
        registry.register(zero_factor)


def test_registry_rejects_negative_emission_factor() -> None:
    """Ensure EmissionFactorRegistry rejects negative emission factors."""
    with pytest.raises(ValueError, match="cannot be negative"):
        EmissionFactor(
            category="Electricity",
            activity_type="electricity_kwh",
            unit="kgCO2e/kWh",
            factor=-0.5,
            source="Test",
            version="1.0",
        )


def test_registry_from_dataframe_rejects_zero_or_negative() -> None:
    """Ensure loading emission factors from DataFrame with zero/negative factors is rejected."""
    bad_df = pd.DataFrame([
        {
            "category": "Electricity",
            "activity_type": "electricity_kwh",
            "unit": "kgCO2e/kWh",
            "factor": 0.0,
            "source": "Faulty Data",
            "version": "1.0",
        },
        {
            "category": "Travel",
            "activity_type": "travel_km",
            "unit": "kgCO2e/km",
            "factor": 0.145,
            "source": "India GHG",
            "version": "2023",
        },
    ])
    with pytest.raises(ValueError, match="must be strictly positive"):
        EmissionFactorRegistry.from_dataframe(bad_df)


# =====================================================================
# 2. Intervention Parameter Boundary & Negative Cost Defense
# =====================================================================

@pytest.mark.parametrize(
    "intervention_cls,kwargs",
    [
        (LEDLightingRetrofit, {"capital_cost": -50000.0}),
        (RooftopSolarInstallation, {"capital_cost": -100000.0}),
        (ACOptimization, {"capital_cost": -25000.0}),
        (WasteSegregation, {"capital_cost": -10000.0}),
        (LowCarbonTransport, {"capital_cost": -20000.0}),
    ],
)
def test_interventions_reject_negative_capital_cost(intervention_cls, kwargs) -> None:
    """Verify all intervention dataclasses reject negative capital costs."""
    with pytest.raises(ValueError, match="capital_cost cannot be negative"):
        intervention_cls(**kwargs)


@pytest.mark.parametrize(
    "intervention_cls,kwargs",
    [
        (LEDLightingRetrofit, {"reduction_fraction": 1.5}),
        (LEDLightingRetrofit, {"reduction_fraction": -0.1}),
        (ACOptimization, {"reduction_fraction": 1.1}),
        (ACOptimization, {"reduction_fraction": -0.05}),
        (WasteSegregation, {"diversion_fraction": 2.0}),
        (WasteSegregation, {"diversion_fraction": -0.2}),
        (LowCarbonTransport, {"ev_adoption_fraction": 1.2}),
        (LowCarbonTransport, {"ev_adoption_fraction": -0.1}),
    ],
)
def test_interventions_reject_out_of_bounds_fractions(intervention_cls, kwargs) -> None:
    """Verify interventions reject fractional parameters outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        intervention_cls(**kwargs)


def test_rooftop_solar_rejects_negative_generation() -> None:
    """Verify RooftopSolarInstallation rejects negative generation."""
    with pytest.raises(ValueError, match="cannot be negative"):
        RooftopSolarInstallation(monthly_generation_kwh=-1000.0)


# =====================================================================
# 3. Physical Boundary Enforcement: Extreme Over-Reduction Clamping
# =====================================================================

def test_extreme_solar_generation_clamps_to_zero() -> None:
    """
    Ensure massive solar generation that exceeds baseline demand is clamped to 0.0
    and never results in negative electricity consumption or negative scenario emissions.
    """
    baseline_df = pd.DataFrame({
        "date": ["2025-01-01"],
        "building": ["Science Complex"],
        "electricity_kwh": [500.0],
        "travel_km": [100.0],
        "waste_kg": [50.0],
        "procurement_inr": [1000.0],
    })

    # Giant solar generation: 50,000 kWh >> 500 kWh baseline
    massive_solar = RooftopSolarInstallation(
        monthly_generation_kwh=50000.0,
        capital_cost=1000000.0,
    )

    # 1. Transformed activity must clamp at 0.0
    transformed_df = massive_solar.apply(baseline_df)
    assert transformed_df["electricity_kwh"].iloc[0] == 0.0

    # 2. Evaluation must yield non-negative scenario emissions
    result = evaluate_intervention(baseline_df, massive_solar)
    assert result.scenario_emissions_kg >= 0.0
    assert result.absolute_reduction_kg > 0.0
    # Reduction must not exceed the original electricity emissions
    ef_elec = EmissionFactorRegistry.default().get_factor_value("electricity_kwh")
    max_possible_reduction = round(500.0 * ef_elec, 4)
    assert result.absolute_reduction_kg == pytest.approx(max_possible_reduction, abs=1e-2)


def test_portfolio_extreme_over_reduction_clamping() -> None:
    """Ensure portfolio evaluation clamps emissions to non-negative even if interventions over-reduce."""
    baseline_df = pd.DataFrame({
        "date": ["2025-01-01"],
        "building": ["Main Hall"],
        "electricity_kwh": [100.0],
        "travel_km": [50.0],
        "waste_kg": [20.0],
        "procurement_inr": [500.0],
    })

    massive_solar = RooftopSolarInstallation(monthly_generation_kwh=100000.0, capital_cost=500000.0)
    p_result = evaluate_portfolio(
        baseline_df=baseline_df,
        portfolio=[massive_solar],
        budget=1000000.0,
    )
    assert p_result.portfolio_emissions_kg >= 0.0


# =====================================================================
# 4. Path Traversal & Pipeline Security
# =====================================================================

@pytest.mark.parametrize(
    "bad_path",
    [
        "../../Windows",
        "..\\..\\system32",
        "folder/../../../etc/passwd",
        "../escape",
    ],
)
def test_pipeline_config_rejects_path_traversal_output_dir(bad_path: str) -> None:
    """Verify PipelineConfig raises ValueError on path traversal attempts in output_dir."""
    with pytest.raises(ValueError, match="path traversal sequence"):
        PipelineConfig(output_dir=bad_path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../../raw_data.csv",
        "../outside.csv",
    ],
)
def test_pipeline_config_rejects_path_traversal_input_files(bad_path: str) -> None:
    """Verify PipelineConfig raises ValueError on path traversal in input file paths."""
    with pytest.raises(ValueError, match="path traversal sequence"):
        PipelineConfig(activity_data_path=bad_path)

    with pytest.raises(ValueError, match="path traversal sequence"):
        PipelineConfig(emission_factors_path=bad_path)


def test_pipeline_config_rejects_sensitive_system_directories() -> None:
    """Verify PipelineConfig rejects pointing output_dir directly to Windows or /etc."""
    import sys
    if sys.platform == "win32":
        target = Path("C:/Windows")
    else:
        target = Path("/etc")

    with pytest.raises(ValueError, match="sensitive system directory"):
        PipelineConfig(output_dir=target)


def test_pipeline_config_rejects_invalid_numeric_parameters() -> None:
    """Verify PipelineConfig rejects negative budgets, horizons, or simulations."""
    with pytest.raises(ValueError, match="optimization_budget_inr cannot be negative"):
        PipelineConfig(optimization_budget_inr=-1000.0)

    with pytest.raises(ValueError, match="train_months must be positive"):
        PipelineConfig(train_months=0)

    with pytest.raises(ValueError, match="forecast_horizon_months must be positive"):
        PipelineConfig(forecast_horizon_months=-12)

    with pytest.raises(ValueError, match="uncertainty_simulations must be positive"):
        PipelineConfig(uncertainty_simulations=0)


# =====================================================================
# 5. Data Corruption & Mathematical Stability
# =====================================================================

def test_carbon_accounting_rejects_nan_activity() -> None:
    """Verify calculate_campus_emissions strictly rejects NaN values."""
    bad_df = pd.DataFrame({
        "electricity_kwh": [1000.0, np.nan],
        "travel_km": [500.0, 500.0],
        "waste_kg": [200.0, 200.0],
        "procurement_inr": [10000.0, 10000.0],
    })
    with pytest.raises(ValueError, match="contains 1 NaN/null values"):
        calculate_campus_emissions(bad_df)


def test_carbon_accounting_rejects_infinite_activity() -> None:
    """Verify calculate_campus_emissions strictly rejects Infinite values."""
    bad_df = pd.DataFrame({
        "electricity_kwh": [1000.0, np.inf],
        "travel_km": [500.0, 500.0],
        "waste_kg": [200.0, 200.0],
        "procurement_inr": [10000.0, 10000.0],
    })
    with pytest.raises(ValueError, match="contains Infinite values"):
        calculate_campus_emissions(bad_df)


def test_carbon_accounting_rejects_negative_activity() -> None:
    """Verify calculate_campus_emissions strictly rejects negative values."""
    bad_df = pd.DataFrame({
        "electricity_kwh": [1000.0, -50.0],
        "travel_km": [500.0, 500.0],
        "waste_kg": [200.0, 200.0],
        "procurement_inr": [10000.0, 10000.0],
    })
    with pytest.raises(ValueError, match="contains negative values"):
        calculate_campus_emissions(bad_df)


def test_carbon_accounting_rejects_missing_columns() -> None:
    """Verify calculate_campus_emissions rejects missing activity columns."""
    bad_df = pd.DataFrame({
        "electricity_kwh": [1000.0],
        "travel_km": [500.0],
        # waste_kg and procurement_inr missing
    })
    with pytest.raises(ValueError, match="missing required column"):
        calculate_campus_emissions(bad_df)


# =====================================================================
# 6. Dashboard Adapter Fault Tolerance & Graceful Error Handling
# =====================================================================

def test_dashboard_adapter_handles_missing_directory(tmp_path: Path) -> None:
    """Verify load_dashboard_data raises DashboardDataError when directory is missing."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(DashboardDataError, match="Pipeline artifacts directory not found"):
        load_dashboard_data(non_existent, use_cache=False)


def test_dashboard_adapter_handles_corrupt_summary_json(tmp_path: Path) -> None:
    """Verify load_dashboard_data handles invalid/corrupted JSON gracefully."""
    from src.dashboard.adapter import REQUIRED_FILES
    run_dir = tmp_path / "corrupt_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    for fn in REQUIRED_FILES:
        (run_dir / fn).write_text("a,b\n1,2\n", encoding="utf-8")
    (run_dir / "pipeline_summary.json").write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(DashboardDataError, match="Failed to parse"):
        load_dashboard_data(run_dir, use_cache=False)


def test_dashboard_adapter_handles_empty_csv_artifact(tmp_path: Path) -> None:
    """Verify load_dashboard_data rejects empty CSV artifacts."""
    from src.dashboard.adapter import REQUIRED_FILES
    run_dir = tmp_path / "empty_csv_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    for fn in REQUIRED_FILES:
        (run_dir / fn).write_text("a,b\n1,2\n", encoding="utf-8")
    with open(run_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": {}, "validation": {}, "carbon_accounting": {}, "optimal_portfolio": {}}, f)

    # Empty campus_emissions.csv
    (run_dir / "campus_emissions.csv").write_text("", encoding="utf-8")

    with pytest.raises(DashboardDataError, match="empty"):
        load_dashboard_data(run_dir, use_cache=False)
