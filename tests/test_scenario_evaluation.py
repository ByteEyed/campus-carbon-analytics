"""
Unit and Integration Tests for Scenario Evaluation Engine
=========================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 14: Scenario Evaluation Test Suite
Validates:
1. Isolated One-At-a-Time (OAT) evaluation of all five Phase 13 interventions.
2. Hand-calculated known-answer mathematical precision.
3. Baseline immutability and mutation leakage prevention.
4. Division-by-zero protection: absolute_reduction == 0 yields cost_per_kg == inf.
5. Zero implementation cost yields cost_per_kg == 0.0.
6. Zero baseline emissions handled gracefully without ZeroDivisionError.
7. Maladaptive interventions (emissions increase, reduction < 0) allowed and not clamped.
8. Physical constraint verification (scenario_emissions >= 0, cost >= 0).
9. Phase 12 activity sensitivity bounds integration.
10. Deterministic repeated evaluation and output DataFrame structuring.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import EmissionFactorRegistry
from src.scenarios.evaluation import (
    ScenarioEvaluationResult,
    evaluate_all_interventions,
    evaluate_intervention,
    generate_baseline_activity_bounds,
    get_12m_baseline,
    scenarios_to_dataframe,
)
from src.scenarios.interventions import (
    ACOptimization,
    BaseIntervention,
    LEDLightingRetrofit,
    LowCarbonTransport,
    RooftopSolarInstallation,
    WasteSegregation,
    get_default_interventions,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_baseline_df() -> pd.DataFrame:
    """Standard single-facility 12-month baseline DataFrame."""
    dates = [f"2025-{m:02d}-01" for m in range(1, 13)]
    return pd.DataFrame({
        "date": dates,
        "building": ["Science & Engineering Complex"] * 12,
        "electricity_kwh": [10000.0] * 12,  # 120,000 kWh/yr
        "travel_km": [5000.0] * 12,        # 60,000 km/yr
        "waste_kg": [2000.0] * 12,          # 24,000 kg/yr
        "procurement_inr": [300000.0] * 12, # 3,600,000 INR/yr
    })


@pytest.fixture
def canonical_12m_baseline() -> pd.DataFrame:
    """Load canonical 2025 baseline from project dataset."""
    return get_12m_baseline(year=2025)


# =====================================================================
# Mock Interventions for Edge Cases
# =====================================================================

class ZeroReductionIntervention(BaseIntervention):
    """Dummy intervention that produces zero activity modification."""

    def __init__(self, cost: float = 50000.0) -> None:
        super().__init__(
            id="INT-ZERO",
            name="Zero Modification Dummy",
            category="Electricity",
            target_column="electricity_kwh",
            capital_cost=cost,
            lifespan_years=5,
        )

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.target_column].to_numpy(dtype=float)


class MaladaptiveIntervention(BaseIntervention):
    """Dummy maladaptive intervention that increases emissions (e.g. embodied load)."""

    def __init__(self, increase_ratio: float = 0.20, cost: float = 100000.0) -> None:
        super().__init__(
            id="INT-MAL",
            name="Maladaptive Activity Spiker",
            category="Electricity",
            target_column="electricity_kwh",
            capital_cost=cost,
            lifespan_years=5,
        )
        self.increase_ratio = increase_ratio

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.target_column].to_numpy(dtype=float) * (1.0 + self.increase_ratio)


# =====================================================================
# 1. Known-Answer Calculation Tests
# =====================================================================

def test_known_answer_led_lighting(mock_baseline_df: pd.DataFrame) -> None:
    """
    Hand-Calculated Known-Answer:
    Baseline:
      12 months * 10,000 kWh = 120,000 kWh.
      EF = 0.716 kgCO2e/kWh.
      Base electricity emissions = 120,000 * 0.716 = 85,920.0 kgCO2e.
    LED 12% Reduction:
      Electricity reduced by 12% = 14,400 kWh saved.
      Avoided electricity emissions = 14,400 * 0.716 = 10,310.4 kgCO2e.
      Cost = INR 500,000.
      MAC = 500,000 / 10,310.4 = 48.4947 INR/kg.
    """
    led = LEDLightingRetrofit(reduction_fraction=0.12, capital_cost=500000.0)
    res = evaluate_intervention(mock_baseline_df, led)

    # Reduction matches hand-calculation exactly
    assert math.isclose(res.absolute_reduction_kg, 10310.4, abs_tol=1e-2)
    expected_mac = round(500000.0 / 10310.4, 4)
    assert math.isclose(res.cost_per_kg_reduced, expected_mac, abs_tol=1e-2)
    assert res.intervention_id == "INT-001"
    assert res.affected_category == "Electricity"


def test_known_answer_solar_installation(mock_baseline_df: pd.DataFrame) -> None:
    """
    Hand-Calculated Known-Answer:
    Monthly generation = 4,000 kWh.
    12 months * 4,000 kWh = 48,000 kWh clean generation.
    Avoided electricity emissions = 48,000 * 0.716 = 34,368.0 kgCO2e.
    Cost = INR 2,000,000.
    MAC = 2,000,000 / 34,368.0 = 58.1937 INR/kg.
    """
    solar = RooftopSolarInstallation(monthly_generation_kwh=4000.0, capital_cost=2000000.0)
    res = evaluate_intervention(mock_baseline_df, solar)

    assert math.isclose(res.absolute_reduction_kg, 34368.0, abs_tol=1e-2)
    expected_mac = round(2000000.0 / 34368.0, 4)
    assert math.isclose(res.cost_per_kg_reduced, expected_mac, abs_tol=1e-2)


# =====================================================================
# 2. Evaluation of All Five Canonical Interventions
# =====================================================================

def test_evaluate_all_five_interventions_standalone(canonical_12m_baseline: pd.DataFrame) -> None:
    """Verify all 5 Phase 13 interventions evaluate successfully on the canonical 2025 baseline."""
    results = evaluate_all_interventions(canonical_12m_baseline)
    assert len(results) == 5

    ids = [r.intervention_id for r in results]
    assert ids == ["INT-001", "INT-002", "INT-003", "INT-004", "INT-005"]

    for r in results:
        assert isinstance(r, ScenarioEvaluationResult)
        assert r.baseline_emissions_kg > 0.0
        assert r.scenario_emissions_kg > 0.0
        assert r.scenario_emissions_kg < r.baseline_emissions_kg
        assert r.absolute_reduction_kg > 0.0
        assert 0.0 < r.percentage_reduction < 100.0
        assert r.cost_per_kg_reduced > 0.0
        assert not math.isinf(r.cost_per_kg_reduced)

        # Consistency check: baseline - scenario == reduction
        assert math.isclose(
            r.absolute_reduction_kg,
            r.baseline_emissions_kg - r.scenario_emissions_kg,
            abs_tol=1e-2,
        )


# =====================================================================
# 3. Baseline Immutability & Mutation Leakage Protection
# =====================================================================

def test_baseline_immutability_across_sequential_evaluations(mock_baseline_df: pd.DataFrame) -> None:
    """Evaluate 3 interventions sequentially; verify baseline_df is never modified."""
    original_copy = mock_baseline_df.copy(deep=True)

    led = LEDLightingRetrofit(reduction_fraction=0.15)
    solar = RooftopSolarInstallation(monthly_generation_kwh=5000.0)
    ac = ACOptimization(reduction_fraction=0.20)

    res1 = evaluate_intervention(mock_baseline_df, led)
    res2 = evaluate_intervention(mock_baseline_df, solar)
    res3 = evaluate_intervention(mock_baseline_df, ac)

    # Baseline emissions must remain strictly identical across all 3 evaluations
    assert res1.baseline_emissions_kg == res2.baseline_emissions_kg == res3.baseline_emissions_kg

    # Input DataFrame must be completely unchanged in memory
    pd.testing.assert_frame_equal(mock_baseline_df, original_copy, check_exact=True)


# =====================================================================
# 4. Zero Reduction & Division-by-Zero Safety Tests
# =====================================================================

def test_zero_reduction_yields_infinite_mac(mock_baseline_df: pd.DataFrame) -> None:
    """If absolute_reduction == 0, cost_per_kg_reduced must be float('inf') without crashing."""
    zero_inv = ZeroReductionIntervention(cost=75000.0)
    res = evaluate_intervention(mock_baseline_df, zero_inv)

    assert res.absolute_reduction_kg == 0.0
    assert res.percentage_reduction == 0.0
    assert math.isinf(res.cost_per_kg_reduced)
    assert res.cost_per_kg_reduced == float("inf")


def test_zero_cost_with_positive_reduction(mock_baseline_df: pd.DataFrame) -> None:
    """If cost == 0 and reduction > 0, cost_per_kg_reduced must be 0.0."""
    free_led = LEDLightingRetrofit(capital_cost=0.0, reduction_fraction=0.10)
    res = evaluate_intervention(mock_baseline_df, free_led)

    assert res.absolute_reduction_kg > 0.0
    assert res.cost_per_kg_reduced == 0.0


def test_zero_reduction_with_zero_cost(mock_baseline_df: pd.DataFrame) -> None:
    """If cost == 0 and reduction == 0, cost_per_kg_reduced is float('inf')."""
    free_zero = ZeroReductionIntervention(cost=0.0)
    res = evaluate_intervention(mock_baseline_df, free_zero)

    assert res.absolute_reduction_kg == 0.0
    assert math.isinf(res.cost_per_kg_reduced)


# =====================================================================
# 5. Maladaptive Intervention (Negative Reduction) Tests
# =====================================================================

def test_maladaptive_intervention_allowed_and_not_clamped(mock_baseline_df: pd.DataFrame) -> None:
    """Ensure maladaptive interventions (emissions increase) yield negative reduction without crashing."""
    mal_inv = MaladaptiveIntervention(increase_ratio=0.10, cost=100000.0)
    res = evaluate_intervention(mock_baseline_df, mal_inv)

    # Reduction must be negative (emissions increased)
    assert res.absolute_reduction_kg < 0.0
    assert res.percentage_reduction < 0.0
    assert res.scenario_emissions_kg > res.baseline_emissions_kg

    # Cost per kg reduced must be negative indicating negative abatement return
    assert res.cost_per_kg_reduced < 0.0


# =====================================================================
# 6. Physical and Boundary Constraints Tests
# =====================================================================

def test_zero_baseline_emissions_handled_safely() -> None:
    """A zero-emission baseline must yield percentage_reduction=0.0 without ZeroDivisionError."""
    dates = [f"2025-{m:02d}-01" for m in range(1, 13)]
    zero_base_df = pd.DataFrame({
        "date": dates,
        "electricity_kwh": [0.0] * 12,
        "travel_km": [0.0] * 12,
        "waste_kg": [0.0] * 12,
        "procurement_inr": [0.0] * 12,
    })
    led = LEDLightingRetrofit(capital_cost=50000.0)
    res = evaluate_intervention(zero_base_df, led)

    assert res.baseline_emissions_kg == 0.0
    assert res.scenario_emissions_kg == 0.0
    assert res.absolute_reduction_kg == 0.0
    assert res.percentage_reduction == 0.0
    assert math.isinf(res.cost_per_kg_reduced)


def test_invalid_inputs_rejected(mock_baseline_df: pd.DataFrame) -> None:
    """Verify type and schema errors for invalid baseline or intervention inputs."""
    led = LEDLightingRetrofit()

    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        evaluate_intervention({"electricity_kwh": [1000]}, led)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="baseline_df cannot be empty"):
        evaluate_intervention(pd.DataFrame(), led)

    with pytest.raises(TypeError, match="Expected BaseIntervention instance"):
        evaluate_intervention(mock_baseline_df, "not_an_intervention")  # type: ignore[arg-type]


# =====================================================================
# 7. Phase 12 Uncertainty Propagation Tests
# =====================================================================

def test_phase_12_uncertainty_bounds_propagation(mock_baseline_df: pd.DataFrame) -> None:
    """Verify lower/upper activity bounds generate valid reduction sensitivity intervals."""
    lower_df, upper_df = generate_baseline_activity_bounds(mock_baseline_df)

    led = LEDLightingRetrofit(reduction_fraction=0.15)
    res = evaluate_intervention(
        mock_baseline_df,
        led,
        baseline_lower_df=lower_df,
        baseline_upper_df=upper_df,
    )

    assert res.absolute_reduction_lower_kg is not None
    assert res.absolute_reduction_upper_kg is not None
    assert res.absolute_reduction_lower_kg <= res.absolute_reduction_kg
    assert res.absolute_reduction_kg <= res.absolute_reduction_upper_kg


# =====================================================================
# 8. Summary DataFrame & Determinism Tests
# =====================================================================

def test_scenarios_to_dataframe_sorting() -> None:
    """Verify summary DataFrame is sorted by Marginal Abatement Cost (cost_per_kg_reduced) ascending."""
    cheap = LEDLightingRetrofit(capital_cost=10000.0)   # Very low MAC
    expensive = RooftopSolarInstallation(capital_cost=5000000.0) # Higher MAC

    dates = [f"2025-{m:02d}-01" for m in range(1, 13)]
    df = pd.DataFrame({
        "date": dates,
        "electricity_kwh": [10000.0] * 12,
        "travel_km": [5000.0] * 12,
        "waste_kg": [2000.0] * 12,
        "procurement_inr": [300000.0] * 12,
    })

    results = [
        evaluate_intervention(df, expensive),
        evaluate_intervention(df, cheap),
    ]

    summary_df = scenarios_to_dataframe(results)
    # cheap intervention must be first
    assert summary_df["intervention_id"].iloc[0] == cheap.id
    assert summary_df["intervention_id"].iloc[1] == expensive.id
    assert summary_df["cost_per_kg_reduced"].iloc[0] <= summary_df["cost_per_kg_reduced"].iloc[1]


def test_evaluation_determinism(mock_baseline_df: pd.DataFrame) -> None:
    """Verify running evaluate_intervention twice returns strictly identical results."""
    led = LEDLightingRetrofit()
    res1 = evaluate_intervention(mock_baseline_df, led)
    res2 = evaluate_intervention(mock_baseline_df, led)

    assert res1 == res2
    assert res1.to_dict() == res2.to_dict()
