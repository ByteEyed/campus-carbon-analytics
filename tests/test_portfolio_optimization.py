"""
Unit and Integration Tests for Portfolio & Budget Optimization Engine
=====================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 15: Decarbonization Portfolio Budget Optimization Test Suite
Validates:
1. Powerset generation: exactly 2^5 = 32 combinations generated deterministically.
2. Negative budget handling: ValueError raised for negative budget.
3. Zero budget handling: selects P-00000 with cost=0, reduction=0, MAC=inf.
4. Insufficient budget: budget < cheapest project (₹150,000) selects P-00000.
5. Unconstrained budget: budget >= total cost selects all 5 projects (P-11111).
6. Exact budget match handling and remaining budget precision.
7. Sequential interaction: verifies portfolio reduction is non-additive and eliminates double-counting
   (e.g., Portfolio(LED, AC) reduction < standalone LED reduction + standalone AC reduction).
8. Strict 4-level deterministic tie-breaker:
   Level 1: Maximize reduction
   Level 2: Minimize cost
   Level 3: Minimize number of projects
   Level 4: Lexicographical portfolio_id
9. Phase 12 activity sensitivity bounds propagation.
10. Immutability: baseline DataFrame is not mutated during powerset evaluation.
11. DataFrame serialization and column consistency.
12. Duplicate intervention deduplication.
13. Execution pipeline integration.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import EmissionFactorRegistry
from src.scenarios.evaluation import (
    evaluate_intervention,
    generate_baseline_activity_bounds,
    get_12m_baseline,
)
from src.scenarios.interventions import (
    ACOptimization,
    BaseIntervention,
    InterventionMechanism,
    LEDLightingRetrofit,
    LowCarbonTransport,
    RooftopSolarInstallation,
    WasteSegregation,
    get_default_interventions,
)
from src.scenarios.optimization import (
    PortfolioResult,
    _generate_portfolio_id,
    _sort_portfolio_key,
    evaluate_portfolio,
    generate_all_portfolios,
    optimize_portfolio,
    portfolios_to_dataframe,
    run_optimization_pipeline,
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
    """Canonical 2025 baseline from project dataset."""
    return get_12m_baseline(year=2025)


# =====================================================================
# 1. Powerset Generation & Deduplication Tests
# =====================================================================

def test_generate_all_portfolios_count() -> None:
    """Ensure exactly 2^5 = 32 combinations are generated for 5 interventions."""
    interventions = get_default_interventions()
    assert len(interventions) == 5

    combos = generate_all_portfolios(interventions)
    assert len(combos) == 32

    # Check size breakdown: 5C0=1, 5C1=5, 5C2=10, 5C3=10, 5C4=5, 5C5=1
    sizes = [len(c) for c in combos]
    assert sizes.count(0) == 1
    assert sizes.count(1) == 5
    assert sizes.count(2) == 10
    assert sizes.count(3) == 10
    assert sizes.count(4) == 5
    assert sizes.count(5) == 1


def test_generate_all_portfolios_deduplication() -> None:
    """If list contains duplicate interventions, deduplicate by ID."""
    interventions = get_default_interventions()
    dups = list(interventions) + [interventions[0], interventions[1]]
    assert len(dups) == 7

    combos = generate_all_portfolios(dups)
    assert len(combos) == 32


def test_generate_portfolio_id_mask() -> None:
    """Verify deterministic binary mask ID generation."""
    order = ["INT-001", "INT-002", "INT-003", "INT-004", "INT-005"]
    assert _generate_portfolio_id(set(), order) == "P-00000"
    assert _generate_portfolio_id({"INT-001"}, order) == "P-10000"
    assert _generate_portfolio_id({"INT-001", "INT-003"}, order) == "P-10100"
    assert _generate_portfolio_id({"INT-005"}, order) == "P-00001"
    assert _generate_portfolio_id(set(order), order) == "P-11111"


# =====================================================================
# 2. Input Validation & Budget Boundary Tests
# =====================================================================

def test_negative_budget_raises_value_error(mock_baseline_df: pd.DataFrame) -> None:
    """Negative budget must raise ValueError."""
    with pytest.raises(ValueError, match="Budget cannot be negative"):
        optimize_portfolio(mock_baseline_df, budget=-100.0)

    with pytest.raises(ValueError, match="Budget cannot be negative"):
        evaluate_portfolio(mock_baseline_df, [], budget=-1.0)


def test_empty_baseline_raises_value_error() -> None:
    """Empty baseline DataFrame must raise ValueError."""
    empty_df = pd.DataFrame(columns=["date", "building", "electricity_kwh"])
    with pytest.raises(ValueError, match="baseline_df cannot be empty"):
        evaluate_portfolio(empty_df, [], budget=1000.0)


def test_zero_budget_selects_empty_portfolio(mock_baseline_df: pd.DataFrame) -> None:
    """
    Zero budget allows only P-00000 (0 cost).
    Metrics must be: cost=0, reduction=0, percentage=0, MAC=inf.
    """
    opt_p, all_32 = optimize_portfolio(mock_baseline_df, budget=0.0)

    assert opt_p.portfolio_id == "P-00000"
    assert opt_p.num_interventions == 0
    assert opt_p.total_cost_inr == 0.0
    assert opt_p.remaining_budget_inr == 0.0
    assert opt_p.is_feasible is True
    assert opt_p.total_absolute_reduction_kg == 0.0
    assert opt_p.percentage_reduction == 0.0
    assert math.isinf(opt_p.cost_per_kg_reduced)


def test_insufficient_budget_below_cheapest(mock_baseline_df: pd.DataFrame) -> None:
    """
    Budget = ₹100,000.
    Cheapest intervention is Waste Segregation at ₹150,000.
    Optimal portfolio must be P-00000.
    """
    opt_p, _ = optimize_portfolio(mock_baseline_df, budget=100_000.0)
    assert opt_p.portfolio_id == "P-00000"
    assert opt_p.num_interventions == 0
    assert opt_p.total_cost_inr == 0.0
    assert opt_p.remaining_budget_inr == 100_000.0


def test_unconstrained_budget_selects_all_five(mock_baseline_df: pd.DataFrame) -> None:
    """
    Total cost of all 5 projects:
    INT-001: 500k + INT-002: 2,500k + INT-003: 300k + INT-004: 150k + INT-005: 1,800k = 5,250,000 INR.
    Budget = 6,000,000 INR.
    Optimal portfolio must be P-11111.
    """
    opt_p, all_32 = optimize_portfolio(mock_baseline_df, budget=6_000_000.0)

    assert opt_p.portfolio_id == "P-11111"
    assert opt_p.num_interventions == 5
    assert opt_p.total_cost_inr == 5_250_000.0
    assert opt_p.remaining_budget_inr == 750_000.0
    assert opt_p.is_feasible is True
    assert opt_p.total_absolute_reduction_kg > 0.0


def test_exact_budget_match(mock_baseline_df: pd.DataFrame) -> None:
    """
    Test exact budget matching:
    Budget = 500,000 INR.
    Feasible portfolios include:
    - P-00000 (cost 0)
    - P-00010 (INT-004, cost 150k)
    - P-00001 (INT-005, cost 300k)
    - P-00011 (INT-004 + INT-005, cost 450k)
    - P-10000 (INT-001, cost 500k -> remaining budget 0)
    - P-00100 (INT-003, cost 500k -> remaining budget 0)
    """
    opt_p, all_32 = optimize_portfolio(mock_baseline_df, budget=500_000.0)
    assert opt_p.is_feasible is True
    assert opt_p.total_cost_inr <= 500_000.0

    # Ensure remaining_budget is properly calculated
    assert opt_p.remaining_budget_inr == round(500_000.0 - opt_p.total_cost_inr, 2)


# =====================================================================
# 3. Double-Counting & Sequential Interaction Tests
# =====================================================================

def test_sequential_interaction_prevents_double_counting(canonical_12m_baseline: pd.DataFrame) -> None:
    """
    Critical Physics Test:
    When LED (12% reduction) and AC (10% reduction) are both implemented on electricity:
    Standalone LED: saves 12% of baseline.
    Standalone AC: saves 10% of baseline.
    If evaluated naively by summing standalone reductions: 12% + 10% = 22%.
    Under sequential compounding:
      1 - (1 - 0.12) * (1 - 0.10) = 1 - (0.88 * 0.90) = 1 - 0.792 = 20.8% < 22%.
    The portfolio reduction MUST be strictly less than the sum of standalone reductions.
    """
    led = LEDLightingRetrofit(reduction_fraction=0.12, capital_cost=500000.0)
    ac = ACOptimization(reduction_fraction=0.10, capital_cost=500000.0)

    # 1. Standalone Phase 14 evaluations
    res_led = evaluate_intervention(canonical_12m_baseline, led)
    res_ac = evaluate_intervention(canonical_12m_baseline, ac)
    sum_standalone_reduction = res_led.absolute_reduction_kg + res_ac.absolute_reduction_kg

    # 2. Portfolio Phase 15 evaluation
    res_combo = evaluate_portfolio(canonical_12m_baseline, [led, ac], budget=2_000_000.0)

    # Verify interaction cannibalism is captured
    assert res_combo.total_absolute_reduction_kg < sum_standalone_reduction
    cannibalism_kg = sum_standalone_reduction - res_combo.total_absolute_reduction_kg
    assert cannibalism_kg > 0.0

    # Check percentage relationship: ~20.8% of electricity emissions avoided
    pct_diff = (sum_standalone_reduction - res_combo.total_absolute_reduction_kg)
    assert pct_diff > 1000.0  # Significant difference on campus scale (>1 ton CO2e)


def test_all_five_portfolio_vs_sum_of_standalones(canonical_12m_baseline: pd.DataFrame) -> None:
    """
    Compare P-11111 total reduction to the sum of standalone Phase 14 reductions.
    Due to electricity efficiency compounding (LED + AC), total reduction must be less.
    """
    interventions = get_default_interventions()
    standalone_sum = 0.0
    for inv in interventions:
        r = evaluate_intervention(canonical_12m_baseline, inv)
        standalone_sum += r.absolute_reduction_kg

    p_11111 = evaluate_portfolio(canonical_12m_baseline, interventions, budget=10_000_000.0)
    assert p_11111.total_absolute_reduction_kg < standalone_sum


# =====================================================================
# 4. Strict 4-Level Deterministic Tie-Breaker Tests
# =====================================================================

def test_tie_breaker_level_1_higher_reduction_wins() -> None:
    """Level 1: Portfolio with higher reduction wins regardless of cost."""
    p_high = PortfolioResult(
        portfolio_id="P-10000",
        selected_interventions=("INT-001",),
        num_interventions=1,
        total_cost_inr=500_000.0,
        remaining_budget_inr=500_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,
        percentage_reduction=20.0,
        cost_per_kg_reduced=250.0,
    )
    p_low = PortfolioResult(
        portfolio_id="P-01000",
        selected_interventions=("INT-002",),
        num_interventions=1,
        total_cost_inr=200_000.0,  # Cheaper, but less reduction
        remaining_budget_inr=800_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8500.0,
        total_absolute_reduction_kg=1500.0,
        percentage_reduction=15.0,
        cost_per_kg_reduced=133.3,
    )

    ranked = sorted([p_low, p_high], key=_sort_portfolio_key)
    assert ranked[0].portfolio_id == "P-10000"


def test_tie_breaker_level_2_lower_cost_wins() -> None:
    """Level 2: Equal reduction -> lower total_cost_inr wins."""
    p_expensive = PortfolioResult(
        portfolio_id="P-10000",
        selected_interventions=("INT-001",),
        num_interventions=1,
        total_cost_inr=500_000.0,
        remaining_budget_inr=500_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,
        percentage_reduction=20.0,
        cost_per_kg_reduced=250.0,
    )
    p_cheap = PortfolioResult(
        portfolio_id="P-01000",
        selected_interventions=("INT-002",),
        num_interventions=1,
        total_cost_inr=400_000.0,  # Lower cost
        remaining_budget_inr=600_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,  # Equal reduction
        percentage_reduction=20.0,
        cost_per_kg_reduced=200.0,
    )

    ranked = sorted([p_expensive, p_cheap], key=_sort_portfolio_key)
    assert ranked[0].portfolio_id == "P-01000"


def test_tie_breaker_level_3_fewer_projects_wins() -> None:
    """Level 3: Equal reduction, equal cost -> fewer projects wins."""
    p_single = PortfolioResult(
        portfolio_id="P-10000",
        selected_interventions=("INT-001",),
        num_interventions=1,  # 1 project
        total_cost_inr=500_000.0,
        remaining_budget_inr=500_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,
        percentage_reduction=20.0,
        cost_per_kg_reduced=250.0,
    )
    p_double = PortfolioResult(
        portfolio_id="P-01100",
        selected_interventions=("INT-002", "INT-003"),
        num_interventions=2,  # 2 projects
        total_cost_inr=500_000.0,  # Equal cost
        remaining_budget_inr=500_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,  # Equal reduction
        percentage_reduction=20.0,
        cost_per_kg_reduced=250.0,
    )

    ranked = sorted([p_double, p_single], key=_sort_portfolio_key)
    assert ranked[0].portfolio_id == "P-10000"


def test_tie_breaker_level_4_lexicographical_id_wins() -> None:
    """Level 4: Equal reduction, equal cost, equal count -> lexicographical portfolio_id wins."""
    p_a = PortfolioResult(
        portfolio_id="P-01000",
        selected_interventions=("INT-002",),
        num_interventions=1,
        total_cost_inr=500_000.0,
        remaining_budget_inr=500_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,
        percentage_reduction=20.0,
        cost_per_kg_reduced=250.0,
    )
    p_b = PortfolioResult(
        portfolio_id="P-10000",
        selected_interventions=("INT-001",),
        num_interventions=1,
        total_cost_inr=500_000.0,
        remaining_budget_inr=500_000.0,
        is_feasible=True,
        baseline_emissions_kg=10000.0,
        portfolio_emissions_kg=8000.0,
        total_absolute_reduction_kg=2000.0,
        percentage_reduction=20.0,
        cost_per_kg_reduced=250.0,
    )

    ranked = sorted([p_b, p_a], key=_sort_portfolio_key)
    # 'P-01000' < 'P-10000'
    assert ranked[0].portfolio_id == "P-01000"


# =====================================================================
# 5. Immutability & Sensitivity Bounds Tests
# =====================================================================

def test_baseline_immutability(mock_baseline_df: pd.DataFrame) -> None:
    """Verify mock_baseline_df is unchanged after running optimize_portfolio."""
    orig = mock_baseline_df.copy(deep=True)
    _, _ = optimize_portfolio(mock_baseline_df, budget=1_000_000.0)
    pd.testing.assert_frame_equal(mock_baseline_df, orig)


def test_phase12_activity_bounds_propagation(mock_baseline_df: pd.DataFrame) -> None:
    """Verify uncertainty bounds propagate correctly into PortfolioResult."""
    lower_df, upper_df = generate_baseline_activity_bounds(mock_baseline_df)
    opt_p, all_32 = optimize_portfolio(
        baseline_df=mock_baseline_df,
        budget=2_000_000.0,
        baseline_lower_df=lower_df,
        baseline_upper_df=upper_df,
    )

    for p in all_32:
        if p.num_interventions == 0:
            assert p.absolute_reduction_lower_kg == 0.0
            assert p.absolute_reduction_upper_kg == 0.0
        else:
            assert p.absolute_reduction_lower_kg is not None
            assert p.absolute_reduction_upper_kg is not None
            assert p.absolute_reduction_lower_kg <= p.absolute_reduction_upper_kg


# =====================================================================
# 6. DataFrame Conversion & Pipeline Tests
# =====================================================================

def test_portfolios_to_dataframe(mock_baseline_df: pd.DataFrame) -> None:
    """Verify structure and types of portfolios_to_dataframe."""
    _, all_32 = optimize_portfolio(mock_baseline_df, budget=1_000_000.0)
    df = portfolios_to_dataframe(all_32)

    assert len(df) == 32
    expected_cols = [
        "portfolio_id",
        "selected_interventions",
        "num_interventions",
        "total_cost_inr",
        "remaining_budget_inr",
        "is_feasible",
        "baseline_emissions_kg",
        "portfolio_emissions_kg",
        "total_absolute_reduction_kg",
        "percentage_reduction",
        "cost_per_kg_reduced",
        "absolute_reduction_lower_kg",
        "absolute_reduction_upper_kg",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify sorting: total_absolute_reduction_kg descending
    assert df["total_absolute_reduction_kg"].is_monotonic_decreasing


def test_run_optimization_pipeline_artifacts() -> None:
    """Run full optimization pipeline and ensure exported CSV files exist and are valid."""
    all_df, tier_df = run_optimization_pipeline()

    assert len(all_df) == 32
    assert len(tier_df) == 6

    f1 = Path("data/processed/portfolio_optimization_all_32.csv")
    f2 = Path("data/processed/optimal_portfolios_by_budget.csv")

    assert f1.exists()
    assert f2.exists()

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    assert len(df1) == 32
    assert len(df2) == 6
