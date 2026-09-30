"""
Campus Decarbonization Portfolio & Budget Optimization Engine
=============================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 15: Budget Optimization
Engine for constrained portfolio selection via Exhaustive Powerset Enumeration (2^N = 32).
Eliminates double-counting by sequentially applying sorted interventions through the
Phase 8 Carbon Accounting Engine, enforcing budget constraints, and resolving ties
using a strict 4-level deterministic tie-breaker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.carbon_accounting import (
    EmissionFactorRegistry,
    calculate_campus_emissions,
)
from src.scenarios.evaluation import (
    generate_baseline_activity_bounds,
    get_12m_baseline,
)
from src.scenarios.interventions import (
    BaseIntervention,
    get_default_interventions,
    sort_interventions,
)

logger = logging.getLogger("campus_carbon.scenarios.optimization")

DEFAULT_RAW_DATA_PATH = Path("data/raw/campus_activity.csv")


@dataclass(frozen=True)
class PortfolioResult:
    """
    Standardized result container for a decarbonization portfolio combination.

    Attributes
    ----------
    portfolio_id : str
        Deterministic binary mask identifier (e.g. 'P-10100', 'P-00000').
    selected_interventions : tuple[str, ...]
        Tuple of intervention IDs included in this portfolio (e.g. ('INT-001', 'INT-003')).
    num_interventions : int
        Number of projects in the portfolio.
    total_cost_inr : float
        Sum of capital costs across all selected interventions (₹ INR).
    remaining_budget_inr : float
        Unallocated budget: budget - total_cost_inr (₹ INR).
    is_feasible : bool
        Whether the portfolio satisfies the budget constraint (total_cost_inr <= budget).
    baseline_emissions_kg : float
        Business-as-usual total campus emissions (kgCO2e).
    portfolio_emissions_kg : float
        Total campus emissions after sequential application of all portfolio interventions (kgCO2e).
    total_absolute_reduction_kg : float
        Emissions avoided: baseline_emissions_kg - portfolio_emissions_kg (kgCO2e).
    percentage_reduction : float
        Relative reduction: (total_absolute_reduction_kg / baseline_emissions_kg) * 100 (%).
    cost_per_kg_reduced : float
        Portfolio Marginal Abatement Cost in INR/kgCO2e (float('inf') if reduction == 0).
    absolute_reduction_lower_kg : float | None, default None
        Sensitivity lower bound for absolute reduction derived from Phase 12 activity bounds.
    absolute_reduction_upper_kg : float | None, default None
        Sensitivity upper bound for absolute reduction derived from Phase 12 activity bounds.
    assumptions : dict[str, Any]
        Metadata recording project details, simulated cost flags, and component breakdowns.
    """

    portfolio_id: str
    selected_interventions: tuple[str, ...]
    num_interventions: int

    # Financials
    total_cost_inr: float
    remaining_budget_inr: float
    is_feasible: bool

    # Emissions Metrics
    baseline_emissions_kg: float
    portfolio_emissions_kg: float
    total_absolute_reduction_kg: float
    percentage_reduction: float
    cost_per_kg_reduced: float

    # Uncertainty Bounds (Phase 12 Integration)
    absolute_reduction_lower_kg: float | None = None
    absolute_reduction_upper_kg: float | None = None

    # Provenance
    assumptions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize portfolio result to dictionary."""
        return asdict(self)


def _generate_portfolio_id(
    selected_ids: set[str],
    canonical_order: Sequence[str],
) -> str:
    """Generate deterministic binary mask ID (e.g. 'P-10100')."""
    mask = "".join("1" if int_id in selected_ids else "0" for int_id in canonical_order)
    return f"P-{mask}"


def generate_all_portfolios(
    interventions: Sequence[BaseIntervention] | None = None,
) -> list[tuple[BaseIntervention, ...]]:
    """
    Generate the complete powerset of all 2^N intervention combinations.

    For the 5 canonical interventions, generates exactly 32 distinct portfolios,
    ordered deterministically from size 0 (empty portfolio) to size N (all interventions).

    Parameters
    ----------
    interventions : Sequence[BaseIntervention] | None, default None
        Candidate interventions. Defaults to get_default_interventions().

    Returns
    -------
    list[tuple[BaseIntervention, ...]]
        List of all 2^N portfolio combinations.
    """
    if interventions is None:
        interventions = get_default_interventions()

    # Deduplicate interventions by ID while preserving order
    seen_ids: set[str] = set()
    unique_invs: list[BaseIntervention] = []
    for inv in interventions:
        if not isinstance(inv, BaseIntervention):
            raise TypeError(f"Expected BaseIntervention instance, got {type(inv).__name__}")
        if inv.id not in seen_ids:
            seen_ids.add(inv.id)
            unique_invs.append(inv)

    portfolios: list[tuple[BaseIntervention, ...]] = []
    n = len(unique_invs)
    for r in range(n + 1):
        for combo in combinations(unique_invs, r):
            portfolios.append(combo)

    return portfolios


def evaluate_portfolio(
    baseline_df: pd.DataFrame,
    portfolio: Sequence[BaseIntervention],
    budget: float,
    canonical_order: Sequence[str] | None = None,
    registry: EmissionFactorRegistry | None = None,
    baseline_lower_df: pd.DataFrame | None = None,
    baseline_upper_df: pd.DataFrame | None = None,
) -> PortfolioResult:
    """
    Evaluate emissions, cost, feasibility, and abatement for a specific portfolio.

    Eliminates double-counting by sorting interventions (multiplicative first, additive second)
    and passing the activity DataFrame sequentially through each intervention's apply() method
    before computing emissions via the Phase 8 Carbon Accounting Engine.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        12-month baseline Activity DataFrame.
    portfolio : Sequence[BaseIntervention]
        Interventions included in this portfolio.
    budget : float
        Available capital budget in INR (₹).
    canonical_order : Sequence[str] | None, default None
        Canonical ID sequence for binary mask ID generation.
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry.
    baseline_lower_df : pd.DataFrame | None, default None
        Phase 12 lower activity bounds.
    baseline_upper_df : pd.DataFrame | None, default None
        Phase 12 upper activity bounds.

    Returns
    -------
    PortfolioResult
        Evaluated metrics for the portfolio.
    """
    if not isinstance(baseline_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for baseline_df, got {type(baseline_df).__name__}")
    if baseline_df.empty:
        raise ValueError("baseline_df cannot be empty.")
    if budget < 0.0:
        raise ValueError(f"Budget cannot be negative, got {budget}")
    if registry is None:
        registry = EmissionFactorRegistry.default()

    if canonical_order is None:
        canonical_order = ["INT-001", "INT-002", "INT-003", "INT-004", "INT-005"]

    # 1. Financial Evaluation
    total_cost = sum(float(inv.capital_cost) for inv in portfolio)
    is_feasible = bool(total_cost <= budget)
    remaining_budget = round(budget - total_cost, 2)

    # 2. Portfolio Identifiers
    selected_ids = tuple(inv.id for inv in portfolio)
    portfolio_id = _generate_portfolio_id(set(selected_ids), canonical_order)

    # 3. Baseline Emissions
    base_activity = baseline_df.copy()
    base_emissions_df = calculate_campus_emissions(base_activity, registry)
    baseline_emissions_kg = round(float(base_emissions_df["total_emissions_kg"].sum()), 2)

    # 4. Sequential Activity Transformation Pipeline (Prevents Double-Counting)
    if len(portfolio) == 0:
        portfolio_emissions_kg = baseline_emissions_kg
        total_reduction_kg = 0.0
        pct_reduction = 0.0
        cost_per_kg = float("inf")
        red_lower_kg = 0.0 if baseline_lower_df is not None else None
        red_upper_kg = 0.0 if baseline_upper_df is not None else None
    else:
        # Physical Stacking Rule: Multiplicative first (priority 1), Additive second (priority 2)
        sorted_portfolio = sort_interventions(portfolio)

        trans_df = base_activity.copy()
        for inv in sorted_portfolio:
            trans_df = inv.apply(trans_df)

        scen_emissions_df = calculate_campus_emissions(trans_df, registry)
        portfolio_emissions_kg = max(0.0, round(float(scen_emissions_df["total_emissions_kg"].sum()), 2))
        total_reduction_kg = round(baseline_emissions_kg - portfolio_emissions_kg, 2)

        if baseline_emissions_kg > 0.0:
            pct_reduction = round((total_reduction_kg / baseline_emissions_kg) * 100.0, 2)
        else:
            pct_reduction = 0.0

        if total_reduction_kg == 0.0:
            cost_per_kg = float("inf")
        else:
            cost_per_kg = round(total_cost / total_reduction_kg, 2)

        # 5. Optional Phase 12 Uncertainty Propagation
        red_lower_kg = None
        red_upper_kg = None
        if baseline_lower_df is not None and baseline_upper_df is not None:
            # Lower bound evaluation
            low_df = baseline_lower_df.copy()
            for inv in sorted_portfolio:
                low_df = inv.apply(low_df)
            em_base_low = float(calculate_campus_emissions(baseline_lower_df, registry)["total_emissions_kg"].sum())
            em_scen_low = float(calculate_campus_emissions(low_df, registry)["total_emissions_kg"].sum())
            delta_low = em_base_low - em_scen_low

            # Upper bound evaluation
            up_df = baseline_upper_df.copy()
            for inv in sorted_portfolio:
                up_df = inv.apply(up_df)
            em_base_up = float(calculate_campus_emissions(baseline_upper_df, registry)["total_emissions_kg"].sum())
            em_scen_up = float(calculate_campus_emissions(up_df, registry)["total_emissions_kg"].sum())
            delta_up = em_base_up - em_scen_up

            red_lower_kg = round(min(delta_low, delta_up), 2)
            red_upper_kg = round(max(delta_low, delta_up), 2)

    assumptions_meta = {
        "budget_inr": round(budget, 2),
        "intervention_names": [inv.name for inv in portfolio],
        "intervention_costs": {inv.id: inv.capital_cost for inv in portfolio},
        "is_simulated_assumption": all(inv.is_simulated_assumption for inv in portfolio) if portfolio else True,
    }

    return PortfolioResult(
        portfolio_id=portfolio_id,
        selected_interventions=selected_ids,
        num_interventions=len(portfolio),
        total_cost_inr=round(total_cost, 2),
        remaining_budget_inr=remaining_budget,
        is_feasible=is_feasible,
        baseline_emissions_kg=baseline_emissions_kg,
        portfolio_emissions_kg=portfolio_emissions_kg,
        total_absolute_reduction_kg=total_reduction_kg,
        percentage_reduction=pct_reduction,
        cost_per_kg_reduced=cost_per_kg,
        absolute_reduction_lower_kg=red_lower_kg,
        absolute_reduction_upper_kg=red_upper_kg,
        assumptions=assumptions_meta,
    )


def _sort_portfolio_key(p: PortfolioResult) -> tuple[float, float, int, str]:
    """
    Strict 4-Level Deterministic Tie-Breaker Key:
    1. Maximize total_absolute_reduction_kg (Descending -> negative float)
    2. Minimize total_cost_inr (Ascending -> positive float)
    3. Minimize num_interventions (Ascending -> positive int)
    4. Lexicographical portfolio_id (Ascending -> string)
    """
    return (
        -p.total_absolute_reduction_kg,
        p.total_cost_inr,
        p.num_interventions,
        p.portfolio_id,
    )


def optimize_portfolio(
    baseline_df: pd.DataFrame,
    budget: float,
    interventions: Sequence[BaseIntervention] | None = None,
    registry: EmissionFactorRegistry | None = None,
    baseline_lower_df: pd.DataFrame | None = None,
    baseline_upper_df: pd.DataFrame | None = None,
) -> tuple[PortfolioResult, list[PortfolioResult]]:
    """
    Execute exhaustive powerset optimization under budget constraint.

    Evaluates all 2^N combinations (32 portfolios for N=5), identifies all feasible
    portfolios conforming to the budget, and selects the optimal portfolio using
    the strict 4-level deterministic tie-breaker.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        12-month baseline Activity DataFrame.
    budget : float
        Capital expenditure budget in INR (₹).
    interventions : Sequence[BaseIntervention] | None, default None
        Candidate interventions. Defaults to get_default_interventions().
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry.
    baseline_lower_df : pd.DataFrame | None, default None
        Optional Phase 12 lower activity bounds.
    baseline_upper_df : pd.DataFrame | None, default None
        Optional Phase 12 upper activity bounds.

    Returns
    -------
    tuple[PortfolioResult, list[PortfolioResult]]
        Tuple of (optimal_portfolio, all_32_portfolios).

    Raises
    ------
    ValueError
        If budget < 0 or baseline_df is empty.
    """
    if budget < 0.0:
        raise ValueError(f"Budget cannot be negative, got {budget}")

    if interventions is None:
        interventions = get_default_interventions()

    canonical_order = [inv.id for inv in interventions]
    combos = generate_all_portfolios(interventions)

    all_portfolios: list[PortfolioResult] = []
    for combo in combos:
        res = evaluate_portfolio(
            baseline_df=baseline_df,
            portfolio=combo,
            budget=budget,
            canonical_order=canonical_order,
            registry=registry,
            baseline_lower_df=baseline_lower_df,
            baseline_upper_df=baseline_upper_df,
        )
        all_portfolios.append(res)

    # Filter to feasible portfolios (cost <= budget)
    feasible = [p for p in all_portfolios if p.is_feasible]
    if not feasible:
        # Theoretical fallback: empty portfolio is always feasible when budget >= 0
        empty_res = [p for p in all_portfolios if p.num_interventions == 0][0]
        return empty_res, all_portfolios

    # Rank feasible portfolios via 4-level tie-breaker
    feasible_sorted = sorted(feasible, key=_sort_portfolio_key)
    optimal_portfolio = feasible_sorted[0]

    return optimal_portfolio, all_portfolios


def portfolios_to_dataframe(portfolios: Sequence[PortfolioResult]) -> pd.DataFrame:
    """
    Convert a sequence of PortfolioResult objects into a formatted DataFrame.

    Parameters
    ----------
    portfolios : Sequence[PortfolioResult]
        Portfolio results.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame sorted by reduction descending, cost ascending.
    """
    records = []
    for p in portfolios:
        records.append({
            "portfolio_id": p.portfolio_id,
            "selected_interventions": ", ".join(p.selected_interventions) if p.selected_interventions else "None (BAU)",
            "num_interventions": p.num_interventions,
            "total_cost_inr": p.total_cost_inr,
            "remaining_budget_inr": p.remaining_budget_inr,
            "is_feasible": p.is_feasible,
            "baseline_emissions_kg": p.baseline_emissions_kg,
            "portfolio_emissions_kg": p.portfolio_emissions_kg,
            "total_absolute_reduction_kg": p.total_absolute_reduction_kg,
            "percentage_reduction": p.percentage_reduction,
            "cost_per_kg_reduced": p.cost_per_kg_reduced,
            "absolute_reduction_lower_kg": p.absolute_reduction_lower_kg,
            "absolute_reduction_upper_kg": p.absolute_reduction_upper_kg,
        })

    df = pd.DataFrame(records)
    return df.sort_values(
        by=["total_absolute_reduction_kg", "total_cost_inr"],
        ascending=[False, True],
    ).reset_index(drop=True)


def run_optimization_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute complete Phase 15 budget optimization workflow across standard institutional tiers.

    1. Loads 12-month forward baseline (2025).
    2. Derives Phase 12 activity bounds.
    3. Evaluates all 32 portfolios across standard budget levels:
       - ₹0 (BAU)
       - ₹5,00,000 (₹5 Lakhs - Small ESG grant)
       - ₹10,00,000 (₹10 Lakhs - Efficiency budget)
       - ₹25,00,000 (₹25 Lakhs - Medium retrofit)
       - ₹35,00,000 (₹35 Lakhs - Major decarbonization grant)
       - ₹60,00,000 (₹60 Lakhs - Unconstrained capital)
    4. Exports all 32 portfolios and optimal portfolio summaries to data/processed/.
    """
    print("=" * 85)
    print("PHASE 15: DECARBONIZATION PORTFOLIO BUDGET OPTIMIZATION (EXHAUSTIVE POWERSET)")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("=" * 85)

    baseline_df = get_12m_baseline(year=2025)
    lower_df, upper_df = generate_baseline_activity_bounds(baseline_df)
    interventions = get_default_interventions()

    print(f"Baseline Horizon : 12 Months ({baseline_df['date'].min()} to {baseline_df['date'].max()})")
    print(f"Interventions    : {len(interventions)} initiatives -> Powerset 2^5 = 32 combinations")

    # 1. Unconstrained 32-portfolio solution space
    unconstrained_budget = 10_000_000.0  # ₹1 Crore
    _, all_32 = optimize_portfolio(
        baseline_df=baseline_df,
        budget=unconstrained_budget,
        interventions=interventions,
        baseline_lower_df=lower_df,
        baseline_upper_df=upper_df,
    )
    all_32_df = portfolios_to_dataframe(all_32)

    out_32_csv = Path("data/processed/portfolio_optimization_all_32.csv")
    out_32_csv.parent.mkdir(parents=True, exist_ok=True)
    all_32_df.to_csv(out_32_csv, index=False)
    print(f"\n[Artifact] Exported all 32 portfolios: {out_32_csv.resolve()}")

    # 2. Evaluate standard institutional budget tiers
    budget_tiers = [
        (0.0, "Zero Budget (BAU)"),
        (500_000.0, "INR 5 Lakhs (Small ESG Grant)"),
        (1_000_000.0, "INR 10 Lakhs (Efficiency Budget)"),
        (2_500_000.0, "INR 25 Lakhs (Medium Retrofit)"),
        (3_500_000.0, "INR 35 Lakhs (Major Grant)"),
        (6_000_000.0, "INR 60 Lakhs (Unconstrained)"),
    ]

    tier_records = []
    print("\n--- OPTIMAL PORTFOLIO SELECTION BY BUDGET TIER ---")
    for b_val, b_label in budget_tiers:
        opt_p, _ = optimize_portfolio(
            baseline_df=baseline_df,
            budget=b_val,
            interventions=interventions,
            baseline_lower_df=lower_df,
            baseline_upper_df=upper_df,
        )
        tier_records.append({
            "budget_tier": b_label,
            "budget_inr": b_val,
            "optimal_portfolio_id": opt_p.portfolio_id,
            "selected_interventions": ", ".join(opt_p.selected_interventions) if opt_p.selected_interventions else "None (BAU)",
            "num_projects": opt_p.num_interventions,
            "total_cost_inr": opt_p.total_cost_inr,
            "remaining_budget_inr": opt_p.remaining_budget_inr,
            "total_reduction_kg": opt_p.total_absolute_reduction_kg,
            "percentage_reduction": opt_p.percentage_reduction,
            "cost_per_kg_reduced": opt_p.cost_per_kg_reduced,
            "reduction_lower_kg": opt_p.absolute_reduction_lower_kg,
            "reduction_upper_kg": opt_p.absolute_reduction_upper_kg,
        })

    tier_df = pd.DataFrame(tier_records)
    display_cols = [
        "budget_tier",
        "optimal_portfolio_id",
        "selected_interventions",
        "total_cost_inr",
        "remaining_budget_inr",
        "total_reduction_kg",
        "percentage_reduction",
        "cost_per_kg_reduced",
    ]
    print(tier_df[display_cols].to_string(index=False))

    out_tier_csv = Path("data/processed/optimal_portfolios_by_budget.csv")
    tier_df.to_csv(out_tier_csv, index=False)
    print(f"\n[Artifact] Exported optimal portfolios by budget: {out_tier_csv.resolve()}")
    print("=" * 85)

    return all_32_df, tier_df


if __name__ == "__main__":
    run_optimization_pipeline()
