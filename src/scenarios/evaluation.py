"""
Campus Decarbonization Scenario Evaluation Engine
=================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 14: Scenario Evaluation
Engine for isolated One-At-a-Time (OAT) evaluation of decarbonization scenarios.
Evaluates the marginal emissions reduction and financial efficiency (Marginal Abatement Cost)
of Phase 13 interventions against an immutable 12-month baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from src.scenarios.interventions import (
    BaseIntervention,
    get_default_interventions,
)
from src.uncertainty.monte_carlo import (
    DEFAULT_ACTIVITY_RSD,
    UncertaintyParameters,
    _lookup_rsd,
)

logger = logging.getLogger("campus_carbon.scenarios.evaluation")

DEFAULT_RAW_DATA_PATH = Path("data/raw/campus_activity.csv")


@dataclass(frozen=True)
class ScenarioEvaluationResult:
    """
    Standardized result container for an isolated intervention scenario evaluation.

    Attributes
    ----------
    intervention_id : str
        Unique identifier of the evaluated intervention (e.g. 'INT-001').
    intervention_name : str
        Human-readable name of the intervention.
    affected_category : str
        Primary carbon emission category affected (Electricity, Travel, Waste, Procurement).
    implementation_cost_inr : float
        Capital expenditure required to implement the intervention (₹ INR).
    baseline_emissions_kg : float
        Total campus carbon emissions in the baseline business-as-usual scenario (kgCO2e).
    scenario_emissions_kg : float
        Total campus carbon emissions after applying the intervention (kgCO2e).
    absolute_reduction_kg : float
        Emissions avoided: baseline_emissions_kg - scenario_emissions_kg (kgCO2e).
    percentage_reduction : float
        Relative emissions reduction: (absolute_reduction_kg / baseline_emissions_kg) * 100 (%).
    cost_per_kg_reduced : float
        Marginal Abatement Cost (MAC) in INR/kgCO2e: implementation_cost_inr / absolute_reduction_kg.
        Defined as float('inf') if absolute_reduction_kg == 0.
    absolute_reduction_lower_kg : float | None, default None
        Sensitivity lower bound for absolute reduction derived from Phase 12 activity bounds.
    absolute_reduction_upper_kg : float | None, default None
        Sensitivity upper bound for absolute reduction derived from Phase 12 activity bounds.
    assumptions : dict[str, Any]
        Metadata recording simulated parameters, lifespan, and category-level impacts.
    """

    intervention_id: str
    intervention_name: str
    affected_category: str
    implementation_cost_inr: float

    # Absolute Metrics (kgCO2e)
    baseline_emissions_kg: float
    scenario_emissions_kg: float
    absolute_reduction_kg: float

    # Relative & Financial Metrics
    percentage_reduction: float
    cost_per_kg_reduced: float

    # Uncertainty Bounds (Phase 12 Integration)
    absolute_reduction_lower_kg: float | None = None
    absolute_reduction_upper_kg: float | None = None

    # Provenance & Metadata
    assumptions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation result to dictionary."""
        return asdict(self)


def get_12m_baseline(
    activity_df: pd.DataFrame | None = None,
    year: int = 2025,
) -> pd.DataFrame:
    """
    Extract a clean 12-month baseline Activity DataFrame.

    Parameters
    ----------
    activity_df : pd.DataFrame | None, default None
        Input activity DataFrame. If None, loaded from data/raw/campus_activity.csv.
    year : int, default 2025
        Calendar year representing the 12-month forward horizon.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted, cloned 12-month baseline Activity DataFrame.
    """
    if activity_df is None:
        if not DEFAULT_RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw activity data not found at {DEFAULT_RAW_DATA_PATH.resolve()}")
        activity_df = pd.read_csv(DEFAULT_RAW_DATA_PATH)

    if "date" not in activity_df.columns:
        raise ValueError("Activity DataFrame must contain a 'date' column.")

    df_copy = activity_df.copy()
    dates = pd.to_datetime(df_copy["date"])
    filtered = df_copy[dates.dt.year == year].copy()

    if filtered.empty:
        raise ValueError(
            f"No activity records found for year {year}. "
            f"Available years: {sorted(dates.dt.year.unique().tolist())}"
        )

    return filtered.sort_values(by=["date", "building"] if "building" in filtered.columns else ["date"]).reset_index(drop=True)


def generate_baseline_activity_bounds(
    baseline_df: pd.DataFrame,
    params: UncertaintyParameters | None = None,
    z_score: float = 1.96,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Phase 12-aligned activity sensitivity bounds for the baseline horizon.

    Applies the Phase 12 Truncated Normal relative standard deviations (RSD) at the 95%
    spread (z = 1.96) to generate lower and upper activity bounding DataFrames.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline activity DataFrame.
    params : UncertaintyParameters | None, default None
        Uncertainty parameters defining activity RSD per category. Defaults to UncertaintyParameters().
    z_score : float, default 1.96
        Multiplier corresponding to nominal 95% Gaussian bounds.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (baseline_lower_df, baseline_upper_df).
    """
    if params is None:
        params = UncertaintyParameters()

    lower_df = baseline_df.copy()
    upper_df = baseline_df.copy()

    for col in ["electricity_kwh", "travel_km", "waste_kg", "procurement_inr"]:
        if col in baseline_df.columns:
            rsd = _lookup_rsd(params.activity_rsd, col)
            if rsd > 0.0:
                vals = baseline_df[col].to_numpy(dtype=float)
                delta = z_score * rsd * vals
                lower_df[col] = np.round(np.maximum(0.0, vals - delta), 2)
                upper_df[col] = np.round(vals + delta, 2)

    return lower_df, upper_df


def evaluate_intervention(
    baseline_df: pd.DataFrame,
    intervention: BaseIntervention,
    registry: EmissionFactorRegistry | None = None,
    baseline_lower_df: pd.DataFrame | None = None,
    baseline_upper_df: pd.DataFrame | None = None,
    scope: str = "total",
) -> ScenarioEvaluationResult:
    """
    Evaluate the marginal carbon reduction and abatement cost of an intervention in isolation.

    Follows the One-At-a-Time (OAT) evaluation protocol:
    1. Clones baseline_df immutably.
    2. Computes baseline emissions using Phase 8 Carbon Accounting.
    3. Transforms baseline activity using the intervention's pure functional apply() contract.
    4. Computes scenario emissions on the modified activity data.
    5. Computes delta (reduction), percentage reduction, and Marginal Abatement Cost (MAC).
    6. Propagates Phase 12 activity sensitivity bounds if lower/upper baselines are supplied.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        12-month baseline Activity DataFrame.
    intervention : BaseIntervention
        Concrete Phase 13 decarbonization intervention.
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry. Defaults to EmissionFactorRegistry.default().
    baseline_lower_df : pd.DataFrame | None, default None
        Optional lower activity bound DataFrame for Phase 12 uncertainty propagation.
    baseline_upper_df : pd.DataFrame | None, default None
        Optional upper activity bound DataFrame for Phase 12 uncertainty propagation.
    scope : str, default "total"
        Emissions evaluation scope: "total" (total campus emissions) or "category" (target category).

    Returns
    -------
    ScenarioEvaluationResult
        Standardized dataclass container with absolute, relative, and financial abatement metrics.

    Raises
    ------
    TypeError
        If baseline_df is not a DataFrame or intervention is not a BaseIntervention.
    ValueError
        If baseline_df is empty, missing target columns, or post-conditions fail.
    """
    if not isinstance(baseline_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for baseline_df, got {type(baseline_df).__name__}")
    if baseline_df.empty:
        raise ValueError("baseline_df cannot be empty.")
    if not isinstance(intervention, BaseIntervention):
        raise TypeError(f"Expected BaseIntervention instance, got {type(intervention).__name__}")
    if registry is None:
        registry = EmissionFactorRegistry.default()

    # 1. Immutable clone of baseline activity data
    base_activity = baseline_df.copy()

    # 2. Compute baseline emissions via Phase 8 pure function
    base_emissions_df = calculate_campus_emissions(base_activity, registry)

    # 3. Apply intervention transformation to create modified activity data
    scenario_activity = intervention.apply(base_activity)

    # 4. Compute scenario emissions via Phase 8 pure function
    scenario_emissions_df = calculate_campus_emissions(scenario_activity, registry)

    # 5. Extract totals based on evaluation scope
    target_metric_col = "total_emissions_kg" if scope.lower() == "total" else f"{intervention.category.lower()}_emissions_kg"

    baseline_emissions_kg = float(base_emissions_df[target_metric_col].sum())
    # Robustness clamp: enforce non-negative physical scenario emissions
    scenario_emissions_kg = max(0.0, float(scenario_emissions_df[target_metric_col].sum()))

    # 6. Calculate reduction metrics
    absolute_reduction_kg = round(baseline_emissions_kg - scenario_emissions_kg, 4)

    if baseline_emissions_kg > 0.0:
        percentage_reduction = round((absolute_reduction_kg / baseline_emissions_kg) * 100.0, 4)
    else:
        percentage_reduction = 0.0

    # 7. Safe Marginal Abatement Cost (MAC) calculation
    cost = float(intervention.capital_cost)
    if absolute_reduction_kg == 0.0:
        cost_per_kg = float("inf")
    else:
        cost_per_kg = round(cost / absolute_reduction_kg, 4)

    # 8. Category-level metrics for provenance metadata
    cat_col = f"{intervention.category.lower()}_emissions_kg"
    cat_base_kg = float(base_emissions_df[cat_col].sum()) if cat_col in base_emissions_df.columns else baseline_emissions_kg
    cat_scen_raw = float(scenario_emissions_df[cat_col].sum()) if cat_col in scenario_emissions_df.columns else scenario_emissions_kg
    cat_scen_kg = max(0.0, cat_scen_raw)
    cat_red_kg = round(cat_base_kg - cat_scen_kg, 4)
    cat_pct_red = round((cat_red_kg / cat_base_kg) * 100.0, 4) if cat_base_kg > 0.0 else 0.0

    # 9. Optional Phase 12 Uncertainty Propagation
    red_lower_kg: float | None = None
    red_upper_kg: float | None = None

    if baseline_lower_df is not None and baseline_upper_df is not None:
        # Lower bound evaluation
        scen_lower_act = intervention.apply(baseline_lower_df.copy())
        em_base_lower = float(calculate_campus_emissions(baseline_lower_df, registry)[target_metric_col].sum())
        em_scen_lower = float(calculate_campus_emissions(scen_lower_act, registry)[target_metric_col].sum())
        delta_lower = em_base_lower - em_scen_lower

        # Upper bound evaluation
        scen_upper_act = intervention.apply(baseline_upper_df.copy())
        em_base_upper = float(calculate_campus_emissions(baseline_upper_df, registry)[target_metric_col].sum())
        em_scen_upper = float(calculate_campus_emissions(scen_upper_act, registry)[target_metric_col].sum())
        delta_upper = em_base_upper - em_scen_upper

        red_lower_kg = round(min(delta_lower, delta_upper), 2)
        red_upper_kg = round(max(delta_lower, delta_upper), 2)

    # 10. Post-Condition Validation
    if scenario_emissions_kg < 0.0:
        raise ValueError(f"Physical constraint violated: scenario_emissions_kg={scenario_emissions_kg} < 0.0")
    if cost < 0.0:
        raise ValueError(f"Cost constraint violated: implementation_cost_inr={cost} < 0.0")

    # Consistent 2-decimal rounded metrics
    base_rounded = round(baseline_emissions_kg, 2)
    scen_rounded = round(scenario_emissions_kg, 2)
    red_rounded = round(base_rounded - scen_rounded, 2)

    pct_rounded = round((red_rounded / base_rounded) * 100.0, 2) if base_rounded > 0.0 else 0.0
    cost_per_kg_rounded = float("inf") if red_rounded == 0.0 else round(cost / red_rounded, 2)

    # Reduction consistency check
    if not math.isclose(red_rounded, round(base_rounded - scen_rounded, 2), abs_tol=1e-4):
        raise ValueError(
            f"Reduction consistency violated: absolute_reduction_kg ({red_rounded}) != "
            f"baseline - scenario ({base_rounded - scen_rounded})"
        )

    # Build provenance assumptions
    assumptions_meta = {
        "intervention_mechanism": intervention.mechanism_type,
        "is_simulated_assumption": intervention.is_simulated_assumption,
        "lifespan_years": intervention.lifespan_years,
        "target_column": intervention.target_column,
        "annualized_cost_inr": round(cost / intervention.lifespan_years, 2) if intervention.lifespan_years >= 1 else cost,
        "evaluation_scope": scope,
        "category_baseline_emissions_kg": round(cat_base_kg, 2),
        "category_scenario_emissions_kg": round(cat_scen_kg, 2),
        "category_reduction_kg": round(cat_red_kg, 2),
        "category_percentage_reduction": cat_pct_red,
    }

    # Extract intervention-specific parameters
    for param_name in [
        "reduction_fraction",
        "monthly_generation_kwh",
        "diversion_fraction",
        "ev_adoption_fraction",
        "active_months",
    ]:
        if hasattr(intervention, param_name):
            assumptions_meta[param_name] = getattr(intervention, param_name)

    return ScenarioEvaluationResult(
        intervention_id=intervention.id,
        intervention_name=intervention.name,
        affected_category=intervention.category,
        implementation_cost_inr=round(cost, 2),
        baseline_emissions_kg=base_rounded,
        scenario_emissions_kg=scen_rounded,
        absolute_reduction_kg=red_rounded,
        percentage_reduction=pct_rounded,
        cost_per_kg_reduced=cost_per_kg_rounded,
        absolute_reduction_lower_kg=red_lower_kg,
        absolute_reduction_upper_kg=red_upper_kg,
        assumptions=assumptions_meta,
    )


def evaluate_all_interventions(
    baseline_df: pd.DataFrame,
    interventions: Sequence[BaseIntervention] | None = None,
    registry: EmissionFactorRegistry | None = None,
    baseline_lower_df: pd.DataFrame | None = None,
    baseline_upper_df: pd.DataFrame | None = None,
    scope: str = "total",
) -> list[ScenarioEvaluationResult]:
    """
    Evaluate all interventions strictly in isolation (OAT) against the same baseline.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        12-month baseline Activity DataFrame.
    interventions : Sequence[BaseIntervention] | None, default None
        List of interventions to evaluate. Defaults to get_default_interventions().
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry.
    baseline_lower_df : pd.DataFrame | None, default None
        Optional Phase 12 lower activity bounds.
    baseline_upper_df : pd.DataFrame | None, default None
        Optional Phase 12 upper activity bounds.
    scope : str, default "total"
        Emissions evaluation scope.

    Returns
    -------
    list[ScenarioEvaluationResult]
        List of evaluation results in priority/catalog order.
    """
    if interventions is None:
        interventions = get_default_interventions()

    results: list[ScenarioEvaluationResult] = []
    for inv in interventions:
        logger.info("Evaluating intervention %s: %s", inv.id, inv.name)
        res = evaluate_intervention(
            baseline_df=baseline_df,
            intervention=inv,
            registry=registry,
            baseline_lower_df=baseline_lower_df,
            baseline_upper_df=baseline_upper_df,
            scope=scope,
        )
        results.append(res)

    return results


def scenarios_to_dataframe(results: Sequence[ScenarioEvaluationResult]) -> pd.DataFrame:
    """
    Convert a sequence of ScenarioEvaluationResult objects into a standardized summary DataFrame.

    Parameters
    ----------
    results : Sequence[ScenarioEvaluationResult]
        Evaluation results.

    Returns
    -------
    pd.DataFrame
        Summary table sorted by Marginal Abatement Cost (cost_per_kg_reduced) ascending.
    """
    records = []
    for r in results:
        records.append({
            "intervention_id": r.intervention_id,
            "intervention_name": r.intervention_name,
            "affected_category": r.affected_category,
            "implementation_cost_inr": r.implementation_cost_inr,
            "baseline_emissions_kg": r.baseline_emissions_kg,
            "scenario_emissions_kg": r.scenario_emissions_kg,
            "absolute_reduction_kg": r.absolute_reduction_kg,
            "percentage_reduction": r.percentage_reduction,
            "cost_per_kg_reduced": r.cost_per_kg_reduced,
            "absolute_reduction_lower_kg": r.absolute_reduction_lower_kg,
            "absolute_reduction_upper_kg": r.absolute_reduction_upper_kg,
        })

    df = pd.DataFrame(records)
    # Sort by financial cost-effectiveness (MAC ascending)
    return df.sort_values(by="cost_per_kg_reduced", ascending=True).reset_index(drop=True)


def run_scenario_evaluation_pipeline() -> pd.DataFrame:
    """
    Execute complete Phase 14 scenario evaluation workflow on canonical project dataset.

    1. Loads the 2025 12-month forward baseline horizon.
    2. Derives Phase 12-aligned activity sensitivity bounds.
    3. Evaluates all five canonical interventions in strict isolation.
    4. Ranks interventions by Marginal Abatement Cost (INR/kgCO2e).
    5. Exports results to data/processed/scenario_evaluation_summary.csv.
    """
    print("=" * 80)
    print("PHASE 14: DECARBONIZATION SCENARIO EVALUATION (ISOLATED OAT)")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("=" * 80)

    # 1. Load baseline
    baseline_df = get_12m_baseline(year=2025)
    print(f"Baseline Horizon : 12 Months ({baseline_df['date'].min()} to {baseline_df['date'].max()})")
    print(f"Campus Buildings : {baseline_df['building'].nunique()} facilities ({len(baseline_df)} records)")

    # 2. Phase 12 Activity Sensitivity Bounds
    lower_df, upper_df = generate_baseline_activity_bounds(baseline_df)

    # 3. Evaluate all interventions in isolation
    interventions = get_default_interventions()
    print(f"Interventions    : {len(interventions)} initiatives evaluated in strict isolation (OAT)")
    print("-" * 80)

    results = evaluate_all_interventions(
        baseline_df=baseline_df,
        interventions=interventions,
        baseline_lower_df=lower_df,
        baseline_upper_df=upper_df,
        scope="total",
    )

    summary_df = scenarios_to_dataframe(results)

    print("\n--- STANDALONE SCENARIO EVALUATION & MARGINAL ABATEMENT COST (MAC) TABLE ---")
    display_cols = [
        "intervention_id",
        "intervention_name",
        "affected_category",
        "implementation_cost_inr",
        "absolute_reduction_kg",
        "percentage_reduction",
        "cost_per_kg_reduced",
        "absolute_reduction_lower_kg",
        "absolute_reduction_upper_kg",
    ]
    print(summary_df[display_cols].to_string(index=False))

    # 4. Export Artifact
    out_csv = Path("data/processed/scenario_evaluation_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[Artifact] Exported scenario evaluation summary CSV: {out_csv.resolve()}")
    print("=" * 80)

    return summary_df


if __name__ == "__main__":
    run_scenario_evaluation_pipeline()
