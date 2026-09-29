"""
Campus Carbon Analytics - Scenario & Intervention Library Module
================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 13: Intervention Scenario Library
Defines decarbonization interventions and their functional activity calculation contracts.

Phase 14: Scenario Evaluation
Provides isolated One-At-a-Time (OAT) evaluation, Marginal Abatement Cost (MAC) calculations,
and Phase 12 activity sensitivity bounds propagation.
"""

from src.scenarios.evaluation import (
    ScenarioEvaluationResult,
    evaluate_all_interventions,
    evaluate_intervention,
    generate_baseline_activity_bounds,
    get_12m_baseline,
    run_scenario_evaluation_pipeline,
    scenarios_to_dataframe,
)
from src.scenarios.interventions import (
    INTERVENTION_CATALOG,
    ACOptimization,
    BaseIntervention,
    HVACOptimization,
    InterventionMechanism,
    LEDLighting,
    LEDLightingRetrofit,
    LowCarbonTransport,
    RooftopSolarInstallation,
    SolarInstallation,
    SustainableEVTransport,
    WasteSegregation,
    WasteSegregationComposting,
    get_default_interventions,
    sort_interventions,
)
from src.scenarios.optimization import (
    PortfolioResult,
    evaluate_portfolio,
    generate_all_portfolios,
    optimize_portfolio,
    portfolios_to_dataframe,
    run_optimization_pipeline,
)

__all__ = [
    "ACOptimization",
    "BaseIntervention",
    "HVACOptimization",
    "INTERVENTION_CATALOG",
    "InterventionMechanism",
    "LEDLighting",
    "LEDLightingRetrofit",
    "LowCarbonTransport",
    "PortfolioResult",
    "RooftopSolarInstallation",
    "ScenarioEvaluationResult",
    "SolarInstallation",
    "SustainableEVTransport",
    "WasteSegregation",
    "WasteSegregationComposting",
    "evaluate_all_interventions",
    "evaluate_intervention",
    "evaluate_portfolio",
    "generate_all_portfolios",
    "generate_baseline_activity_bounds",
    "get_12m_baseline",
    "get_default_interventions",
    "optimize_portfolio",
    "portfolios_to_dataframe",
    "run_optimization_pipeline",
    "run_scenario_evaluation_pipeline",
    "scenarios_to_dataframe",
    "sort_interventions",
]

