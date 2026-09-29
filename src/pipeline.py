"""
Campus Carbon Analytics - End-to-End Pipeline Orchestrator
==========================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 16: End-to-End Integration
Orchestrates the entire campus decarbonization pipeline:
  Activity Data
  -> Validation (Phase 7)
  -> Carbon Accounting (Phase 8)
  -> Historical Analysis (Phase 8-9)
  -> Forecasting (Phase 10-11)
  -> Uncertainty Quantification (Phase 12)
  -> Scenario Evaluation (Phase 13-14)
  -> Budget Optimization (Phase 15)
  -> Integrated Results & Artifact Serialization

Reuses existing implementations without duplicating calculation logic,
enforces explicit contracts between stages, and guarantees complete reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Phase 7: Data Validation
from src.data_validation import (
    DataValidationError,
    REQUIRED_COLUMNS,
    ValidationReport,
    validate_activity_dataframe,
)

# Phase 8: Carbon Accounting
from src.carbon_accounting import (
    CarbonAccountingSummary,
    EmissionFactorRegistry,
    aggregate_emissions_by_category,
    aggregate_emissions_by_date,
    calculate_campus_emissions,
    calculate_category_contributions,
    calculate_emissions_per_student,
    calculate_total_emissions,
)

# Phase 11: Forecasting
from src.forecasting.holt_winters import HoltWintersForecaster
from src.forecasting.models import ForecastResult

# Phase 12: Uncertainty
from src.uncertainty.monte_carlo import (
    UncertaintyParameters,
    decompose_all_targets,
    run_monte_carlo_pipeline,
)

# Phase 13-14: Scenarios & Interventions
from src.scenarios.evaluation import (
    ScenarioEvaluationResult,
    evaluate_all_interventions,
    generate_baseline_activity_bounds,
    get_12m_baseline,
    scenarios_to_dataframe,
)
from src.scenarios.interventions import (
    BaseIntervention,
    get_default_interventions,
)

# Phase 15: Budget Optimization
from src.scenarios.optimization import (
    PortfolioResult,
    optimize_portfolio,
    portfolios_to_dataframe,
)

# Phase 16: Configuration
from src.pipeline_config import PipelineConfig

logger = logging.getLogger("campus_carbon.pipeline")

# Standard forecast target labels
TARGET_LABELS: dict[str, str] = {
    "total_emissions_kg": "Total Campus Emissions (kgCO2e)",
    "electricity_emissions_kg": "Electricity Emissions (kgCO2e)",
    "travel_emissions_kg": "Travel Emissions (kgCO2e)",
}


# =====================================================================
# Data Contracts & Intermediate Validators
# =====================================================================

class PipelineContractError(ValueError):
    """Exception raised when an intermediate pipeline data contract is violated."""


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate input activity data conforms strictly to DATA_DICTIONARY.md."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input activity data must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise PipelineContractError("Input activity DataFrame cannot be empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PipelineContractError(f"Activity DataFrame is missing required column(s): {missing}")


def validate_emissions_dataframe(emissions_df: pd.DataFrame) -> None:
    """Validate calculated emissions table conforms to Phase 8 accounting contract."""
    expected_cols = [
        "date",
        "electricity_emissions_kg",
        "travel_emissions_kg",
        "waste_emissions_kg",
        "procurement_emissions_kg",
        "total_emissions_kg",
        "total_emissions_mt",
    ]
    missing = [c for c in expected_cols if c not in emissions_df.columns]
    if missing:
        raise PipelineContractError(f"Emissions DataFrame is missing required column(s): {missing}")

    if (emissions_df["total_emissions_kg"] < 0).any():
        raise PipelineContractError("Emissions DataFrame contains negative emissions values.")

    if emissions_df["total_emissions_kg"].isna().any():
        raise PipelineContractError("Emissions DataFrame contains NaN values.")


def validate_time_series_continuity(
    monthly_df: pd.DataFrame,
    required_history_months: int = 36,
) -> None:
    """
    Validate chronological continuity of monthly aggregated emissions time series.

    Enforces:
    1. Minimum history requirement (e.g. 24m train + 12m test = 36 months).
    2. Continuous calendar sequence with zero missing monthly records.
    3. Chronological sorting.
    """
    if not isinstance(monthly_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for monthly time series, got {type(monthly_df).__name__}")
    if monthly_df.empty:
        raise PipelineContractError("Monthly emissions time series cannot be empty.")
    if "date" not in monthly_df.columns:
        raise PipelineContractError("Monthly emissions DataFrame missing 'date' column.")

    if len(monthly_df) < required_history_months:
        raise PipelineContractError(
            f"Insufficient historical timeline: dataset contains {len(monthly_df)} months, "
            f"but a minimum of {required_history_months} months is required for train/test splits."
        )

    dates = pd.to_datetime(monthly_df["date"])
    expected_range = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq="MS")
    if len(dates) != len(expected_range):
        raise PipelineContractError(
            f"Monthly time series has temporal gaps: expected {len(expected_range)} contiguous "
            f"months from {dates.iloc[0].strftime('%Y-%m')} to {dates.iloc[-1].strftime('%Y-%m')}, "
            f"found {len(dates)} records."
        )


def validate_forecast_projections(projections_df: pd.DataFrame) -> None:
    """Validate forecast projections table conforms to Phase 11 contract."""
    expected_cols = ["date", "target", "point_forecast", "lower_bound_95", "upper_bound_95"]
    missing = [c for c in expected_cols if c not in projections_df.columns]
    if missing:
        raise PipelineContractError(f"Forecast projections DataFrame missing required column(s): {missing}")

    # Enforce physical non-negativity and interval ordering: lower <= point <= upper
    for idx, row in projections_df.iterrows():
        pt = row["point_forecast"]
        lb = row["lower_bound_95"]
        ub = row["upper_bound_95"]
        if pt < 0:
            raise PipelineContractError(f"Forecast point projection at row {idx} is negative: {pt}")
        if lb > pt + 1e-4:
            raise PipelineContractError(f"Forecast lower bound ({lb}) exceeds point forecast ({pt}) at row {idx}")
        if pt > ub + 1e-4:
            raise PipelineContractError(f"Forecast point forecast ({pt}) exceeds upper bound ({ub}) at row {idx}")


def validate_portfolio_results(
    optimal: PortfolioResult,
    all_portfolios: list[PortfolioResult],
    budget: float,
) -> None:
    """Validate Phase 15 budget optimization results."""
    if not isinstance(optimal, PortfolioResult):
        raise TypeError(f"Expected PortfolioResult for optimal portfolio, got {type(optimal).__name__}")
    if len(all_portfolios) != 32:
        raise PipelineContractError(
            f"Expected exactly 32 portfolios in powerset evaluation, got {len(all_portfolios)}"
        )
    if not optimal.is_feasible:
        raise PipelineContractError(
            f"Optimal portfolio {optimal.portfolio_id} violates budget constraint "
            f"(cost {optimal.total_cost_inr} > budget {budget})"
        )


# =====================================================================
# Integrated Pipeline Result Container
# =====================================================================

@dataclass
class PipelineResult:
    """
    Unified result container for the complete end-to-end analytics execution.

    Attributes
    ----------
    config : PipelineConfig
        Configuration parameters applied during execution.
    validation_report : ValidationReport
        Phase 7 data validation report and rejection audit log.
    cleaned_activity_df : pd.DataFrame
        Phase 7 cleaned activity records.
    emissions_df : pd.DataFrame
        Phase 8 record-level calculated emissions.
    accounting_summary : CarbonAccountingSummary
        Phase 8 historical carbon accounting summary.
    historical_monthly_emissions : pd.DataFrame
        Phase 8 monthly aggregated campus emissions time series.
    forecast_projections : pd.DataFrame
        Phase 11 12-month forward projections with 95% prediction intervals.
    uncertainty_decomposition : pd.DataFrame
        Phase 12 One-at-a-Time variance decomposition summary.
    uncertainty_simulation : pd.DataFrame
        Phase 12 Monte Carlo simulation confidence interval trajectory.
    scenario_evaluations : list[ScenarioEvaluationResult]
        Phase 14 isolated intervention evaluations with MAC metrics.
    optimal_portfolio : PortfolioResult
        Phase 15 optimal decarbonization portfolio under budget constraint.
    all_portfolios : list[PortfolioResult]
        Phase 15 complete 32-portfolio solution space.
    """

    config: PipelineConfig
    validation_report: ValidationReport
    cleaned_activity_df: pd.DataFrame
    emissions_df: pd.DataFrame
    accounting_summary: CarbonAccountingSummary
    historical_monthly_emissions: pd.DataFrame
    forecast_projections: pd.DataFrame
    uncertainty_decomposition: pd.DataFrame
    uncertainty_simulation: pd.DataFrame
    scenario_evaluations: list[ScenarioEvaluationResult]
    optimal_portfolio: PortfolioResult
    all_portfolios: list[PortfolioResult]

    def export_artifacts(self, output_dir: Path | str | None = None) -> dict[str, Path]:
        """
        Serialize all pipeline dataframes, reports, and metadata to disk.

        Parameters
        ----------
        output_dir : Path | str | None, default None
            Target directory. Defaults to self.config.output_dir.

        Returns
        -------
        dict[str, Path]
            Dictionary of exported file paths.
        """
        out_dir = Path(output_dir) if output_dir is not None else self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting pipeline execution artifacts to: %s", out_dir.resolve())

        exported_files: dict[str, Path] = {}

        # 1. Cleaned Activity Data
        p_act = out_dir / "cleaned_activity.csv"
        self.cleaned_activity_df.to_csv(p_act, index=False)
        exported_files["cleaned_activity"] = p_act

        # 2. Facility Emissions
        p_em = out_dir / "campus_emissions.csv"
        self.emissions_df.to_csv(p_em, index=False)
        exported_files["campus_emissions"] = p_em

        # 3. Historical Monthly Emissions
        p_mo = out_dir / "historical_monthly_emissions.csv"
        self.historical_monthly_emissions.to_csv(p_mo, index=False)
        exported_files["historical_monthly_emissions"] = p_mo

        # 4. Forecast Projections
        p_fc = out_dir / "forecast_projections.csv"
        self.forecast_projections.to_csv(p_fc, index=False)
        exported_files["forecast_projections"] = p_fc

        # 5. Uncertainty Variance Decomposition
        p_ud = out_dir / "uncertainty_variance_decomposition.csv"
        self.uncertainty_decomposition.to_csv(p_ud, index=False)
        exported_files["uncertainty_variance_decomposition"] = p_ud

        # 6. Uncertainty Monte Carlo Simulation
        p_us = out_dir / "uncertainty_monte_carlo_simulation.csv"
        self.uncertainty_simulation.to_csv(p_us, index=False)
        exported_files["uncertainty_simulation"] = p_us

        # 7. Scenario Evaluations
        p_sc = out_dir / "scenario_evaluations.csv"
        scen_df = scenarios_to_dataframe(self.scenario_evaluations)
        scen_df.to_csv(p_sc, index=False)
        exported_files["scenario_evaluations"] = p_sc

        # 8. All 32 Portfolios & Optimal Portfolio
        p_p32 = out_dir / "portfolio_optimization_all_32.csv"
        p32_df = portfolios_to_dataframe(self.all_portfolios)
        p32_df.to_csv(p_p32, index=False)
        exported_files["portfolio_optimization_all_32"] = p_p32

        # 9. Pipeline Summary JSON
        p_json = out_dir / "pipeline_summary.json"
        summary_data = {
            "config": self.config.to_dict(),
            "validation": {
                "total_records": self.validation_report.total_records,
                "valid_records": self.validation_report.valid_records,
                "rejected_records": self.validation_report.rejected_records,
                "completeness_percentage": self.validation_report.completeness_percentage,
                "errors_by_type": self.validation_report.errors_by_type,
            },
            "carbon_accounting": {
                "total_emissions_kg": self.accounting_summary.total_emissions_kg,
                "total_emissions_mt": self.accounting_summary.total_emissions_mt,
                "emissions_per_student_kg": self.accounting_summary.emissions_per_student_kg,
                "category_contributions": self.accounting_summary.category_contributions,
            },
            "optimal_portfolio": self.optimal_portfolio.to_dict(),
        }
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        exported_files["pipeline_summary"] = p_json

        return exported_files


# =====================================================================
# Pipeline Orchestrator
# =====================================================================

def run_pipeline(
    config: PipelineConfig | None = None,
    raw_activity_df: pd.DataFrame | None = None,
    registry: EmissionFactorRegistry | None = None,
    interventions: Sequence[BaseIntervention] | None = None,
) -> PipelineResult:
    """
    Execute the integrated campus carbon analytics pipeline.

    Orchestrates:
      1. Load & Validate Inputs
      2. Activity Data Validation (Phase 7)
      3. Carbon Accounting & Aggregation (Phase 8)
      4. Primary Holt-Winters Time Series Forecasting (Phase 11)
      5. Monte Carlo Uncertainty Quantification & Variance Decomposition (Phase 12)
      6. Baseline Extraction & Activity Bounds Derivation (Phase 12-14)
      7. Isolated Scenario Evaluations (Phase 14)
      8. Constrained Portfolio Budget Optimization (Phase 15)
      9. Integrated Result Packaging & Artifact Export

    Parameters
    ----------
    config : PipelineConfig | None, default None
        Pipeline configuration settings. Defaults to standard PipelineConfig().
    raw_activity_df : pd.DataFrame | None, default None
        Optional in-memory raw activity dataframe. If None, loaded from config.activity_data_path.
    registry : EmissionFactorRegistry | None, default None
        Optional in-memory emission factor registry. If None, loaded from config.emission_factors_path.
    interventions : Sequence[BaseIntervention] | None, default None
        Optional list of decarbonization interventions. Defaults to get_default_interventions().

    Returns
    -------
    PipelineResult
        Consolidated execution container holding all intermediate and final outputs.

    Raises
    ------
    FileNotFoundError
        If required activity or emission factor files are absent from disk.
    PipelineContractError
        If data schemas or intermediate results fail integrity checks.
    ValueError
        If parameters, budgets, or numeric values violate physical boundaries.
    """
    if config is None:
        config = PipelineConfig()

    logger.info("Initializing campus carbon analytics pipeline with seed=%d", config.random_seed)

    # -------------------------------------------------------------
    # STAGE 1: Load Raw Inputs & Documented Factors
    # -------------------------------------------------------------
    if raw_activity_df is None:
        if not config.activity_data_path.exists():
            raise FileNotFoundError(
                f"Activity data file not found at: {config.activity_data_path.resolve()}"
            )
        logger.info("Loading raw activity data from: %s", config.activity_data_path)
        raw_activity_df = pd.read_csv(config.activity_data_path)

    validate_input_dataframe(raw_activity_df)

    if registry is None:
        if not config.emission_factors_path.exists():
            raise FileNotFoundError(
                f"Emission factors file not found at: {config.emission_factors_path.resolve()}"
            )
        logger.info("Loading emission factor registry from: %s", config.emission_factors_path)
        registry = EmissionFactorRegistry.from_csv(config.emission_factors_path)

    # -------------------------------------------------------------
    # STAGE 2: Validate Activity Data (Phase 7 Engine)
    # -------------------------------------------------------------
    logger.info("Stage 2: Validating activity data schema and constraints...")
    cleaned_df, val_report = validate_activity_dataframe(
        df=raw_activity_df,
        raise_on_missing_columns=config.fail_fast,
    )

    if config.fail_fast and val_report.rejected_records > 0:
        raise PipelineContractError(
            f"Activity data validation failed: {val_report.rejected_records} invalid records "
            f"detected ({val_report.completeness_percentage:.1f}% completeness)."
        )

    if cleaned_df.empty:
        raise PipelineContractError("No valid activity records remain after validation.")

    # -------------------------------------------------------------
    # STAGE 3: Carbon Accounting (Phase 8 Engine)
    # -------------------------------------------------------------
    logger.info("Stage 3: Computing campus carbon emissions...")
    emissions_df = calculate_campus_emissions(cleaned_df, registry)
    validate_emissions_dataframe(emissions_df)

    # Historical monthly aggregation
    monthly_emissions_df = aggregate_emissions_by_date(emissions_df)
    validate_time_series_continuity(
        monthly_emissions_df,
        required_history_months=config.train_months + config.forecast_horizon_months,
    )

    # Build accounting summary report
    tot_kg = calculate_total_emissions(emissions_df)
    tot_mt = round(tot_kg / 1000.0, 2)
    em_student = calculate_emissions_per_student(emissions_df)
    cat_contrib = calculate_category_contributions(emissions_df)
    by_cat = aggregate_emissions_by_category(emissions_df)

    accounting_summary = CarbonAccountingSummary(
        total_emissions_kg=tot_kg,
        total_emissions_mt=tot_mt,
        emissions_per_student_kg=em_student,
        category_contributions=cat_contrib,
        emissions_by_category=by_cat,
        emissions_by_date=monthly_emissions_df,
    )

    # -------------------------------------------------------------
    # STAGE 4: Time Series Forecasting (Phase 11 Primary Model)
    # -------------------------------------------------------------
    logger.info("Stage 4: Fitting primary Holt-Winters model and projecting 12-month horizon...")
    targets = ["total_emissions_kg", "electricity_emissions_kg", "travel_emissions_kg"]
    last_date = pd.to_datetime(monthly_emissions_df["date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=config.forecast_horizon_months,
        freq="MS",
    ).strftime("%Y-%m-%d")

    fc_rows: list[dict[str, Any]] = []
    for tgt in targets:
        y_series = monthly_emissions_df[tgt].to_numpy(dtype=float)
        hw = HoltWintersForecaster(
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            random_state=config.random_seed,
            simulation_repetitions=2000,
        ).fit(y_series)

        fc = hw.predict(steps=config.forecast_horizon_months, confidence_level=0.95)
        for d, pt, lb, ub in zip(future_dates, fc.point_forecast, fc.lower_bound, fc.upper_bound):
            fc_rows.append({
                "date": d,
                "target": tgt,
                "target_label": TARGET_LABELS.get(tgt, tgt),
                "point_forecast": round(float(pt), 4),
                "lower_bound_95": round(float(lb), 4),
                "upper_bound_95": round(float(ub), 4),
            })

    forecast_projections = pd.DataFrame(fc_rows)
    validate_forecast_projections(forecast_projections)

    # -------------------------------------------------------------
    # STAGE 5: Uncertainty Analysis & Monte Carlo Simulation (Phase 12)
    # -------------------------------------------------------------
    logger.info(
        "Stage 5: Decomposing uncertainty and running Monte Carlo simulation (N=%d)...",
        config.uncertainty_simulations,
    )
    uncertainty_params = UncertaintyParameters(
        seed=config.random_seed,
        n_simulations=config.uncertainty_simulations,
    )

    # 1. Variance Decomposition across targets
    uncertainty_decomp_df, _ = decompose_all_targets(
        df=cleaned_df,
        registry=registry,
        params=uncertainty_params,
        targets=targets,
        train_months=config.train_months,
        forecast_horizon=config.forecast_horizon_months,
    )

    # 2. Monte Carlo Simulation for confidence trajectory
    mc_simulation_df = run_monte_carlo_pipeline(
        df=cleaned_df,
        registry=registry,
        params=uncertainty_params,
        target="total_emissions_kg",
        train_months=config.train_months,
        forecast_horizon=config.forecast_horizon_months,
    )

    # -------------------------------------------------------------
    # STAGE 6: Baseline Horizon Extraction & Bounds (Phase 12 & 14)
    # -------------------------------------------------------------
    logger.info("Stage 6: Extracting %d baseline and propagating activity bounds...", config.baseline_year)
    baseline_df = get_12m_baseline(activity_df=cleaned_df, year=config.baseline_year)
    baseline_lower_df, baseline_upper_df = generate_baseline_activity_bounds(
        baseline_df=baseline_df,
        params=uncertainty_params,
    )

    # -------------------------------------------------------------
    # STAGE 7: Isolated Scenario Evaluation (Phase 14)
    # -------------------------------------------------------------
    logger.info("Stage 7: Evaluating decarbonization interventions in isolation...")
    if interventions is None:
        interventions = get_default_interventions()

    scenario_evaluations = evaluate_all_interventions(
        baseline_df=baseline_df,
        interventions=interventions,
        registry=registry,
        baseline_lower_df=baseline_lower_df,
        baseline_upper_df=baseline_upper_df,
    )

    # -------------------------------------------------------------
    # STAGE 8: Constrained Portfolio Budget Optimization (Phase 15)
    # -------------------------------------------------------------
    logger.info(
        "Stage 8: Optimizing portfolio powerset under budget constraint (INR %s)...",
        f"{config.optimization_budget_inr:,.2f}",
    )
    optimal_portfolio, all_portfolios = optimize_portfolio(
        baseline_df=baseline_df,
        budget=config.optimization_budget_inr,
        interventions=interventions,
        registry=registry,
        baseline_lower_df=baseline_lower_df,
        baseline_upper_df=baseline_upper_df,
    )
    validate_portfolio_results(optimal_portfolio, all_portfolios, config.optimization_budget_inr)

    # -------------------------------------------------------------
    # STAGE 9: Assemble Pipeline Result & Export
    # -------------------------------------------------------------
    result = PipelineResult(
        config=config,
        validation_report=val_report,
        cleaned_activity_df=cleaned_df,
        emissions_df=emissions_df,
        accounting_summary=accounting_summary,
        historical_monthly_emissions=monthly_emissions_df,
        forecast_projections=forecast_projections,
        uncertainty_decomposition=uncertainty_decomp_df,
        uncertainty_simulation=mc_simulation_df,
        scenario_evaluations=scenario_evaluations,
        optimal_portfolio=optimal_portfolio,
        all_portfolios=all_portfolios,
    )

    if config.export_artifacts_on_finish:
        result.export_artifacts()

    logger.info(
        "Pipeline execution successfully completed. Optimal Portfolio: %s "
        "(Reduction: %.2f kgCO2e / %.2f%%, Cost: INR %.2f)",
        optimal_portfolio.portfolio_id,
        optimal_portfolio.total_absolute_reduction_kg,
        optimal_portfolio.percentage_reduction,
        optimal_portfolio.total_cost_inr,
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Campus Carbon Analytics - Integrated End-to-End Pipeline (Phase 16)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Number of Monte Carlo simulations for uncertainty quantification (default: 1000).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Optimization capital expenditure budget in INR (default: 2,500,000.0).",
    )
    parser.add_argument(
        "--activity-data",
        type=str,
        default=None,
        help="Path to campus activity CSV (default: data/raw/campus_activity.csv).",
    )
    parser.add_argument(
        "--emission-factors",
        type=str,
        default=None,
        help="Path to emission factors CSV (default: data/emission_factors.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to export pipeline artifacts (default: data/processed/pipeline_run).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.config:
        cfg = PipelineConfig.from_json(args.config)
    else:
        cfg_kwargs: dict[str, Any] = {}
        if args.simulations is not None:
            cfg_kwargs["uncertainty_simulations"] = args.simulations
        if args.budget is not None:
            cfg_kwargs["optimization_budget_inr"] = args.budget
        if args.activity_data is not None:
            cfg_kwargs["activity_data_path"] = Path(args.activity_data)
        if args.emission_factors is not None:
            cfg_kwargs["emission_factors_path"] = Path(args.emission_factors)
        if args.output_dir is not None:
            cfg_kwargs["output_dir"] = Path(args.output_dir)
        if args.seed is not None:
            cfg_kwargs["random_seed"] = args.seed
        cfg = PipelineConfig(**cfg_kwargs)

    print("=" * 80)
    print("RUNNING CAMPUS CARBON ANALYTICS INTEGRATED PIPELINE (PHASE 16)")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("=" * 80)
    print(f"Configuration : Budget = INR {cfg.optimization_budget_inr:,.2f} | MC Simulations = {cfg.uncertainty_simulations} | Seed = {cfg.random_seed}")
    print(f"Activity Data : {cfg.activity_data_path}")
    print(f"Factors Data  : {cfg.emission_factors_path}")
    print(f"Output Dir    : {cfg.output_dir}")
    print("-" * 80)

    res = run_pipeline(cfg)

    print("\n" + res.accounting_summary.summary())
    print("\n--- OPTIMAL DECARBONIZATION PORTFOLIO ---")
    print(f"Portfolio ID        : {res.optimal_portfolio.portfolio_id}")
    print(f"Projects Included   : {', '.join(res.optimal_portfolio.selected_interventions)}")
    print(f"Total Cost          : INR {res.optimal_portfolio.total_cost_inr:,.2f}")
    print(f"Remaining Budget    : INR {res.optimal_portfolio.remaining_budget_inr:,.2f}")
    print(f"Avoided Carbon      : {res.optimal_portfolio.total_absolute_reduction_kg:,.2f} kgCO2e")
    print(f"Campus Reduction    : {res.optimal_portfolio.percentage_reduction:.2f}%")
    print(f"Marginal Abatement  : INR {res.optimal_portfolio.cost_per_kg_reduced:.2f}/kgCO2e")
    print(f"Artifacts Saved To  : {cfg.output_dir.resolve()}")
    print("=" * 80)
