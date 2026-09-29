"""
Campus Carbon Analytics - Forecasting Module
============================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 10: Forecasting Baseline Implementation
Phase 11: Primary Forecasting Model Implementation
Methodology References:
- docs/FORECASTING_METHOD.md
- docs/FORECASTING_BASELINE.md
- docs/FORECASTING.md
"""

from src.forecasting.baselines import (
    BaseBaseline,
    NaiveBaseline,
    SeasonalNaiveBaseline,
)
from src.forecasting.experiment import (
    generate_baseline_comparison_plot,
    load_and_aggregate_monthly_emissions,
    run_baseline_experiment,
    run_experiment,
    split_chronological,
)
from src.forecasting.holt_winters import (
    HoltWintersForecaster,
    HoltWintersPrediction,
    evaluate_primary_model,
    generate_primary_comparison_plot,
    run_primary_experiment,
)
from src.forecasting.metrics import (
    ForecastMetrics,
    calculate_coverage_probability,
    calculate_improvement,
    calculate_mae,
    calculate_mape,
    calculate_residuals,
    calculate_rmse,
    evaluate_forecast,
)
from src.forecasting.models import (
    BaseForecaster,
    ForecastResult,
    SeasonalNaiveForecaster,
)
from src.forecasting.pipeline import (
    export_forecast_results,
    generate_forecast_visualizations,
    generate_future_projections,
    prepare_monthly_emissions,
    run_forecast_experiment,
    run_pipeline,
    split_chronological_data,
)

__all__ = [
    # Baselines (Phase 10)
    "BaseBaseline",
    "NaiveBaseline",
    "SeasonalNaiveBaseline",
    # Experiment (Phase 10)
    "load_and_aggregate_monthly_emissions",
    "split_chronological",
    "run_baseline_experiment",
    "generate_baseline_comparison_plot",
    "run_experiment",
    # Primary Model (Phase 11)
    "HoltWintersForecaster",
    "HoltWintersPrediction",
    "evaluate_primary_model",
    "generate_primary_comparison_plot",
    "run_primary_experiment",
    # Metrics
    "ForecastMetrics",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "calculate_coverage_probability",
    "calculate_residuals",
    "calculate_improvement",
    "evaluate_forecast",
    # Pipeline & Containers
    "BaseForecaster",
    "ForecastResult",
    "SeasonalNaiveForecaster",
    "prepare_monthly_emissions",
    "split_chronological_data",
    "run_forecast_experiment",
    "generate_future_projections",
    "export_forecast_results",
    "generate_forecast_visualizations",
    "run_pipeline",
]
