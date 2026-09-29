"""
Campus Carbon Analytics - Forecasting Module
============================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Methodology Reference: docs/FORECASTING_METHOD.md
"""

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
    HoltWintersForecaster,
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
    # Metrics
    "ForecastMetrics",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "calculate_coverage_probability",
    "calculate_residuals",
    "calculate_improvement",
    "evaluate_forecast",
    # Models
    "BaseForecaster",
    "ForecastResult",
    "SeasonalNaiveForecaster",
    "HoltWintersForecaster",
    # Pipeline
    "prepare_monthly_emissions",
    "split_chronological_data",
    "run_forecast_experiment",
    "generate_future_projections",
    "export_forecast_results",
    "generate_forecast_visualizations",
    "run_pipeline",
]
