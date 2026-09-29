"""
Unit and Integration Tests for Campus Carbon Emissions Forecasting
===================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Tests:
1. Forecasting metrics (MAE, RMSE, MAPE, coverage, improvement, input validation).
2. Seasonal Naive baseline model (fitting, cycling predictions, 95% PI bounds).
3. Holt-Winters Exponential Smoothing model (additive trend & season, Monte Carlo 95% PI).
4. Chronological splitting (strict time-based split, no data leakage).
5. Full forecasting pipeline and artifacts generation.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

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
    ForecastResult,
    HoltWintersForecaster,
    SeasonalNaiveForecaster,
)
from src.forecasting.pipeline import (
    export_forecast_results,
    generate_future_projections,
    prepare_monthly_emissions,
    run_forecast_experiment,
    split_chronological_data,
)


# =====================================================================
# 1. METRICS TESTS
# =====================================================================

class TestForecastMetrics:
    """Test suite for forecasting evaluation metrics."""

    def test_calculate_mae_hand_computed(self) -> None:
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 330.0]
        # |100-110|=10, |200-190|=10, |300-330|=30. Mean = 50 / 3 = 16.6667
        mae = calculate_mae(actual, predicted)
        assert mae == pytest.approx(16.6667, abs=1e-3)

    def test_calculate_rmse_hand_computed(self) -> None:
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 16.0, 30.0]
        # (2^2 + (-4)^2 + 0^2) / 3 = (4 + 16 + 0) / 3 = 20 / 3 = 6.6667; sqrt = 2.5820
        rmse = calculate_rmse(actual, predicted)
        assert rmse == pytest.approx(2.5820, abs=1e-3)

    def test_calculate_mape_hand_computed(self) -> None:
        actual = [100.0, 200.0]
        predicted = [110.0, 180.0]
        # |10/100| = 0.10, |20/200| = 0.10 -> Mean = 10.0%
        mape = calculate_mape(actual, predicted)
        assert mape == pytest.approx(10.0, abs=1e-3)

    def test_calculate_mape_raises_on_zero(self) -> None:
        with pytest.raises(ValueError, match="contain zero"):
            calculate_mape([10.0, 0.0, 30.0], [10.0, 5.0, 30.0])

    def test_calculate_coverage_probability(self) -> None:
        actual = [10.0, 20.0, 30.0, 40.0]
        lower = [8.0, 15.0, 35.0, 38.0]
        upper = [12.0, 25.0, 45.0, 42.0]
        # Within: 10 in [8,12] (yes), 20 in [15,25] (yes), 30 in [35,45] (no), 40 in [38,42] (yes)
        # 3 out of 4 = 75.0%
        cov = calculate_coverage_probability(actual, lower, upper)
        assert cov == pytest.approx(75.0, abs=1e-2)

    def test_calculate_residuals(self) -> None:
        actual = [100.0, 200.0]
        predicted = [95.0, 210.0]
        res = calculate_residuals(actual, predicted)
        np.testing.assert_allclose(res, [5.0, -10.0])

    def test_calculate_improvement(self) -> None:
        # Baseline = 100, Model = 70 -> Improvement = 30%
        imp = calculate_improvement(100.0, 70.0)
        assert imp == pytest.approx(30.0)

        # Baseline = 100, Model = 120 -> Improvement = -20% (worse)
        worse_imp = calculate_improvement(100.0, 120.0)
        assert worse_imp == pytest.approx(-20.0)

    def test_calculate_improvement_raises_on_non_positive_baseline(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_improvement(0.0, 10.0)
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_improvement(-5.0, 10.0)

    def test_metrics_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Dimension mismatch"):
            calculate_mae([1, 2, 3], [1, 2])

    def test_metrics_empty_inputs(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_mae([], [])

    def test_metrics_nan_inf_rejection(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            calculate_mae([1.0, float("nan")], [1.0, 2.0])
        with pytest.raises(ValueError, match="Infinite"):
            calculate_mae([1.0, 2.0], [1.0, float("inf")])

    def test_evaluate_forecast_integration(self) -> None:
        actual = [100.0, 150.0, 200.0, 250.0]
        predicted = [105.0, 145.0, 195.0, 260.0]
        lower = [90.0, 130.0, 180.0, 240.0]
        upper = [120.0, 170.0, 220.0, 270.0]

        metrics = evaluate_forecast(actual, predicted, lower, upper)
        assert isinstance(metrics, ForecastMetrics)
        assert metrics.mae > 0.0
        assert metrics.rmse >= metrics.mae
        assert metrics.mape > 0.0
        assert metrics.coverage_pct == 100.0
        assert metrics.mean_residual is not None
        assert metrics.std_residual is not None


# =====================================================================
# 2. FORECAST RESULT & MODEL TESTS
# =====================================================================

class TestForecastResult:
    """Test suite for ForecastResult container."""

    def test_forecast_result_valid(self) -> None:
        pt = [10.0, 20.0]
        lb = [8.0, 18.0]
        ub = [12.0, 22.0]
        res = ForecastResult(point_forecast=pt, lower_bound=lb, upper_bound=ub)
        assert len(res.point_forecast) == 2
        df = res.to_dataframe()
        assert list(df.columns) == ["point_forecast", "lower_bound", "upper_bound"]

    def test_forecast_result_with_dates(self) -> None:
        res = ForecastResult([10.0, 20.0], [8.0, 18.0], [12.0, 22.0])
        df = res.to_dataframe(dates=["2025-01-01", "2025-02-01"])
        assert "date" in df.columns
        assert df["date"].iloc[0] == "2025-01-01"

    def test_forecast_result_invalid_dates_length(self) -> None:
        res = ForecastResult([10.0, 20.0], [8.0, 18.0], [12.0, 22.0])
        with pytest.raises(ValueError, match="Length of dates"):
            res.to_dataframe(dates=["2025-01-01"])

    def test_forecast_result_negative_lower_bound_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ForecastResult([10.0], [-1.0], [15.0])

    def test_forecast_result_inversion_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed upper bound"):
            ForecastResult([10.0], [15.0], [12.0])


class TestSeasonalNaiveForecaster:
    """Test suite for SeasonalNaiveForecaster."""

    def test_seasonal_period_validation(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            SeasonalNaiveForecaster(seasonal_period=0)

    def test_fit_insufficient_history(self) -> None:
        model = SeasonalNaiveForecaster(seasonal_period=12)
        with pytest.raises(ValueError, match="less than required minimum"):
            model.fit(np.arange(10, dtype=float))

    def test_predict_before_fit_raises(self) -> None:
        model = SeasonalNaiveForecaster(seasonal_period=12)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(12)

    def test_predict_values_cycle_correctly(self) -> None:
        # 24 months of synthetic seasonal pattern: month value = month % 12
        cycle = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
        train_series = np.concatenate([cycle, cycle + 5])  # 24 months

        model = SeasonalNaiveForecaster(seasonal_period=12).fit(train_series)
        fc = model.predict(steps=12)

        # In Seasonal Naive, next 12 months should match the last 12 months of training
        expected = train_series[-12:]
        np.testing.assert_allclose(fc.point_forecast, expected)

    def test_prediction_intervals_expand_and_sandwich(self) -> None:
        series = np.sin(np.linspace(0, 4 * np.pi, 24)) * 10 + 100
        model = SeasonalNaiveForecaster(seasonal_period=12).fit(series)
        fc = model.predict(steps=24, confidence_level=0.95)

        # Lower bound non-negative
        assert (fc.lower_bound >= 0.0).all()
        # Lower <= Point <= Upper
        assert (fc.lower_bound <= fc.point_forecast).all()
        assert (fc.point_forecast <= fc.upper_bound).all()

        # Interval width for year 2 (step 13..24) should be wider than year 1 (step 1..12)
        width_yr1 = fc.upper_bound[0] - fc.lower_bound[0]
        width_yr2 = fc.upper_bound[12] - fc.lower_bound[12]
        assert width_yr2 > width_yr1


class TestHoltWintersForecaster:
    """Test suite for HoltWintersForecaster."""

    def test_fit_insufficient_history(self) -> None:
        model = HoltWintersForecaster(seasonal_periods=12)
        # Needs at least 2 full cycles (24 points)
        with pytest.raises(ValueError, match="less than required minimum"):
            model.fit(np.arange(20, dtype=float))

    def test_predict_before_fit_raises(self) -> None:
        model = HoltWintersForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(12)

    def test_fit_and_predict_properties(self) -> None:
        # Create 24 months of trending seasonal data
        t = np.arange(24, dtype=float)
        series = 100.0 + 1.5 * t + 10.0 * np.sin(2 * np.pi * t / 12)

        model = HoltWintersForecaster(
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            random_state=42,
            simulation_repetitions=500,
        )
        model.fit(series)
        assert model.is_fitted
        assert len(model.fitted_values) == 24
        assert len(model.residuals) == 24
        assert "ExponentialSmoothing" in model.model_summary

        fc = model.predict(steps=12, confidence_level=0.95)
        assert isinstance(fc, ForecastResult)
        assert len(fc.point_forecast) == 12
        assert len(fc.lower_bound) == 12
        assert len(fc.upper_bound) == 12

        # Assert physical validity
        assert (fc.lower_bound >= 0.0).all()
        assert (fc.lower_bound <= fc.point_forecast).all()
        assert (fc.point_forecast <= fc.upper_bound).all()


# =====================================================================
# 3. PIPELINE & INTEGRATION TESTS
# =====================================================================

class TestForecastingPipeline:
    """Test suite for data loading, splitting, and pipeline execution."""

    def test_prepare_monthly_emissions(self) -> None:
        df = prepare_monthly_emissions("data/processed/campus_emissions.csv")
        assert len(df) == 36
        assert "date" in df.columns
        assert "total_emissions_kg" in df.columns
        assert "electricity_emissions_kg" in df.columns
        assert "travel_emissions_kg" in df.columns

        # Verify strict ascending chronology
        dates = pd.to_datetime(df["date"])
        assert dates.is_monotonic_increasing

    def test_split_chronological_data(self) -> None:
        df = prepare_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological_data(df, train_months=24, test_months=12)

        assert len(train_df) == 24
        assert len(test_df) == 12

        # Check date boundaries
        assert train_df["date"].iloc[0] == "2023-01-01"
        assert train_df["date"].iloc[-1] == "2024-12-01"
        assert test_df["date"].iloc[0] == "2025-01-01"
        assert test_df["date"].iloc[-1] == "2025-12-01"

        # Check no overlap
        train_set = set(train_df["date"])
        test_set = set(test_df["date"])
        assert train_set.isdisjoint(test_set)

    def test_split_insufficient_length_raises(self) -> None:
        short_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="MS").strftime("%Y-%m-%d"),
            "total_emissions_kg": range(10),
        })
        with pytest.raises(ValueError, match="insufficient"):
            split_chronological_data(short_df, train_months=24, test_months=12)

    def test_run_forecast_experiment_holt_winters_beats_baseline(self) -> None:
        """
        Verify that Holt-Winters outperforms Seasonal Naive baseline
        on our campus data per docs/FORECASTING_METHOD.md.
        """
        df = prepare_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological_data(df, train_months=24, test_months=12)

        summary_df, test_results, metrics = run_forecast_experiment(train_df, test_df)

        assert len(summary_df) == 6  # 3 targets * 2 models

        # Verify Holt-Winters achieves lower MAE than Seasonal Naive for all 3 targets
        for target in ["total_emissions_kg", "electricity_emissions_kg", "travel_emissions_kg"]:
            naive_mae = metrics[target]["Seasonal Naive"].mae
            hw_mae = metrics[target]["Holt-Winters"].mae
            assert hw_mae < naive_mae, (
                f"Holt-Winters MAE ({hw_mae:.2f}) did not beat Naive MAE ({naive_mae:.2f}) for {target}"
            )

            # Check coverage probability >= 80%
            hw_cov = metrics[target]["Holt-Winters"].coverage_pct
            assert hw_cov is not None and hw_cov >= 80.0

    def test_generate_future_projections(self) -> None:
        df = prepare_monthly_emissions("data/processed/campus_emissions.csv")
        projections_df, projections_dict = generate_future_projections(df, steps=12)

        assert len(projections_df) == 36  # 12 months * 3 targets
        assert "2026-01-01" in projections_df["date"].values
        assert "2026-12-01" in projections_df["date"].values

        # Bounds check
        assert (projections_df["lower_bound_95"] <= projections_df["point_forecast"]).all()
        assert (projections_df["point_forecast"] <= projections_df["upper_bound_95"]).all()

    def test_export_forecast_results(self, tmp_path: Path) -> None:
        df = prepare_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological_data(df, train_months=24, test_months=12)
        summary_df, test_results, _ = run_forecast_experiment(train_df, test_df)
        projections_df, _ = generate_future_projections(df, steps=12)

        sum_path, fc_path = export_forecast_results(
            monthly_df=df,
            summary_df=summary_df,
            test_results=test_results,
            projections_df=projections_df,
            output_dir=tmp_path,
        )

        assert sum_path.exists()
        assert fc_path.exists()

        loaded_fc = pd.read_csv(fc_path)
        assert len(loaded_fc) == 144  # 48 months * 3 targets
        assert set(loaded_fc["split_type"].unique()) == {"train", "test", "future_forecast"}
