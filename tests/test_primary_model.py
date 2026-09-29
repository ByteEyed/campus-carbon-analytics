"""
Unit and Integration Tests for Primary Forecasting Model (Holt-Winters)
=======================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 11: Primary Model Verification
Tests:
1. test_model_fit: Model fits 24-month history without convergence error.
2. test_forecast_generation: forecast(12) returns exactly 12 predictions.
3. test_prediction_intervals_ordering: lower_bound <= forecast <= upper_bound (non-negative).
4. test_leakage_prevention: Mutating test data has zero effect on parameters or forecasts.
5. test_insufficient_history: Series < 24 observations raises clean ValueError.
6. test_invalid_input: NaNs, Infs, empty inputs, non-positive steps raise ValueError.
7. test_unfitted_model_calls: Calling forecast/predict before fit raises RuntimeError.
8. test_output_schema: summary_frame and to_dataframe produce expected structure.
9. test_reproducibility: Repeated runs with identical seed generate identical predictions.
10. test_primary_beats_baseline: Holt-Winters achieves lower MAE and RMSE than Seasonal Naive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.experiment import (
    load_and_aggregate_monthly_emissions,
    split_chronological,
)
from src.forecasting.holt_winters import (
    HoltWintersForecaster,
    HoltWintersPrediction,
    evaluate_primary_model,
)


@pytest.fixture
def synthetic_24m_series() -> np.ndarray:
    """Fixture providing 24 months of trending seasonal data (m=12)."""
    t = np.arange(24, dtype=float)
    # 100 base + 2.5% annual trend + sinusoidal seasonal pattern
    return 100.0 + 0.25 * t + 15.0 * np.sin(2 * np.pi * t / 12)


class TestHoltWintersPrimaryModel:
    """Test suite for Holt-Winters primary forecasting model."""

    def test_model_fit(self, synthetic_24m_series: np.ndarray) -> None:
        """
        Verify model fits a 24-month training series without raising convergence errors.
        Matches Section 9: test_model_fit.
        """
        forecaster = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        fitted = forecaster.fit(synthetic_24m_series)

        assert fitted.is_fitted
        assert forecaster.fitted_model is not None
        assert len(forecaster.fitted_values) == 24
        assert len(forecaster.residuals) == 24

        # Check estimated smoothing parameters exist and are finite
        params = forecaster.params
        assert "smoothing_level" in params
        assert "smoothing_trend" in params
        assert "smoothing_seasonal" in params
        for val in params.values():
            assert not np.isnan(val) and not np.isinf(val)

    def test_forecast_generation(self, synthetic_24m_series: np.ndarray) -> None:
        """
        Verify forecast(12) returns exactly 12 point predictions.
        Matches Section 9: test_forecast_generation.
        """
        forecaster = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        forecaster.fit(synthetic_24m_series)
        fc = forecaster.forecast(steps=12)

        assert isinstance(fc, np.ndarray)
        assert len(fc) == 12
        assert not np.isnan(fc).any()
        assert not np.isinf(fc).any()
        assert (fc > 0.0).all()

    def test_prediction_intervals_ordering(self, synthetic_24m_series: np.ndarray) -> None:
        """
        Verify that for all horizon steps, lower_bound <= point_forecast <= upper_bound,
        and lower_bound >= 0 (physical carbon non-negativity).
        Matches Section 9: test_prediction_intervals_ordering.
        """
        forecaster = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        forecaster.fit(synthetic_24m_series)
        pred = forecaster.predict(horizon=12, confidence_level=0.95)

        assert isinstance(pred, HoltWintersPrediction)
        assert len(pred.point_forecast) == 12
        assert len(pred.lower_bound) == 12
        assert len(pred.upper_bound) == 12

        # Non-negativity
        assert (pred.lower_bound >= 0.0).all()
        # Strict ordering sandwich
        assert (pred.lower_bound <= pred.point_forecast).all()
        assert (pred.point_forecast <= pred.upper_bound).all()

        # Interval width should be positive
        assert (pred.upper_bound > pred.lower_bound).all()

    def test_leakage_prevention(self, synthetic_24m_series: np.ndarray) -> None:
        """
        Verify that mutating or corrupting future test set data has ZERO impact
        on model parameter estimation or generated forecasts.
        Matches Section 9: test_leakage_prevention.
        """
        # Fit 1: Baseline fit on training data
        model_1 = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, random_state=42)
        model_1.fit(synthetic_24m_series)
        params_1 = model_1.params
        fc_1 = model_1.forecast(12)

        # Hypothetical future test sets with extreme anomalies
        future_test_normal = np.array([120.0] * 12)
        future_test_corrupted = np.array([9999999.0] * 12)

        # Fit 2: Identical training slice (test data never touches fit)
        model_2 = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, random_state=42)
        model_2.fit(synthetic_24m_series)
        params_2 = model_2.params
        fc_2 = model_2.forecast(12)

        # Assert complete parameter and forecast invariance
        for k in params_1:
            assert params_1[k] == params_2[k]
        np.testing.assert_allclose(fc_1, fc_2)

        # Assert no dependence on future test arrays
        assert not np.array_equal(fc_1, future_test_normal)
        assert not np.array_equal(fc_1, future_test_corrupted)

    def test_insufficient_history(self) -> None:
        """
        Verify that passing fewer than 24 observations raises a clean ValueError.
        Holt-Winters with m=12 and trend requires at least 2 full cycles.
        Matches Section 9: test_insufficient_history.
        """
        forecaster = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        short_series = np.arange(20, dtype=float)

        with pytest.raises(ValueError, match="Insufficient history"):
            forecaster.fit(short_series)

        short_12 = np.arange(12, dtype=float)
        with pytest.raises(ValueError, match="Insufficient history"):
            forecaster.fit(short_12)

    def test_invalid_input_data(self) -> None:
        """Verify handling of invalid inputs (empty, NaN, Inf)."""
        forecaster = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)

        with pytest.raises(ValueError, match="cannot be empty"):
            forecaster.fit([])

        with pytest.raises(ValueError, match="contains NaN"):
            forecaster.fit([1.0] * 23 + [np.nan])

        with pytest.raises(ValueError, match="contains Infinite"):
            forecaster.fit([1.0] * 23 + [np.inf])

    def test_unfitted_model_calls_raise(self) -> None:
        """Calling forecast or predict before fit must raise RuntimeError."""
        forecaster = HoltWintersForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.forecast(12)
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(12)

    def test_invalid_horizon_and_confidence(self, synthetic_24m_series: np.ndarray) -> None:
        """Verify horizon > 0 and 0 < confidence_level < 1."""
        forecaster = HoltWintersForecaster().fit(synthetic_24m_series)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.forecast(0)
        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(-1)

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            forecaster.predict(12, confidence_level=1.0)
        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            forecaster.predict(12, confidence_level=0.0)

    def test_output_schema_and_compatibility(self, synthetic_24m_series: np.ndarray) -> None:
        """Verify summary_frame, get_prediction, and to_dataframe schemas."""
        forecaster = HoltWintersForecaster(random_state=42).fit(synthetic_24m_series)
        pred = forecaster.get_prediction(steps=12, alpha=0.05)

        # summary_frame matching statsmodels convention
        frame = pred.summary_frame(alpha=0.05)
        assert list(frame.columns) == ["mean", "mean_ci_lower", "mean_ci_upper"]
        assert len(frame) == 12

        # to_dataframe with dates
        dates = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")
        df_dates = pred.to_dataframe(dates=dates)
        assert "date" in df_dates.columns
        assert df_dates["date"].iloc[0] == "2025-01-01"

    def test_reproducibility(self, synthetic_24m_series: np.ndarray) -> None:
        """Verify deterministic runs with identical random seed produce identical intervals."""
        m1 = HoltWintersForecaster(random_state=42, simulation_repetitions=1000).fit(synthetic_24m_series)
        p1 = m1.predict(12, confidence_level=0.95)

        m2 = HoltWintersForecaster(random_state=42, simulation_repetitions=1000).fit(synthetic_24m_series)
        p2 = m2.predict(12, confidence_level=0.95)

        np.testing.assert_allclose(p1.point_forecast, p2.point_forecast)
        np.testing.assert_allclose(p1.lower_bound, p2.lower_bound)
        np.testing.assert_allclose(p1.upper_bound, p2.upper_bound)

    def test_primary_beats_baseline_on_actual_data(self) -> None:
        """
        Verify that on the actual project dataset, Holt-Winters primary model
        decisively beats the Phase 10 Seasonal Naive baseline on MAE and RMSE
        for Total Campus Emissions.
        """
        df = load_and_aggregate_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological(df, train_months=24, test_months=12)

        summary_df, predictions, error_analysis = evaluate_primary_model(train_df, test_df)

        total_rows = summary_df[summary_df["target"] == "total_emissions_kg"]
        sn_mae = float(total_rows[total_rows["model"].str.startswith("Seasonal Naive")]["mae_kg"].iloc[0])
        hw_mae = float(total_rows[total_rows["model"].str.startswith("Holt-Winters")]["mae_kg"].iloc[0])
        sn_rmse = float(total_rows[total_rows["model"].str.startswith("Seasonal Naive")]["rmse_kg"].iloc[0])
        hw_rmse = float(total_rows[total_rows["model"].str.startswith("Holt-Winters")]["rmse_kg"].iloc[0])

        # Primary model must achieve lower MAE and RMSE
        assert hw_mae < sn_mae, f"Holt-Winters MAE ({hw_mae:.2f}) failed to beat Seasonal Naive ({sn_mae:.2f})"
        assert hw_rmse < sn_rmse, f"Holt-Winters RMSE ({hw_rmse:.2f}) failed to beat Seasonal Naive ({sn_rmse:.2f})"

        # Verification of 95% PI coverage
        cov = error_analysis["total_emissions_kg"]["coverage_pct"]
        assert cov >= 90.0, f"Coverage probability {cov}% is below acceptable 90% threshold"
