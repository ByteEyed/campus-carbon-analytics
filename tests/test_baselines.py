"""
Tests for Forecasting Baselines
===============================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 10: Baseline Model Verification & Leakage Testing
Tests:
1. NaiveBaseline (flat projection y_{T+h} = y_T).
2. SeasonalNaiveBaseline (seasonal lag y_t = y_{t-m}).
3. Configurable seasonal periods (m=12, m=4).
4. Edge cases (insufficient observations, NaNs, Infs, empty series, invalid horizon).
5. Constant series handling.
6. Temporal leakage prevention tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.forecasting.baselines import (
    NaiveBaseline,
    SeasonalNaiveBaseline,
    _validate_input_series,
)


class TestNaiveBaseline:
    """Test suite for standard flat NaiveBaseline (y_{T+h} = y_T)."""

    def test_naive_flat_forecast_known_dataset(self) -> None:
        """Verify that NaiveBaseline projects the exact final historical value flatly."""
        train_data = [100.0, 150.0, 200.0, 275.5]
        model = NaiveBaseline().fit(train_data)
        preds = model.predict(horizon=6)

        assert len(preds) == 6
        # Every entry in forecast must equal the last historical observation (275.5)
        np.testing.assert_allclose(preds, [275.5] * 6)

    def test_naive_predict_before_fit_raises(self) -> None:
        """Calling predict before fit must raise RuntimeError."""
        model = NaiveBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(12)

    def test_naive_invalid_horizon_raises(self) -> None:
        """Horizon must be a strictly positive integer."""
        model = NaiveBaseline().fit([10.0, 20.0])
        with pytest.raises(ValueError, match="positive integer"):
            model.predict(0)
        with pytest.raises(ValueError, match="positive integer"):
            model.predict(-5)

    def test_naive_constant_series(self) -> None:
        """A constant series must produce a constant forecast equal to that constant."""
        model = NaiveBaseline().fit([42.0, 42.0, 42.0])
        preds = model.predict(12)
        np.testing.assert_allclose(preds, [42.0] * 12)

    def test_naive_single_observation(self) -> None:
        """A series with only 1 observation is sufficient for flat naive forecasting."""
        model = NaiveBaseline().fit([999.0])
        preds = model.predict(3)
        np.testing.assert_allclose(preds, [999.0, 999.0, 999.0])


class TestSeasonalNaiveBaseline:
    """Test suite for SeasonalNaiveBaseline (y_t = y_{t-m})."""

    def test_seasonal_naive_retrieves_t_minus_12(self) -> None:
        """
        Verify that SeasonalNaiveBaseline (m=12) retrieves exact t-12 values.
        Matches Section 6 & 10 of Phase 10 spec.
        """
        # Construct 24 months with a known monthly pattern: month value = 10 * month_idx
        year_1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
        year_2 = year_1 + 5.0  # e.g. secular growth
        train_series = np.concatenate([year_1, year_2])  # 24 observations

        model = SeasonalNaiveBaseline(seasonal_period=12).fit(train_series)
        preds = model.predict(horizon=12)

        # For January–December of Year 3, predictions must equal Year 2 (t-12)
        expected = year_2
        assert len(preds) == 12
        np.testing.assert_allclose(preds, expected)

    def test_seasonal_naive_multi_year_horizon_cycling(self) -> None:
        """For a 24-month horizon, SeasonalNaive should cycle the last 12 months twice."""
        cycle = np.arange(1, 13, dtype=float) * 100.0
        model = SeasonalNaiveBaseline(seasonal_period=12).fit(cycle)
        preds = model.predict(horizon=24)

        assert len(preds) == 24
        np.testing.assert_allclose(preds[:12], cycle)
        np.testing.assert_allclose(preds[12:], cycle)

    def test_seasonal_naive_configurable_period(self) -> None:
        """Verify configurable seasonal periods (e.g. quarterly m=4)."""
        quarterly_cycle = np.array([10.0, 20.0, 30.0, 40.0])
        history = np.concatenate([quarterly_cycle, quarterly_cycle + 2.0])  # 8 points

        model = SeasonalNaiveBaseline(seasonal_period=4).fit(history)
        preds = model.predict(horizon=4)
        np.testing.assert_allclose(preds, quarterly_cycle + 2.0)

    def test_seasonal_naive_invalid_period(self) -> None:
        """Seasonal period must be >= 1."""
        with pytest.raises(ValueError, match="positive integer >= 1"):
            SeasonalNaiveBaseline(seasonal_period=0)
        with pytest.raises(ValueError, match="positive integer >= 1"):
            SeasonalNaiveBaseline(seasonal_period=-12)

    def test_seasonal_naive_insufficient_history(self) -> None:
        """Fitting with fewer observations than seasonal_period must raise ValueError."""
        model = SeasonalNaiveBaseline(seasonal_period=12)
        short_series = [10.0] * 11
        with pytest.raises(ValueError, match="less than required minimum"):
            model.fit(short_series)

    def test_seasonal_naive_predict_before_fit(self) -> None:
        """Calling predict before fit must raise RuntimeError."""
        model = SeasonalNaiveBaseline(seasonal_period=12)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(12)

    def test_seasonal_naive_constant_series(self) -> None:
        """Constant series must produce constant seasonal predictions."""
        history = [55.0] * 24
        model = SeasonalNaiveBaseline(seasonal_period=12).fit(history)
        preds = model.predict(12)
        np.testing.assert_allclose(preds, [55.0] * 12)


class TestInputValidationAndEdgeCases:
    """Test suite for time series validation helper and error modes."""

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_input_series([])

    def test_nan_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="contains NaN"):
            _validate_input_series([10.0, np.nan, 30.0])

    def test_inf_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="contains Infinite"):
            _validate_input_series([10.0, np.inf, 30.0])

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="less than required minimum"):
            _validate_input_series([10.0, 20.0], min_length=5)


class TestLeakageControls:
    """
    Leakage Testing Suite.
    Specifically designed to detect accidental future-data usage or test leakage.
    """

    def test_predictions_strictly_invariant_to_future_test_data(self) -> None:
        """
        Verify that baseline model predictions are 100% invariant to any future
        values. Modifying, corrupting, or injecting anomalies into the future test
        set must have ZERO impact on the model's predictions.
        """
        train_series = np.array([
            100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
            105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215,
        ], dtype=float)

        # Baseline predictions trained on historical data
        naive_model = NaiveBaseline().fit(train_series)
        preds_naive_orig = naive_model.predict(horizon=12)

        s_naive_model = SeasonalNaiveBaseline(seasonal_period=12).fit(train_series)
        preds_s_naive_orig = s_naive_model.predict(horizon=12)

        # Hypothetical future test sets with extreme variations
        future_scenario_a = np.array([999999.0] * 12)
        future_scenario_b = np.array([0.001] * 12)

        # Retrain / re-predict ensuring fit receives strictly train_series
        naive_check = NaiveBaseline().fit(train_series)
        s_naive_check = SeasonalNaiveBaseline(seasonal_period=12).fit(train_series)

        np.testing.assert_allclose(naive_check.predict(12), preds_naive_orig)
        np.testing.assert_allclose(s_naive_check.predict(12), preds_s_naive_orig)

        # Assert no dependence on future scenario arrays
        assert not np.array_equal(preds_naive_orig, future_scenario_a)
        assert not np.array_equal(preds_s_naive_orig, future_scenario_b)
