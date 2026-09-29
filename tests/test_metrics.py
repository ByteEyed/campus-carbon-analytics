"""
Tests for Forecasting Metrics
=============================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 10: Evaluation Metrics Verification
Tests:
1. MAE, RMSE, MAPE against hand-calculated numerical examples.
2. Division-by-zero protection in MAPE.
3. Constant series error metrics (zero error).
4. Relative percentage improvement calculator.
5. Error handling for NaNs, Infs, dimension mismatches, and empty series.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.forecasting.metrics import (
    ForecastMetrics,
    calculate_improvement,
    calculate_mae,
    calculate_mape,
    calculate_residuals,
    calculate_rmse,
    evaluate_forecast,
)


class TestMetricsCalculation:
    """Test suite for numerical precision of metrics against hand calculations."""

    def test_calculate_mae_hand_calculated(self) -> None:
        """
        Hand-calculated Oracle:
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 330.0]
        abs_errors = [10.0, 10.0, 30.0]
        MAE = 50.0 / 3 = 16.6667
        """
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 330.0]
        mae = calculate_mae(actual, predicted)
        assert mae == pytest.approx(16.6667, abs=1e-3)

    def test_calculate_rmse_hand_calculated(self) -> None:
        """
        Hand-calculated Oracle:
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 16.0, 30.0]
        squared_errors = [4.0, 16.0, 0.0]
        Mean squared error = 20.0 / 3 = 6.6667
        RMSE = sqrt(6.6667) = 2.5820
        """
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 16.0, 30.0]
        rmse = calculate_rmse(actual, predicted)
        assert rmse == pytest.approx(2.5820, abs=1e-3)

    def test_calculate_mape_hand_calculated(self) -> None:
        """
        Hand-calculated Oracle:
        actual = [100.0, 200.0]
        predicted = [110.0, 180.0]
        pct_errors = [10/100, 20/200] = [0.10, 0.10]
        MAPE = 10.0%
        """
        actual = [100.0, 200.0]
        predicted = [110.0, 180.0]
        mape = calculate_mape(actual, predicted)
        assert mape == pytest.approx(10.0, abs=1e-3)

    def test_calculate_mape_division_by_zero_guard(self) -> None:
        """MAPE must strictly raise ValueError when actual contains zero."""
        with pytest.raises(ValueError, match="contain zero"):
            calculate_mape([10.0, 0.0, 30.0], [10.0, 5.0, 30.0])

    def test_perfect_prediction_zero_error(self) -> None:
        """When actual equals predicted, all errors must be exactly 0.0."""
        actual = [123.45, 678.90, 456.78]
        predicted = [123.45, 678.90, 456.78]

        assert calculate_mae(actual, predicted) == 0.0
        assert calculate_rmse(actual, predicted) == 0.0
        assert calculate_mape(actual, predicted) == 0.0

    def test_calculate_residuals(self) -> None:
        """Verify residual formula: actual - predicted."""
        actual = [150.0, 250.0]
        predicted = [140.0, 265.0]
        res = calculate_residuals(actual, predicted)
        np.testing.assert_allclose(res, [10.0, -15.0])

    def test_calculate_improvement(self) -> None:
        """
        Formula: ((baseline - model) / baseline) * 100
        Baseline MAE = 1000, Model MAE = 200 -> 80% improvement.
        """
        assert calculate_improvement(1000.0, 200.0) == pytest.approx(80.0)
        # Worse model: Baseline = 100, Model = 150 -> -50% improvement
        assert calculate_improvement(100.0, 150.0) == pytest.approx(-50.0)

    def test_calculate_improvement_non_positive_baseline_raises(self) -> None:
        """Baseline error must be strictly positive."""
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_improvement(0.0, 50.0)
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_improvement(-10.0, 50.0)


class TestMetricsErrorHandling:
    """Test suite for defensive input validation in metrics."""

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Dimension mismatch"):
            calculate_mae([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_arrays_raise(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_mae([], [])

    def test_nan_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="contain NaN"):
            calculate_mae([1.0, np.nan], [1.0, 2.0])
        with pytest.raises(ValueError, match="contain NaN"):
            calculate_mae([1.0, 2.0], [1.0, np.nan])

    def test_inf_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="contain Infinite"):
            calculate_mae([1.0, np.inf], [1.0, 2.0])

    def test_evaluate_forecast_integration(self) -> None:
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 310.0]
        metrics = evaluate_forecast(actual, predicted)
        assert isinstance(metrics, ForecastMetrics)
        assert metrics.mae == 10.0
        assert metrics.rmse == pytest.approx(10.0)
        assert metrics.mape > 0.0
