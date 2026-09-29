"""
Tests for Forecasting Baseline Experiment
=========================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 10: Experiment Pipeline & Leakage Verification
Tests:
1. Data loading and monthly aggregation.
2. Chronological split (24 train, 12 test, no date overlap).
3. Monotonic date ordering.
4. Baseline experiment execution (Standard Naive vs Seasonal Naive).
5. Seasonal Naive outperforms Standard Naive across all 3 targets.
6. Temporal leakage prevention in experiment pipeline.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.forecasting.experiment import (
    DEFAULT_TARGETS,
    load_and_aggregate_monthly_emissions,
    run_baseline_experiment,
    split_chronological,
)


class TestChronologicalDataSplit:
    """Test suite for chronological data splitting and temporal integrity."""

    def test_load_and_aggregate_monthly_emissions(self) -> None:
        """Verify aggregated monthly dataset structure, row count, and date continuity."""
        df = load_and_aggregate_monthly_emissions("data/processed/campus_emissions.csv")
        assert len(df) == 36
        assert "date" in df.columns
        for target in DEFAULT_TARGETS:
            assert target in df.columns

        # Verify strict ascending chronology
        dates = pd.to_datetime(df["date"])
        assert dates.is_monotonic_increasing

    def test_chronological_split_lengths_and_dates(self) -> None:
        """
        Verify that chronological split returns exactly 24 train and 12 test rows,
        with no overlapping dates and strict temporal ordering.
        Matches Section 4 & 10 of Phase 10 spec.
        """
        df = load_and_aggregate_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological(df, train_months=24, test_months=12)

        # 1. Exact row counts
        assert len(train_df) == 24
        assert len(test_df) == 12

        # 2. Exact date boundaries
        assert train_df["date"].iloc[0] == "2023-01-01"
        assert train_df["date"].iloc[-1] == "2024-12-01"
        assert test_df["date"].iloc[0] == "2025-01-01"
        assert test_df["date"].iloc[-1] == "2025-12-01"

        # 3. No overlapping dates
        train_dates = set(train_df["date"])
        test_dates = set(test_df["date"])
        assert train_dates.isdisjoint(test_dates)

        # 4. Strict temporal boundary (no lookahead)
        train_end = pd.to_datetime(train_df["date"].iloc[-1])
        test_start = pd.to_datetime(test_df["date"].iloc[0])
        assert train_end < test_start

    def test_split_insufficient_length_raises(self) -> None:
        """Splitting a dataset shorter than train + test must raise ValueError."""
        short_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=15, freq="MS").strftime("%Y-%m-%d"),
            "total_emissions_kg": range(15),
        })
        with pytest.raises(ValueError, match="insufficient"):
            split_chronological(short_df, train_months=24, test_months=12)

    def test_split_non_monotonic_dates_raises(self) -> None:
        """Splitting non-chronological data must raise ValueError."""
        bad_df = pd.DataFrame({
            "date": ["2023-02-01", "2023-01-01"] * 18,
            "total_emissions_kg": range(36),
        })
        with pytest.raises(ValueError, match="monotonic"):
            split_chronological(bad_df, train_months=24, test_months=12)


class TestBaselineExperimentExecution:
    """Test suite for baseline experiment evaluation."""

    def test_seasonal_naive_beats_standard_naive_on_all_targets(self) -> None:
        """
        Verify that Seasonal Naive outperforms Standard Naive baseline
        on all 3 targets by a massive margin (>90% MAE improvement)
        due to the pronounced academic seasonality.
        Matches Section 12 Acceptance Criteria.
        """
        df = load_and_aggregate_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological(df, train_months=24, test_months=12)

        summary_df, predictions = run_baseline_experiment(train_df, test_df)

        # 3 targets * 2 models = 6 rows
        assert len(summary_df) == 6

        for target in DEFAULT_TARGETS:
            target_rows = summary_df[summary_df["target"] == target]
            naive_mae = float(target_rows[target_rows["model"].str.startswith("Standard")]["mae_kg"].iloc[0])
            s_naive_mae = float(target_rows[target_rows["model"].str.startswith("Seasonal")]["mae_kg"].iloc[0])

            # Seasonal Naive must achieve lower error
            assert s_naive_mae < naive_mae, (
                f"Seasonal Naive MAE ({s_naive_mae:.2f}) failed to beat Standard Naive ({naive_mae:.2f}) for {target}"
            )

            # Improvement should be >90% on this monthly campus data
            imp = float(target_rows[target_rows["model"].str.startswith("Seasonal")]["mae_improvement_pct"].iloc[0])
            assert imp > 90.0

            # Verify predictions exist
            assert target in predictions
            assert len(predictions[target]["Standard Naive"]) == 12
            assert len(predictions[target]["Seasonal Naive"]) == 12

    def test_experiment_temporal_leakage_guard(self) -> None:
        """
        Verify that mutating test_df values does not change the model forecasts.
        Proves that baseline models receive ONLY training data.
        """
        df = load_and_aggregate_monthly_emissions("data/processed/campus_emissions.csv")
        train_df, test_df = split_chronological(df, train_months=24, test_months=12)

        # Run 1: baseline experiment with original test set
        _, preds_1 = run_baseline_experiment(train_df, test_df)

        # Run 2: modify test set with 10x artificial values
        corrupted_test_df = test_df.copy()
        corrupted_test_df["total_emissions_kg"] = corrupted_test_df["total_emissions_kg"] * 10.0

        _, preds_2 = run_baseline_experiment(train_df, corrupted_test_df)

        # Forecast predictions must be 100% identical
        for target in DEFAULT_TARGETS:
            np.testing.assert_allclose(
                preds_1[target]["Standard Naive"],
                preds_2[target]["Standard Naive"],
            )
            np.testing.assert_allclose(
                preds_1[target]["Seasonal Naive"],
                preds_2[target]["Seasonal Naive"],
            )
