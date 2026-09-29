"""
Unit and Integration Tests for Uncertainty Quantification Module
=================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 12: Uncertainty Analysis Test Suite
Validates:
1. Reproducibility across identical seeds.
2. Physical bounds enforcement (all perturbed inputs >= 0.0).
3. Zero-variance flag behavior (RSD=0 matches deterministic Phase 11 forecast).
4. Variance decomposition normalization (percentages sum strictly to 100.0%).
5. Parameter validation (negative RSD or invalid simulation count raises ValueError).
6. Output structure, intervals, and simulation modes.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import (
    CATEGORY_TO_ACTIVITY_COL,
    EmissionFactorRegistry,
    aggregate_emissions_by_date,
    calculate_campus_emissions,
)
from src.forecasting.holt_winters import HoltWintersForecaster
from src.uncertainty.monte_carlo import (
    DEFAULT_ACTIVITY_RSD,
    DEFAULT_EF_RSD,
    UncertaintyParameters,
    VarianceDecomposition,
    decompose_uncertainty,
    generate_perturbed_inputs,
    run_monte_carlo_pipeline,
)


@pytest.fixture
def sample_activity_df() -> pd.DataFrame:
    """Load canonical activity dataset for testing."""
    path = Path("data/raw/campus_activity.csv")
    if not path.exists():
        pytest.skip("data/raw/campus_activity.csv not found")
    return pd.read_csv(path)


@pytest.fixture
def default_registry() -> EmissionFactorRegistry:
    """Load default documented emission factor registry."""
    return EmissionFactorRegistry.default()


# =====================================================================
# 1. Parameter Validation Tests
# =====================================================================

def test_invalid_parameters() -> None:
    """Test that passing negative relative standard deviations raises ValueError."""
    # Negative activity RSD
    with pytest.raises(ValueError, match="Relative standard deviation cannot be negative"):
        UncertaintyParameters(activity_rsd={"electricity_kwh": -0.05})

    # Negative emission factor RSD
    with pytest.raises(ValueError, match="Relative standard deviation cannot be negative"):
        UncertaintyParameters(ef_rsd={"travel_km": -0.10})


def test_invalid_simulation_counts() -> None:
    """Test that non-positive simulation counts raise ValueError."""
    with pytest.raises(ValueError, match="n_simulations must be a positive integer"):
        UncertaintyParameters(n_simulations=0)

    with pytest.raises(ValueError, match="n_simulations must be a positive integer"):
        UncertaintyParameters(n_simulations=-50)


def test_non_finite_rsd() -> None:
    """Test that NaN or Infinite RSD values raise ValueError."""
    with pytest.raises(ValueError, match="must be finite"):
        UncertaintyParameters(activity_rsd={"electricity_kwh": float("nan")})

    with pytest.raises(ValueError, match="must be finite"):
        UncertaintyParameters(ef_rsd={"travel_km": float("inf")})


def test_invalid_parameter_types() -> None:
    """Test type validation for seeds and RSD mappings."""
    with pytest.raises(TypeError, match="seed must be an integer"):
        UncertaintyParameters(seed="42")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="activity_rsd must be a dictionary"):
        UncertaintyParameters(activity_rsd=[0.02, 0.05])  # type: ignore[arg-type]


# =====================================================================
# 2. Physical Non-Negativity & Bounds Tests
# =====================================================================

def test_physical_bounds(sample_activity_df: pd.DataFrame, default_registry: EmissionFactorRegistry) -> None:
    """Ensure no perturbation engine ever produces a negative activity or negative EF."""
    # Test with extreme relative standard deviations (e.g. 100% or 200%)
    extreme_params = UncertaintyParameters(
        activity_rsd={c: 2.0 for c in CATEGORY_TO_ACTIVITY_COL.values()},
        ef_rsd={c: 2.0 for c in CATEGORY_TO_ACTIVITY_COL.values()},
        seed=123,
    )

    for trial in range(10):
        p_df, p_reg = generate_perturbed_inputs(
            sample_activity_df,
            default_registry,
            extreme_params,
            rng=np.random.default_rng(trial),
        )

        # 1. Activity data non-negativity
        for col in CATEGORY_TO_ACTIVITY_COL.values():
            assert (p_df[col] >= 0.0).all(), f"Found negative activity in {col}"

        # 2. Emission factor non-negativity
        for f in p_reg.list_factors():
            assert f.factor >= 0.0, f"Found negative emission factor for {f.category}: {f.factor}"


# =====================================================================
# 3. Reproducibility Tests
# =====================================================================

def test_mc_reproducibility(sample_activity_df: pd.DataFrame, default_registry: EmissionFactorRegistry) -> None:
    """Ensure running the MC engine twice with seed=42 yields identical DataFrames."""
    params = UncertaintyParameters(
        activity_rsd=DEFAULT_ACTIVITY_RSD,
        ef_rsd=DEFAULT_EF_RSD,
        seed=42,
        n_simulations=30,  # Fast execution for test suite
    )

    df1 = run_monte_carlo_pipeline(
        df=sample_activity_df,
        registry=default_registry,
        params=params,
        target="total_emissions_kg",
        forecast_horizon=12,
    )

    df2 = run_monte_carlo_pipeline(
        df=sample_activity_df,
        registry=default_registry,
        params=params,
        target="total_emissions_kg",
        forecast_horizon=12,
    )

    pd.testing.assert_frame_equal(df1, df2, check_exact=True)


# =====================================================================
# 4. Zero-Variance Flag Test
# =====================================================================

def test_zero_variance_flag(sample_activity_df: pd.DataFrame, default_registry: EmissionFactorRegistry) -> None:
    """If activity_rsd = 0 and ef_rsd = 0, output forecast must perfectly match deterministic Phase 11 forecast."""
    zero_params = UncertaintyParameters(
        activity_rsd={col: 0.0 for col in CATEGORY_TO_ACTIVITY_COL.values()},
        ef_rsd={col: 0.0 for col in CATEGORY_TO_ACTIVITY_COL.values()},
        seed=42,
        n_simulations=20,
    )

    # 1. Deterministic Phase 11 baseline forecast
    emissions = calculate_campus_emissions(sample_activity_df, default_registry)
    monthly = aggregate_emissions_by_date(emissions)
    train_y = monthly["total_emissions_kg"].iloc[:24].to_numpy(dtype=float)
    deterministic_pt = HoltWintersForecaster().fit(train_y).forecast(12)

    # 2. Monte Carlo pipeline with zero variance
    mc_result = run_monte_carlo_pipeline(
        df=sample_activity_df,
        registry=default_registry,
        params=zero_params,
        target="total_emissions_kg",
        forecast_horizon=12,
    )

    # Point forecast and simulated mean must match deterministic forecast
    np.testing.assert_allclose(mc_result["point_forecast"].to_numpy(), deterministic_pt, rtol=1e-5)
    np.testing.assert_allclose(mc_result["mean"].to_numpy(), deterministic_pt, rtol=1e-5)
    np.testing.assert_allclose(mc_result["ci_lower_95"].to_numpy(), deterministic_pt, rtol=1e-5)
    np.testing.assert_allclose(mc_result["ci_upper_95"].to_numpy(), deterministic_pt, rtol=1e-5)

    # Standard deviation and relative uncertainty must be strictly zero
    assert (mc_result["std"] == 0.0).all()
    assert (mc_result["relative_uncertainty_pct"] == 0.0).all()


# =====================================================================
# 5. Variance Decomposition Normalization Test
# =====================================================================

def test_variance_decomposition_sums_to_100(
    sample_activity_df: pd.DataFrame,
    default_registry: EmissionFactorRegistry,
) -> None:
    """Ensure the output percentages equal 100.0%."""
    params = UncertaintyParameters(
        activity_rsd=DEFAULT_ACTIVITY_RSD,
        ef_rsd=DEFAULT_EF_RSD,
        seed=42,
        n_simulations=40,  # Fast execution
    )

    for target in ["total_emissions_kg", "electricity_emissions_kg", "travel_emissions_kg"]:
        decomp = decompose_uncertainty(
            df=sample_activity_df,
            registry=default_registry,
            params=params,
            target=target,
            forecast_horizon=12,
        )

        assert isinstance(decomp, VarianceDecomposition)
        assert decomp.target == target

        # Check non-negativity of percentage contributions
        assert decomp.forecast_variance_pct >= 0.0
        assert decomp.activity_variance_pct >= 0.0
        assert decomp.ef_variance_pct >= 0.0

        # Check exact sum to 100.0%
        pct_sum = round(
            decomp.forecast_variance_pct + decomp.activity_variance_pct + decomp.ef_variance_pct,
            2,
        )
        assert math.isclose(pct_sum, 100.0, abs_tol=1e-4), f"Sum is {pct_sum} for {target}"

        # Check total variance consistency
        assert decomp.total_variance > 0.0
        assert decomp.forecast_variance >= 0.0
        assert decomp.activity_variance >= 0.0
        assert decomp.ef_variance >= 0.0


# =====================================================================
# 6. Pipeline Modes and Structure Tests
# =====================================================================

def test_monte_carlo_modes(sample_activity_df: pd.DataFrame, default_registry: EmissionFactorRegistry) -> None:
    """Verify that mode options 'activity_only' and 'ef_only' function correctly."""
    params = UncertaintyParameters(seed=42, n_simulations=20)

    res_act = run_monte_carlo_pipeline(
        df=sample_activity_df,
        registry=default_registry,
        params=params,
        mode="activity_only",
        target="total_emissions_kg",
    )
    assert len(res_act) == 12
    assert "ci_lower_95" in res_act.columns
    assert "ci_upper_95" in res_act.columns

    res_ef = run_monte_carlo_pipeline(
        df=sample_activity_df,
        registry=default_registry,
        params=params,
        mode="ef_only",
        target="total_emissions_kg",
    )
    assert len(res_ef) == 12

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Unknown mode"):
        run_monte_carlo_pipeline(
            df=sample_activity_df,
            registry=default_registry,
            params=params,
            mode="invalid_mode",
        )


def test_monte_carlo_interval_sandwiching(
    sample_activity_df: pd.DataFrame,
    default_registry: EmissionFactorRegistry,
) -> None:
    """Verify that confidence bounds satisfy lower <= upper and non-negativity."""
    params = UncertaintyParameters(seed=42, n_simulations=30)
    df = run_monte_carlo_pipeline(
        df=sample_activity_df,
        registry=default_registry,
        params=params,
        target="total_emissions_kg",
    )

    # 1. Lower bound is non-negative
    assert (df["ci_lower_95"] >= 0.0).all()

    # 2. Lower bound is <= Upper bound
    assert (df["ci_lower_95"] <= df["ci_upper_95"]).all()

    # 3. Relative uncertainty is non-negative
    assert (df["relative_uncertainty_pct"] >= 0.0).all()
