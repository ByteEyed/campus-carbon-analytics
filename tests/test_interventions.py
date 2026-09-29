"""
Unit and Integration Tests for Decarbonization Intervention Scenario Library
=============================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 13: Intervention Scenario Library Test Suite
Validates:
1. Core schema, metadata, and simulated assumption transparency.
2. Mathematical correctness for all five canonical interventions.
3. Seasonal masking behavior for AC / HVAC optimization.
4. Additive bounds enforcement and zero-clamping for solar generation.
5. Strict parameter bounds, non-negative costs, and invalid value rejection.
6. Target category and column compatibility.
7. Interaction sorting (multiplicative before additive).
8. Serialization roundtrip.
9. Determinism and non-interference with collateral columns.
10. Seamless compatibility with Phase 8 Carbon Accounting.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import calculate_campus_emissions
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
    intervention_from_dict,
    sort_interventions,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_single_row_df() -> pd.DataFrame:
    """Mock single-row activity DataFrame."""
    return pd.DataFrame({
        "date": ["2024-05-01"],
        "building": ["Central Library & Student Hub"],
        "electricity_kwh": [10000.0],
        "travel_km": [5000.0],
        "waste_kg": [2000.0],
        "procurement_inr": [300000.0],
    })


@pytest.fixture
def mock_full_year_df() -> pd.DataFrame:
    """Mock 12-month activity series for seasonal masking tests."""
    dates = [f"2024-{m:02d}-01" for m in range(1, 13)]
    return pd.DataFrame({
        "date": dates,
        "electricity_kwh": [10000.0] * 12,
        "travel_km": [5000.0] * 12,
        "waste_kg": [2000.0] * 12,
        "procurement_inr": [300000.0] * 12,
    })


@pytest.fixture
def raw_activity_df() -> pd.DataFrame:
    """Load canonical campus activity data."""
    path = Path("data/raw/campus_activity.csv")
    if not path.exists():
        pytest.skip("data/raw/campus_activity.csv not found")
    return pd.read_csv(path)


# =====================================================================
# 1. Catalog and Schema Validity Tests
# =====================================================================

def test_catalog_and_default_interventions() -> None:
    """Verify that all 5 interventions exist in catalog and instantiate with valid defaults."""
    assert len(INTERVENTION_CATALOG) == 5
    expected_ids = {"INT-001", "INT-002", "INT-003", "INT-004", "INT-005"}
    assert set(INTERVENTION_CATALOG.keys()) == expected_ids

    defaults = get_default_interventions()
    assert len(defaults) == 5

    for intervention in defaults:
        assert isinstance(intervention, BaseIntervention)
        assert intervention.id in expected_ids
        assert intervention.name
        assert intervention.capital_cost >= 0.0
        assert intervention.lifespan_years >= 1
        assert intervention.is_simulated_assumption is True
        assert intervention.priority in (1, 2)
        assert intervention.mechanism_type in (
            InterventionMechanism.MULTIPLICATIVE.value,
            InterventionMechanism.ADDITIVE.value,
        )


def test_canonical_aliases() -> None:
    """Verify that canonical aliases point to identical classes."""
    assert LEDLighting is LEDLightingRetrofit
    assert SolarInstallation is RooftopSolarInstallation
    assert HVACOptimization is ACOptimization
    assert WasteSegregationComposting is WasteSegregation
    assert SustainableEVTransport is LowCarbonTransport


# =====================================================================
# 2. Mathematical Correctness Tests for Each Intervention
# =====================================================================

def test_int_001_led_lighting_correctness(mock_single_row_df: pd.DataFrame) -> None:
    """Verify INT-001 reduces electricity_kwh multiplicatively and preserves other columns."""
    led = LEDLightingRetrofit(reduction_fraction=0.12)
    res = led.apply(mock_single_row_df)

    # 10000 * (1 - 0.12) = 8800.0
    assert res["electricity_kwh"].iloc[0] == 8800.0
    # Unrelated columns must remain exactly identical
    assert res["travel_km"].iloc[0] == mock_single_row_df["travel_km"].iloc[0]
    assert res["waste_kg"].iloc[0] == mock_single_row_df["waste_kg"].iloc[0]
    assert res["procurement_inr"].iloc[0] == mock_single_row_df["procurement_inr"].iloc[0]


def test_int_002_solar_installation_correctness(mock_single_row_df: pd.DataFrame) -> None:
    """Verify INT-002 subtracts generation additively."""
    solar = RooftopSolarInstallation(monthly_generation_kwh=4000.0)
    res = solar.apply(mock_single_row_df)

    # 10000 - 4000 = 6000.0
    assert res["electricity_kwh"].iloc[0] == 6000.0
    assert res["travel_km"].iloc[0] == mock_single_row_df["travel_km"].iloc[0]


def test_int_002_solar_installation_boundary_zero_clamp() -> None:
    """Verify INT-002 clamps to 0.0 when generation exceeds consumption (no negative electricity)."""
    df = pd.DataFrame({"electricity_kwh": [3500.0]})
    solar = RooftopSolarInstallation(monthly_generation_kwh=5000.0)
    res = solar.apply(df)

    # max(0, 3500 - 5000) = 0.0
    assert res["electricity_kwh"].iloc[0] == 0.0


def test_int_002_solar_multi_building_proportional_allocation() -> None:
    """Verify INT-002 allocates solar generation proportionally across buildings per date."""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "building": ["Building A", "Building B"],
        "electricity_kwh": [6000.0, 4000.0],  # Total = 10,000 kWh
    })
    solar = RooftopSolarInstallation(monthly_generation_kwh=2000.0)
    res = solar.apply(df)

    # Building A has 60% load -> receives 60% of 2000 = 1200 reduction -> 4800.0
    # Building B has 40% load -> receives 40% of 2000 = 800 reduction -> 3200.0
    assert res["electricity_kwh"].iloc[0] == 4800.0
    assert res["electricity_kwh"].iloc[1] == 3200.0
    # Total reduction across the date is exactly 2000.0 kWh
    assert (df["electricity_kwh"].sum() - res["electricity_kwh"].sum()) == 2000.0


def test_int_002_solar_target_building_override() -> None:
    """Verify INT-002 subtracts only from the specified building if target_building is set."""
    df = pd.DataFrame({
        "building": ["Building A", "Building B"],
        "electricity_kwh": [6000.0, 4000.0],
    })
    solar = RooftopSolarInstallation(monthly_generation_kwh=2000.0, target_building="Building B")
    res = solar.apply(df)

    assert res["electricity_kwh"].iloc[0] == 6000.0  # Unchanged
    assert res["electricity_kwh"].iloc[1] == 2000.0  # 4000 - 2000


def test_int_003_ac_optimization_seasonal_masking(mock_full_year_df: pd.DataFrame) -> None:
    """Verify INT-003 only curtails electricity during specified summer months."""
    ac = ACOptimization(reduction_fraction=0.15, active_months=(4, 5, 6, 7, 8))
    res = ac.apply(mock_full_year_df)

    summer_months = {4, 5, 6, 7, 8}
    for i, date_str in enumerate(mock_full_year_df["date"]):
        month = int(date_str.split("-")[1])
        if month in summer_months:
            # 10000 * (1 - 0.15) = 8500.0
            assert res["electricity_kwh"].iloc[i] == 8500.0, f"Month {month} should be curtailed"
        else:
            # Winter / off-season months must remain 10000.0
            assert res["electricity_kwh"].iloc[i] == 10000.0, f"Month {month} should not be curtailed"


def test_int_004_waste_segregation_correctness(mock_single_row_df: pd.DataFrame) -> None:
    """Verify INT-004 reduces waste_kg multiplicatively."""
    waste_int = WasteSegregation(diversion_fraction=0.35)
    res = waste_int.apply(mock_single_row_df)

    # 2000 * (1 - 0.35) = 1300.0
    assert res["waste_kg"].iloc[0] == 1300.0
    assert res["electricity_kwh"].iloc[0] == mock_single_row_df["electricity_kwh"].iloc[0]


def test_int_005_low_carbon_transport_correctness(mock_single_row_df: pd.DataFrame) -> None:
    """Verify INT-005 reduces travel_km multiplicatively."""
    ev = LowCarbonTransport(ev_adoption_fraction=0.20)
    res = ev.apply(mock_single_row_df)

    # 5000 * (1 - 0.20) = 4000.0
    assert res["travel_km"].iloc[0] == 4000.0
    assert res["electricity_kwh"].iloc[0] == mock_single_row_df["electricity_kwh"].iloc[0]


# =====================================================================
# 3. Parameter Validation and Boundary Rejection Tests
# =====================================================================

def test_negative_capital_cost_rejected() -> None:
    """Ensure negative capital costs raise ValueError."""
    with pytest.raises(ValueError, match="capital_cost cannot be negative"):
        LEDLightingRetrofit(capital_cost=-10000.0)


def test_invalid_lifespan_rejected() -> None:
    """Ensure lifespan_years < 1 or non-integer raises ValueError / TypeError."""
    with pytest.raises(ValueError, match="lifespan_years must be at least 1"):
        LEDLightingRetrofit(lifespan_years=0)

    with pytest.raises(TypeError, match="lifespan_years must be an integer"):
        LEDLightingRetrofit(lifespan_years="5")  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_frac", [-0.05, 1.05, float("nan"), float("inf")])
def test_invalid_reduction_fraction_rejected(invalid_frac: float) -> None:
    """Ensure reduction fractions outside [0.0, 1.0] or non-finite raise ValueError."""
    with pytest.raises(ValueError):
        LEDLightingRetrofit(reduction_fraction=invalid_frac)

    with pytest.raises(ValueError):
        WasteSegregation(diversion_fraction=invalid_frac)

    with pytest.raises(ValueError):
        LowCarbonTransport(ev_adoption_fraction=invalid_frac)


def test_invalid_monthly_generation_rejected() -> None:
    """Ensure negative or non-finite solar generation raises ValueError."""
    with pytest.raises(ValueError, match="monthly_generation_kwh cannot be negative"):
        RooftopSolarInstallation(monthly_generation_kwh=-100.0)

    with pytest.raises(ValueError, match="must be finite"):
        RooftopSolarInstallation(monthly_generation_kwh=float("nan"))


def test_invalid_active_months_rejected() -> None:
    """Ensure invalid active months raise ValueError."""
    with pytest.raises(ValueError, match="active_months cannot be empty"):
        ACOptimization(active_months=())

    with pytest.raises(ValueError, match="must contain integers from 1 to 12"):
        ACOptimization(active_months=(4, 5, 13))

    with pytest.raises(ValueError, match="must contain integers from 1 to 12"):
        ACOptimization(active_months=(0, 1, 2))


def test_target_category_mismatch_rejected() -> None:
    """Ensure mismatch between category and target_column raises ValueError."""
    with pytest.raises(ValueError, match="Target column mismatch"):
        LEDLightingRetrofit(target_column="travel_km")


def test_invalid_identifier_and_name_rejected() -> None:
    """Ensure empty id or name raises ValueError."""
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        LEDLightingRetrofit(id="")

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        LEDLightingRetrofit(name="   ")


# =====================================================================
# 4. Input Validation & DataFrame Safety Tests
# =====================================================================

def test_apply_missing_column_rejected(mock_single_row_df: pd.DataFrame) -> None:
    """Ensure missing target column raises ValueError."""
    df_missing = mock_single_row_df.drop(columns=["electricity_kwh"])
    led = LEDLightingRetrofit()
    with pytest.raises(ValueError, match="Target column 'electricity_kwh' not found"):
        led.apply(df_missing)


def test_apply_empty_df_rejected() -> None:
    """Ensure empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame({"electricity_kwh": []})
    led = LEDLightingRetrofit()
    with pytest.raises(ValueError, match="Activity DataFrame cannot be empty"):
        led.apply(empty_df)


def test_apply_non_dataframe_rejected() -> None:
    """Ensure non-DataFrame argument raises TypeError."""
    led = LEDLightingRetrofit()
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        led.apply({"electricity_kwh": [1000.0]})  # type: ignore[arg-type]


def test_apply_nan_in_target_column_rejected() -> None:
    """Ensure NaN values in activity data raise ValueError."""
    df_nan = pd.DataFrame({"electricity_kwh": [1000.0, np.nan, 2000.0]})
    led = LEDLightingRetrofit()
    with pytest.raises(ValueError, match="contains NaN values"):
        led.apply(df_nan)


def test_ac_optimization_missing_date_rejected() -> None:
    """Ensure ACOptimization raises ValueError when neither date nor month column is provided."""
    df_no_date = pd.DataFrame({"electricity_kwh": [1000.0, 2000.0]})
    ac = ACOptimization()
    with pytest.raises(ValueError, match="requires a 'date' or 'month' column"):
        ac.apply(df_no_date)


# =====================================================================
# 5. Interaction Ordering & Priority Tests
# =====================================================================

def test_sort_interventions_priority_ordering() -> None:
    """Verify Multiplicative (priority=1) is sorted before Additive (priority=2)."""
    led = LEDLightingRetrofit()          # Multiplicative, priority=1
    solar = RooftopSolarInstallation()  # Additive, priority=2
    ac = ACOptimization()                # Multiplicative, priority=1

    # Given unordered list: [solar, led, ac]
    unordered = [solar, led, ac]
    ordered = sort_interventions(unordered)

    # Multiplicative (led, ac) must precede Additive (solar)
    assert ordered[0].priority == 1
    assert ordered[1].priority == 1
    assert ordered[2].priority == 2
    assert isinstance(ordered[2], RooftopSolarInstallation)


def test_sequential_stacking_multiplicative_then_additive() -> None:
    """Verify that applying Multiplicative then Additive matches the physical stacking rule."""
    df = pd.DataFrame({"electricity_kwh": [10000.0]})
    led = LEDLightingRetrofit(reduction_fraction=0.20)         # Reduces 20% -> 8000 kWh
    solar = RooftopSolarInstallation(monthly_generation_kwh=5000.0)  # Subtracts 5000 kWh -> 3000 kWh

    # Correct physical order: LED then Solar
    step1 = led.apply(df)
    assert step1["electricity_kwh"].iloc[0] == 8000.0
    step2 = solar.apply(step1)
    assert step2["electricity_kwh"].iloc[0] == 3000.0


# =====================================================================
# 6. Serialization & Determinism Tests
# =====================================================================

def test_serialization_roundtrip() -> None:
    """Verify to_dict and intervention_from_dict preserve all properties."""
    for original in get_default_interventions():
        d = original.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == original.id

        reconstructed = intervention_from_dict(d)
        assert type(reconstructed) is type(original)
        assert reconstructed.id == original.id
        assert reconstructed.capital_cost == original.capital_cost
        assert reconstructed.lifespan_years == original.lifespan_years
        assert reconstructed.target_column == original.target_column


def test_deterministic_behavior(mock_single_row_df: pd.DataFrame) -> None:
    """Verify repeated applications produce strictly identical outputs."""
    led = LEDLightingRetrofit()
    out1 = led.apply(mock_single_row_df)
    out2 = led.apply(mock_single_row_df)
    pd.testing.assert_frame_equal(out1, out2, check_exact=True)


# =====================================================================
# 7. End-to-End Integration with Phase 8 Carbon Accounting
# =====================================================================

def test_end_to_end_with_carbon_accounting(raw_activity_df: pd.DataFrame) -> None:
    """Verify transformed activity data passes cleanly into calculate_campus_emissions."""
    # 1. Baseline emissions
    base_emissions = calculate_campus_emissions(raw_activity_df)
    base_elec_kg = base_emissions["electricity_emissions_kg"].sum()

    # 2. Apply LED Lighting Retrofit to raw activity data
    led = LEDLightingRetrofit(reduction_fraction=0.12)
    post_activity_df = led.apply(raw_activity_df)

    # 3. Recompute emissions on post-intervention activity data
    post_emissions = calculate_campus_emissions(post_activity_df)
    post_elec_kg = post_emissions["electricity_emissions_kg"].sum()

    # Post-intervention electricity emissions must be lower by exactly ~12%
    expected_ratio = 1.0 - 0.12
    actual_ratio = post_elec_kg / base_elec_kg
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-3)

    # Travel, waste, and procurement emissions must remain identical
    assert post_emissions["travel_emissions_kg"].sum() == base_emissions["travel_emissions_kg"].sum()
    assert post_emissions["waste_emissions_kg"].sum() == base_emissions["waste_emissions_kg"].sum()
    assert post_emissions["procurement_emissions_kg"].sum() == base_emissions["procurement_emissions_kg"].sum()
