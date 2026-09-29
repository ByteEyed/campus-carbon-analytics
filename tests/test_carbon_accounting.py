"""
Tests for Campus Carbon Accounting Engine
=========================================
Tests calculation accuracy, edge cases, failure modes, registry configurability,
and aggregations:
- Normal calculation
- Zero activity
- Missing factor
- Invalid factor (negative, NaN, Inf, non-numeric)
- Negative activity
- Registry operations & configurability (roundtrip, override collision)
- Unit matching validation
- Known-answer precision test against hand-computed values
- NaN / Inf error handling in DataFrame pipeline
- Missing required columns in DataFrame pipeline
- Strict factors fallback handling
- Date, category, total, and per-student aggregations
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.carbon_accounting import (
    CATEGORY_ELECTRICITY,
    CATEGORY_PROCUREMENT,
    CATEGORY_TRAVEL,
    CATEGORY_WASTE,
    CarbonAccountingSummary,
    EmissionFactor,
    EmissionFactorRegistry,
    aggregate_emissions_by_category,
    aggregate_emissions_by_date,
    calculate_campus_emissions,
    calculate_category_contributions,
    calculate_emissions,
    calculate_emissions_per_student,
    calculate_total_emissions,
    run_carbon_accounting,
)
from src.data_generator import generate_campus_activity_data


# -----------------------------------------------------------------------------
# Unit Tests for calculate_emissions()
# -----------------------------------------------------------------------------


def test_calculate_emissions_normal():
    """Verify standard calculation produces expected emissions (activity * factor)."""
    # 10,000 kWh * 0.716 kgCO2e/kWh = 7,160.0 kgCO2e
    result = calculate_emissions(10000.0, 0.716)
    assert result == 7160.0

    # 250.5 km * 0.140 kgCO2e/km = 35.07 kgCO2e
    result_travel = calculate_emissions(250.5, 0.140)
    assert result_travel == 35.07


def test_calculate_emissions_zero_activity():
    """Verify zero activity or zero factor returns exactly 0.0 emissions."""
    assert calculate_emissions(0.0, 0.716) == 0.0
    assert calculate_emissions(500.0, 0.0) == 0.0
    assert calculate_emissions(0.0, 0.0) == 0.0


def test_calculate_emissions_missing_factor():
    """Verify missing factor or missing activity raises ValueError."""
    with pytest.raises(ValueError, match="cannot be None"):
        calculate_emissions(100.0, None)  # type: ignore

    with pytest.raises(ValueError, match="cannot be None"):
        calculate_emissions(None, 0.716)  # type: ignore


def test_calculate_emissions_invalid_factor():
    """Verify invalid factors (negative, NaN, inf, non-numeric) raise ValueError."""
    # Negative factor
    with pytest.raises(ValueError, match="Negative emission factor is invalid"):
        calculate_emissions(100.0, -0.716)

    # NaN factor
    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        calculate_emissions(100.0, float("nan"))

    # Inf factor
    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        calculate_emissions(100.0, float("inf"))

    # Non-numeric string
    with pytest.raises(ValueError, match="not a valid number"):
        calculate_emissions(100.0, "invalid_factor")  # type: ignore


def test_calculate_emissions_negative_activity():
    """Verify negative activity values are strictly rejected."""
    with pytest.raises(ValueError, match="Negative activity value is invalid"):
        calculate_emissions(-500.0, 0.716)

    with pytest.raises(ValueError, match="Negative activity value is invalid"):
        calculate_emissions(-0.01, 0.140)


# -----------------------------------------------------------------------------
# EmissionFactor & Registry Tests
# -----------------------------------------------------------------------------


def test_emission_factor_validation():
    """Verify EmissionFactor dataclass validation rules."""
    # Valid instance
    factor = EmissionFactor(
        category="Electricity",
        activity_type="grid_kwh",
        unit="kgCO2e/kWh",
        factor=0.716,
        source="CEA CO2 Database",
        version="2024",
    )
    assert factor.factor == 0.716

    # Empty category
    with pytest.raises(ValueError, match="'category' cannot be empty"):
        EmissionFactor("", "grid_kwh", "kgCO2e/kWh", 0.5, "CEA", "2024")

    # Negative factor
    with pytest.raises(ValueError, match="cannot be negative"):
        EmissionFactor("Electricity", "grid_kwh", "kgCO2e/kWh", -0.5, "CEA", "2024")

    # NaN factor
    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        EmissionFactor("Electricity", "grid_kwh", "kgCO2e/kWh", float("nan"), "CEA", "2024")


def test_emission_factor_unit_mismatch():
    """Verify unit mismatch error when registering unexpected units (e.g. MWh vs kWh)."""
    with pytest.raises(ValueError, match="Unit mismatch for category 'Electricity'"):
        EmissionFactor(
            category="Electricity",
            activity_type="electricity_kwh",
            unit="kgCO2e/MWh",  # Mismatched unit!
            factor=716.0,
            source="CEA",
            version="2024",
        )


def test_emission_factor_zero_warning(caplog: pytest.LogCaptureFixture):
    """Verify zero emission factor logs an explicit warning."""
    with caplog.at_level(logging.WARNING, logger="campus_carbon.accounting"):
        factor = EmissionFactor(
            category="Electricity",
            activity_type="electricity_kwh",
            unit="kgCO2e/kWh",
            factor=0.0,
            source="100% On-site Solar PPA",
            version="2024",
        )
    assert factor.factor == 0.0
    assert any("registered as 0.0" in r.message for r in caplog.records)


def test_emission_factor_registry_lookup_and_configurability():
    """Verify registry supports case-insensitive lookups, custom factors, and overrides."""
    registry = EmissionFactorRegistry.default()

    # Default lookup by activity_type and category
    ef_elec = registry.get_factor("electricity_kwh")
    assert ef_elec.factor == 0.716
    assert registry.get_factor_value("ELECTRICITY") == 0.716

    # Configurable override (e.g. green grid tariff)
    new_factor = EmissionFactor(
        category="Electricity",
        activity_type="electricity_kwh",
        unit="kgCO2e/kWh",
        factor=0.350,
        source="Green Tariff PPA Documentation",
        version="2025.1",
    )
    registry.register(new_factor)
    assert registry.get_factor_value("electricity_kwh") == 0.350
    assert registry.get_factor_value("Electricity") == 0.350

    # Missing factor lookup
    with pytest.raises(KeyError, match="missing or not registered"):
        registry.get_factor("UnknownCategory")


def test_registry_csv_roundtrip(tmp_path: Path):
    """Verify registry exports to CSV and reloads without loss of fidelity."""
    registry = EmissionFactorRegistry.default()
    csv_file = tmp_path / "test_factors.csv"
    registry.to_csv(csv_file)

    loaded_reg = EmissionFactorRegistry.from_csv(csv_file)
    assert len(loaded_reg.list_factors()) == len(registry.list_factors())
    assert loaded_reg.get_factor_value("electricity_kwh") == 0.716
    assert loaded_reg.get_factor_value("travel_km") == 0.140


def test_from_dataframe_validation():
    """Verify from_dataframe checks for empty dataframes and missing required columns."""
    with pytest.raises(ValueError, match="cannot be empty"):
        EmissionFactorRegistry.from_dataframe(pd.DataFrame())

    with pytest.raises(ValueError, match="missing required column"):
        EmissionFactorRegistry.from_dataframe(
            pd.DataFrame({"category": ["Electricity"], "factor": [0.5]})
        )


# -----------------------------------------------------------------------------
# DataFrame Level Emission Calculations & Precision Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_activity_df() -> pd.DataFrame:
    """Fixture providing clean 6-row sample dataset across 2 months and 3 buildings."""
    return generate_campus_activity_data(
        start_date="2023-01-01",
        end_date="2023-02-01",
        num_buildings=3,
        seed=42,
    )


def test_known_answer_emission_calculation():
    """
    Known-Answer Test (Oracle verification).
    Hand-calculated values:
    electricity: 15900.64 * 0.716 = 11384.85824 -> round(4) = 11384.8582
    travel:      38600.36 * 0.140 = 5404.0504
    waste:       2664.63  * 0.446 = 1188.42498 -> round(4) = 1188.425
    procurement: 187000.13 * 0.00042 = 78.5400546 -> round(4) = 78.5401
    total_kg:    11384.8582 + 5404.0504 + 1188.425 + 78.5401 = 18055.8737
    total_mt:    18055.8737 / 1000.0 = 18.0559
    per_student: 18055.8737 / 1319 = 13.6891
    """
    df_single = pd.DataFrame([{
        "date": "2023-01-01",
        "building": "Academic Block - Arts & Humanities",
        "electricity_kwh": 15900.64,
        "travel_km": 38600.36,
        "waste_kg": 2664.63,
        "procurement_inr": 187000.13,
        "student_count": 1319,
        "staff_count": 54,
    }])
    result = calculate_campus_emissions(df_single)

    assert result.loc[0, "electricity_emissions_kg"] == 11384.8582
    assert result.loc[0, "travel_emissions_kg"] == 5404.0504
    assert result.loc[0, "waste_emissions_kg"] == 1188.425
    assert result.loc[0, "procurement_emissions_kg"] == 78.5401
    assert result.loc[0, "total_emissions_kg"] == 18055.8737
    assert result.loc[0, "total_emissions_mt"] == 18.0559
    assert result.loc[0, "emissions_per_student_kg"] == 13.6891


def test_calculate_campus_emissions_structure(sample_activity_df: pd.DataFrame):
    """Verify DataFrame-level calculation of emissions and per-student metrics."""
    emissions_df = calculate_campus_emissions(sample_activity_df)

    expected_cols = [
        "electricity_emissions_kg",
        "travel_emissions_kg",
        "waste_emissions_kg",
        "procurement_emissions_kg",
        "total_emissions_kg",
        "total_emissions_mt",
        "emissions_per_student_kg",
    ]
    for col in expected_cols:
        assert col in emissions_df.columns

    # Verify mathematical sum
    calc_sum = (
        emissions_df["electricity_emissions_kg"]
        + emissions_df["travel_emissions_kg"]
        + emissions_df["waste_emissions_kg"]
        + emissions_df["procurement_emissions_kg"]
    ).round(2)
    assert (emissions_df["total_emissions_kg"].round(2) == calc_sum).all()

    # Metric tonnes conversion
    assert (emissions_df["total_emissions_mt"] == (emissions_df["total_emissions_kg"] / 1000.0).round(4)).all()


def test_calculate_campus_emissions_missing_columns(sample_activity_df: pd.DataFrame):
    """Verify DataFrame-level calculation rejects missing required activity columns."""
    bad_df = sample_activity_df.drop(columns=["electricity_kwh"])
    with pytest.raises(ValueError, match="missing required column"):
        calculate_campus_emissions(bad_df)


def test_calculate_campus_emissions_nan_error(sample_activity_df: pd.DataFrame):
    """Verify DataFrame-level calculation detects and rejects NaN values."""
    bad_df = sample_activity_df.copy()
    bad_df.loc[0, "travel_km"] = np.nan
    with pytest.raises(ValueError, match="contains 1 NaN/null values"):
        calculate_campus_emissions(bad_df)


def test_calculate_campus_emissions_inf_error(sample_activity_df: pd.DataFrame):
    """Verify DataFrame-level calculation detects and rejects Infinite values."""
    bad_df = sample_activity_df.copy()
    bad_df.loc[1, "procurement_inr"] = np.inf
    with pytest.raises(ValueError, match="contains Infinite values"):
        calculate_campus_emissions(bad_df)


def test_calculate_campus_emissions_negative_activity_error(sample_activity_df: pd.DataFrame):
    """Verify DataFrame-level calculation rejects negative activity values."""
    bad_df = sample_activity_df.copy()
    bad_df.loc[0, "electricity_kwh"] = -100.0

    with pytest.raises(ValueError, match="contains negative values"):
        calculate_campus_emissions(bad_df)


# -----------------------------------------------------------------------------
# Aggregations & Summary Tests
# -----------------------------------------------------------------------------


def test_aggregate_emissions_by_date(sample_activity_df: pd.DataFrame):
    """
    Verify emissions aggregated by date sum correctly.
    Verifies that student_count is NOT summed in date aggregation to avoid denominator dilution.
    """
    emissions_df = calculate_campus_emissions(sample_activity_df)
    by_date = aggregate_emissions_by_date(emissions_df)

    assert len(by_date) == 2  # 2 months
    assert list(by_date["date"]) == ["2023-01-01", "2023-02-01"]
    assert by_date["total_emissions_kg"].sum() == pytest.approx(
        emissions_df["total_emissions_kg"].sum(), rel=1e-3
    )

    # Date aggregation must not contain diluted per-student metric
    assert "emissions_per_student_kg" not in by_date.columns
    assert "student_count" not in by_date.columns


def test_aggregate_emissions_by_category(sample_activity_df: pd.DataFrame):
    """Verify emissions aggregated by category and percentage contributions."""
    emissions_df = calculate_campus_emissions(sample_activity_df)
    by_cat = aggregate_emissions_by_category(emissions_df)

    assert len(by_cat) == 4
    categories = set(by_cat["category"])
    assert categories == {CATEGORY_ELECTRICITY, CATEGORY_TRAVEL, CATEGORY_WASTE, CATEGORY_PROCUREMENT}

    # Percentages must sum to 100%
    assert pytest.approx(by_cat["contribution_percentage"].sum(), abs=0.1) == 100.0


def test_calculate_total_and_contributions(sample_activity_df: pd.DataFrame):
    """Verify total emissions and category contribution dictionary helper."""
    emissions_df = calculate_campus_emissions(sample_activity_df)

    total_kg = calculate_total_emissions(emissions_df)
    assert total_kg > 0
    assert total_kg == round(float(emissions_df["total_emissions_kg"].sum()), 2)

    contributions = calculate_category_contributions(emissions_df)
    assert isinstance(contributions, dict)
    assert len(contributions) == 4
    assert pytest.approx(sum(contributions.values()), abs=0.1) == 100.0


def test_emissions_per_student_safe_division():
    """Verify emissions per student calculation handles regular, zero, and unique student headcounts."""
    # Regular dataset: 3000 kg / 150 student-months = 20.0 kg/student-month
    df = pd.DataFrame({
        "total_emissions_kg": [1000.0, 2000.0],
        "student_count": [50, 100],
    })
    assert calculate_emissions_per_student(df) == 20.0

    # With campus unique student headcount provided: 3000 kg / 100 unique students = 30.0 kg/student
    assert calculate_emissions_per_student(df, unique_students=100) == 30.0
    assert calculate_emissions_per_student(df, unique_students=0) == 0.0

    # Zero student count edge case
    df_zero = pd.DataFrame({
        "total_emissions_kg": [1000.0],
        "student_count": [0],
    })
    assert calculate_emissions_per_student(df_zero) == 0.0


def test_carbon_accounting_summary_string(sample_activity_df: pd.DataFrame):
    """Verify CarbonAccountingSummary formatting and labels."""
    emissions_df = calculate_campus_emissions(sample_activity_df)
    summary = CarbonAccountingSummary(
        total_emissions_kg=calculate_total_emissions(emissions_df),
        total_emissions_mt=round(calculate_total_emissions(emissions_df) / 1000.0, 2),
        emissions_per_student_kg=calculate_emissions_per_student(emissions_df),
        category_contributions=calculate_category_contributions(emissions_df),
        emissions_by_category=aggregate_emissions_by_category(emissions_df),
        emissions_by_date=aggregate_emissions_by_date(emissions_df),
    )
    summary_text = summary.summary()
    assert "CAMPUS CARBON ACCOUNTING SUMMARY" in summary_text
    assert "MTCO2e / metric tonnes" in summary_text
    assert "Electricity" in summary_text
    assert "Travel" in summary_text


def test_strict_factors_file_not_found(tmp_path: Path):
    """Verify strict_factors=True raises FileNotFoundError if factors file is absent."""
    non_existent = tmp_path / "missing_factors.csv"
    with pytest.raises(FileNotFoundError, match="Emission factors file required but not found"):
        run_carbon_accounting(
            activity_path="data/processed/cleaned_activity.csv",
            factors_path=non_existent,
            strict_factors=True,
        )


def test_single_row_pipeline():
    """Verify single-row DataFrame processes through entire pipeline without edge case failures."""
    df_one = pd.DataFrame([{
        "date": "2024-01-01",
        "building": "Test Block",
        "electricity_kwh": 5000.0,
        "travel_km": 1200.0,
        "waste_kg": 300.0,
        "procurement_inr": 45000.0,
        "student_count": 250,
        "staff_count": 20,
    }])
    emissions_df = calculate_campus_emissions(df_one)
    assert len(emissions_df) == 1
    assert emissions_df.loc[0, "total_emissions_kg"] > 0

    by_date = aggregate_emissions_by_date(emissions_df)
    assert len(by_date) == 1

    by_cat = aggregate_emissions_by_category(emissions_df)
    assert len(by_cat) == 4
    assert pytest.approx(by_cat["contribution_percentage"].sum(), abs=0.1) == 100.0


def test_end_to_end_carbon_accounting_pipeline(tmp_path: Path):
    """Verify end-to-end execution with real files."""
    out_csv = tmp_path / "processed_emissions.csv"
    emissions_df, summary = run_carbon_accounting(
        activity_path="data/processed/cleaned_activity.csv",
        factors_path="data/emission_factors.csv",
        output_path=out_csv,
    )

    assert out_csv.exists()
    assert len(emissions_df) == 180
    assert summary.total_emissions_kg > 0
    assert summary.total_emissions_mt > 0
    assert summary.emissions_per_student_kg > 0
    assert len(summary.category_contributions) == 4
    assert len(summary.summary()) > 100
