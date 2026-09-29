"""
Tests for Synthetic Campus Activity Data Generator
=================================================
Validates schema compliance, reproducibility, configurability,
seasonality characteristics, and failure modes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    REQUIRED_COLUMNS,
    generate_campus_activity_data,
    save_campus_activity_data,
    validate_campus_activity_data,
)


def test_default_generation_shape_and_columns():
    """Verify default generation produces ~3 years (36 months) for 5 buildings with 8 columns."""
    df = generate_campus_activity_data()

    assert isinstance(df, pd.DataFrame)
    # 36 months * 5 buildings = 180 rows
    assert df.shape == (180, 8)
    assert list(df.columns) == REQUIRED_COLUMNS

    # Date range checks
    unique_dates = df["date"].unique()
    assert len(unique_dates) == 36
    assert unique_dates[0] == "2023-01-01"
    assert unique_dates[-1] == "2025-12-01"

    # Building checks
    buildings = df["building"].unique()
    assert len(buildings) == 5
    for b in buildings:
        assert (df["building"] == b).sum() == 36


def test_reproducibility():
    """Verify identical random seed produces bit-for-bit identical datasets."""
    df1 = generate_campus_activity_data(seed=123)
    df2 = generate_campus_activity_data(seed=123)
    pd.testing.assert_frame_equal(df1, df2)

    # Different seeds should produce different outputs
    df3 = generate_campus_activity_data(seed=999)
    assert not df1.equals(df3)


def test_configurable_building_count():
    """Verify generator scales with custom building counts."""
    # Fewer buildings than default
    df_3 = generate_campus_activity_data(num_buildings=3)
    assert len(df_3["building"].unique()) == 3
    assert len(df_3) == 3 * 36

    # More buildings than default (tests dynamic profile extension)
    df_7 = generate_campus_activity_data(num_buildings=7)
    assert len(df_7["building"].unique()) == 7
    assert len(df_7) == 7 * 36


def test_configurable_date_range():
    """Verify generator respects custom date ranges."""
    df_1yr = generate_campus_activity_data(
        start_date="2024-01-01",
        end_date="2024-12-01",
        num_buildings=4,
    )
    assert len(df_1yr["date"].unique()) == 12
    assert len(df_1yr) == 12 * 4
    assert df_1yr["date"].min() == "2024-01-01"
    assert df_1yr["date"].max() == "2024-12-01"


def test_custom_building_names():
    """Verify custom building names are applied correctly."""
    custom_names = ["North Wing", "South Wing", "Innovation Lab"]
    df = generate_campus_activity_data(
        num_buildings=3,
        building_names=custom_names,
    )
    assert set(df["building"].unique()) == set(custom_names)


def test_include_simulated_flag():
    """Verify optional simulated flag is included when requested."""
    df = generate_campus_activity_data(include_simulated_flag=True)
    assert "is_simulated" in df.columns
    assert df["is_simulated"].all()


def test_realistic_seasonality_and_variation():
    """Verify institutional seasonality patterns (vacations, climate cooling, fiscal cycles)."""
    df = generate_campus_activity_data(seed=42)
    df["month"] = pd.to_datetime(df["date"]).dt.month

    # 1. Summer break (June) student drop vs Peak semester (October)
    june_students = df[df["month"] == 6]["student_count"].mean()
    oct_students = df[df["month"] == 10]["student_count"].mean()
    assert june_students < 0.60 * oct_students

    # 2. Summer break travel drop vs Peak semester travel
    june_travel = df[df["month"] == 6]["travel_km"].mean()
    oct_travel = df[df["month"] == 10]["travel_km"].mean()
    assert june_travel < 0.50 * oct_travel

    # 3. Fiscal year-end procurement spike in March
    march_proc = df[df["month"] == 3]["procurement_inr"].mean()
    feb_proc = df[df["month"] == 2]["procurement_inr"].mean()
    apr_proc = df[df["month"] == 4]["procurement_inr"].mean()
    assert march_proc > feb_proc
    assert march_proc > apr_proc

    # 4. Building-specific archetypes:
    # Science & Engineering should consume more electricity than Arts & Humanities
    sci_elec = df[df["building"] == "Science & Engineering Complex"]["electricity_kwh"].mean()
    arts_elec = df[df["building"] == "Academic Block - Arts & Humanities"]["electricity_kwh"].mean()
    assert sci_elec > 1.5 * arts_elec

    # Hostel & Dining should have highest waste
    hostel_waste = df[df["building"] == "Hostel & Dining Complex"]["waste_kg"].mean()
    admin_waste = df[df["building"] == "Administrative Headquarters"]["waste_kg"].mean()
    assert hostel_waste > 3.0 * admin_waste


def test_validation_passes_valid_data():
    """Verify validation passes on proper data."""
    df = generate_campus_activity_data(validate=False)
    assert validate_campus_activity_data(df) is True


def test_validation_missing_columns():
    """Verify validation fails if required columns are missing."""
    df = generate_campus_activity_data(validate=False)
    df_missing = df.drop(columns=["electricity_kwh"])
    with pytest.raises(ValueError, match="Missing required column"):
        validate_campus_activity_data(df_missing)


def test_validation_null_values():
    """Verify validation fails if null values are injected."""
    df = generate_campus_activity_data(validate=False)
    df.loc[0, "waste_kg"] = np.nan
    with pytest.raises(ValueError, match="contains null values"):
        validate_campus_activity_data(df)


def test_validation_negative_values():
    """Verify validation detects physically impossible negative values."""
    df = generate_campus_activity_data(validate=False)
    df.loc[2, "electricity_kwh"] = -50.0
    with pytest.raises(ValueError, match="Negative values detected in column 'electricity_kwh'"):
        validate_campus_activity_data(df)


def test_validation_non_integer_counts():
    """Verify validation fails if student or staff count is fractional."""
    df = generate_campus_activity_data(validate=False)
    df["student_count"] = df["student_count"].astype(float)
    df.loc[0, "student_count"] = 520.5
    with pytest.raises(ValueError, match="must contain integer values"):
        validate_campus_activity_data(df)


def test_validation_duplicate_records():
    """Verify validation fails if duplicate (date, building) entries exist."""
    df = generate_campus_activity_data(validate=False)
    duplicate_row = df.iloc[[0]].copy()
    df_dup = pd.concat([df, duplicate_row], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_campus_activity_data(df_dup)


def test_validation_non_monotonic_dates():
    """Verify validation fails if dates per building are out of order."""
    df = generate_campus_activity_data(validate=False)
    # Swap two rows of the same building
    b_indices = df[df["building"] == "Administrative Headquarters"].index
    if len(b_indices) >= 2:
        val0 = df.loc[b_indices[0], "date"]
        val1 = df.loc[b_indices[1], "date"]
        df.loc[b_indices[0], "date"] = val1
        df.loc[b_indices[1], "date"] = val0
        with pytest.raises(ValueError, match="not monotonically increasing"):
            validate_campus_activity_data(df)


def test_invalid_parameters():
    """Verify generator catches invalid arguments."""
    with pytest.raises(ValueError, match="cannot be after end_date"):
        generate_campus_activity_data(start_date="2025-01-01", end_date="2023-01-01")

    with pytest.raises(ValueError, match="positive integer"):
        generate_campus_activity_data(num_buildings=0)

    with pytest.raises(ValueError, match="must match num_buildings"):
        generate_campus_activity_data(num_buildings=3, building_names=["B1", "B2"])


def test_save_campus_activity_data(tmp_path: Path):
    """Verify CSV export functionality."""
    df = generate_campus_activity_data()
    out_file = tmp_path / "subfolder" / "test_activity.csv"
    saved = save_campus_activity_data(df, out_file)

    assert saved.exists()
    loaded_df = pd.read_csv(saved)
    assert len(loaded_df) == len(df)
    assert list(loaded_df.columns) == list(df.columns)
