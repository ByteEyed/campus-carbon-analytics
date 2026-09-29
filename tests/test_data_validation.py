"""
Tests for Campus Activity Data Validation Pipeline
==================================================
Tests detection and reporting of all major validation failure modes:
- Missing required columns
- Missing values (NaN, null, empty strings)
- Duplicate records
- Invalid dates
- Negative activity values
- Invalid building values
- Invalid numeric values (strings, infinity, fractional headcounts)
- Policy enforcement: Suspicious values are NOT silently repaired
- ValidationReport metrics: total, valid, rejected, completeness %, error breakdown
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_campus_activity_data
from src.data_validation import (
    DEFAULT_ALLOWED_BUILDINGS,
    ERROR_TYPE_DUPLICATES,
    ERROR_TYPE_INVALID_BUILDING,
    ERROR_TYPE_INVALID_DATES,
    ERROR_TYPE_INVALID_NUMERIC,
    ERROR_TYPE_MISSING_COLUMNS,
    ERROR_TYPE_MISSING_VALUES,
    ERROR_TYPE_NEGATIVE_VALUES,
    REQUIRED_COLUMNS,
    DataValidationError,
    ValidationReport,
    validate_activity_dataframe,
    validate_activity_file,
)


@pytest.fixture
def clean_sample_df() -> pd.DataFrame:
    """Fixture providing clean 10-row campus activity data."""
    df = generate_campus_activity_data(
        start_date="2023-01-01",
        end_date="2023-02-01",
        num_buildings=5,
        seed=42,
    )
    return df.copy()


def test_clean_data_passes_validation(clean_sample_df: pd.DataFrame):
    """Verify clean dataset has 100% completeness and zero errors."""
    cleaned_df, report = validate_activity_dataframe(clean_sample_df)

    assert isinstance(report, ValidationReport)
    assert report.total_records == len(clean_sample_df)
    assert report.valid_records == len(clean_sample_df)
    assert report.rejected_records == 0
    assert report.completeness_percentage == 100.0
    assert len(cleaned_df) == len(clean_sample_df)
    for err_type, count in report.errors_by_type.items():
        assert count == 0


def test_detects_missing_required_columns(clean_sample_df: pd.DataFrame):
    """Verify detection of missing required columns."""
    df_missing = clean_sample_df.drop(columns=["electricity_kwh", "waste_kg"])

    # Without raising
    cleaned_df, report = validate_activity_dataframe(
        df_missing, raise_on_missing_columns=False
    )
    assert report.total_records == len(clean_sample_df)
    assert report.valid_records == 0
    assert report.rejected_records == len(clean_sample_df)
    assert report.completeness_percentage == 0.0
    assert report.errors_by_type[ERROR_TYPE_MISSING_COLUMNS] == 2
    assert len(cleaned_df) == 0

    # With raising
    with pytest.raises(DataValidationError, match="missing required column"):
        validate_activity_dataframe(df_missing, raise_on_missing_columns=True)


def test_detects_missing_values(clean_sample_df: pd.DataFrame):
    """Verify detection of NaN, null, and empty string missing values."""
    df_corrupted = clean_sample_df.copy()
    # Inject NaN in numeric column
    df_corrupted.loc[0, "electricity_kwh"] = np.nan
    # Inject empty string in date
    df_corrupted.loc[1, "date"] = "   "
    # Inject None in building
    df_corrupted.loc[2, "building"] = None

    cleaned_df, report = validate_activity_dataframe(df_corrupted)

    assert report.total_records == 10
    assert report.rejected_records == 3
    assert report.valid_records == 7
    assert report.errors_by_type[ERROR_TYPE_MISSING_VALUES] >= 2
    assert report.errors_by_type[ERROR_TYPE_INVALID_BUILDING] >= 1
    # Check that rejected rows' corruptions are NOT in cleaned_df
    assert not cleaned_df["electricity_kwh"].isna().any()
    assert not (cleaned_df["date"].str.strip() == "").any()
    assert not cleaned_df["building"].isna().any()
    assert len(cleaned_df) == 7


def test_detects_duplicate_records(clean_sample_df: pd.DataFrame):
    """Verify detection of duplicate (date, building) entries."""
    df_duplicates = clean_sample_df.copy()
    # Duplicate the first row at the bottom
    dup_row = df_duplicates.iloc[[0]].copy()
    df_with_dups = pd.concat([df_duplicates, dup_row], ignore_index=True)

    cleaned_df, report = validate_activity_dataframe(df_with_dups)

    assert report.total_records == 11
    assert report.valid_records == 10
    assert report.rejected_records == 1
    assert report.errors_by_type[ERROR_TYPE_DUPLICATES] == 1
    assert len(cleaned_df) == 10


def test_detects_invalid_dates(clean_sample_df: pd.DataFrame):
    """Verify detection of unparseable dates, impossible calendar dates, and out-of-range years."""
    df_bad_dates = clean_sample_df.copy()
    df_bad_dates.loc[0, "date"] = "2023-02-31"  # Impossible day
    df_bad_dates.loc[1, "date"] = "not-a-date"  # Non-date string
    df_bad_dates.loc[2, "date"] = "1975-01-01"  # Year out of campus operational range

    cleaned_df, report = validate_activity_dataframe(df_bad_dates)

    assert report.errors_by_type[ERROR_TYPE_INVALID_DATES] == 3
    assert report.rejected_records >= 3
    assert report.completeness_percentage < 100.0


def test_detects_negative_activity_values(clean_sample_df: pd.DataFrame):
    """Verify detection of physical non-negativity violations across activity columns."""
    df_negative = clean_sample_df.copy()
    df_negative.loc[0, "electricity_kwh"] = -150.0
    df_negative.loc[1, "travel_km"] = -10.5
    df_negative.loc[2, "waste_kg"] = -5.0
    df_negative.loc[3, "procurement_inr"] = -1000.0
    df_negative.loc[4, "student_count"] = -25

    cleaned_df, report = validate_activity_dataframe(df_negative)

    assert report.errors_by_type[ERROR_TYPE_NEGATIVE_VALUES] == 5
    assert report.rejected_records == 5
    assert report.valid_records == 5


def test_detects_invalid_building_values(clean_sample_df: pd.DataFrame):
    """Verify detection of unknown, purely numeric, and empty building names."""
    df_bad_bldg = clean_sample_df.copy()
    # Unknown building not in whitelist
    df_bad_bldg.loc[0, "building"] = "Offsite Corporate Warehouse"
    # Pure numeric building name
    df_bad_bldg.loc[1, "building"] = "99999"
    # Whitespace-only building
    df_bad_bldg.loc[2, "building"] = "   "

    cleaned_df, report = validate_activity_dataframe(df_bad_bldg)

    assert report.errors_by_type[ERROR_TYPE_INVALID_BUILDING] == 3
    assert report.rejected_records >= 3


def test_custom_allowed_buildings(clean_sample_df: pd.DataFrame):
    """Verify custom building whitelist can be specified."""
    custom_allowed = {"Custom Building A", "Custom Building B"}
    df_custom = clean_sample_df.copy()
    df_custom.loc[0, "building"] = "Custom Building A"

    cleaned_df, report = validate_activity_dataframe(
        df_custom,
        allowed_buildings=custom_allowed,
        enforce_known_buildings=True,
    )
    # Only row 0 matches custom whitelist
    assert report.valid_records == 1
    assert report.rejected_records == 9


def test_detects_invalid_numeric_values(clean_sample_df: pd.DataFrame):
    """Verify detection of non-numeric strings, infinity, and fractional headcounts."""
    df_invalid_num = clean_sample_df.astype(object).copy()
    # Non-numeric string in float column
    df_invalid_num.loc[0, "electricity_kwh"] = "forty-two-thousand"
    # Infinity in float column
    df_invalid_num.loc[1, "procurement_inr"] = np.inf
    # Fractional student count
    df_invalid_num.loc[2, "student_count"] = 512.75
    # Fractional staff count
    df_invalid_num.loc[3, "staff_count"] = 40.5

    cleaned_df, report = validate_activity_dataframe(df_invalid_num)

    assert report.errors_by_type[ERROR_TYPE_INVALID_NUMERIC] == 4
    assert report.rejected_records == 4
    assert report.valid_records == 6


def test_no_silent_repair_policy(clean_sample_df: pd.DataFrame):
    """
    Verify policy requirement: Suspicious values are NOT silently repaired.
    Negative values must NOT be converted to absolute values;
    Fractional headcounts must NOT be rounded and kept;
    Missing values must NOT be imputed.
    """
    df_suspicious = clean_sample_df.astype(object).copy()
    df_suspicious.loc[0, "electricity_kwh"] = -25000.0  # Should NOT become 25000.0
    df_suspicious.loc[1, "student_count"] = 800.4       # Should NOT become 800
    df_suspicious.loc[2, "waste_kg"] = np.nan           # Should NOT be filled with mean

    cleaned_df, report = validate_activity_dataframe(df_suspicious)

    # All 3 suspicious rows must be rejected, not repaired
    assert report.rejected_records == 3
    assert len(cleaned_df) == len(clean_sample_df) - 3

    # Ensure none of the suspicious values exist in the cleaned dataset
    assert not (cleaned_df["electricity_kwh"] == 25000.0).any()
    assert not (cleaned_df["electricity_kwh"] < 0).any()
    assert not cleaned_df["student_count"].isin([800.4, 800.0]).any()


def test_multi_error_reporting(clean_sample_df: pd.DataFrame):
    """Verify a row with multiple distinct errors records all error types in report."""
    df_multi = clean_sample_df.astype(object).copy()
    # Row 0 has invalid date, invalid building, negative electricity, and fractional student count
    df_multi.loc[0, "date"] = "invalid-date"
    df_multi.loc[0, "building"] = "Unknown Facility"
    df_multi.loc[0, "electricity_kwh"] = -100.0
    df_multi.loc[0, "student_count"] = 12.5

    cleaned_df, report = validate_activity_dataframe(df_multi)

    assert report.rejected_records == 1
    assert report.errors_by_type[ERROR_TYPE_INVALID_DATES] >= 1
    assert report.errors_by_type[ERROR_TYPE_INVALID_BUILDING] >= 1
    assert report.errors_by_type[ERROR_TYPE_NEGATIVE_VALUES] >= 1
    assert report.errors_by_type[ERROR_TYPE_INVALID_NUMERIC] >= 1

    # Report serialization
    report_dict = report.to_dict()
    assert report_dict["total_records"] == 10
    assert report_dict["rejected_records"] == 1
    assert "error_details" in report_dict
    assert len(report.summary()) > 100


def test_logging_captures_rejections(clean_sample_df: pd.DataFrame, caplog: pytest.LogCaptureFixture):
    """Verify validation rejections and decisions are explicitly logged."""
    df_corrupted = clean_sample_df.copy()
    df_corrupted.loc[0, "electricity_kwh"] = -999.0

    with caplog.at_level(logging.WARNING, logger="campus_carbon.data_validation"):
        cleaned_df, report = validate_activity_dataframe(df_corrupted)

    assert any("Row 0 REJECTED" in record.message for record in caplog.records)
    assert any("negative_activity_values" in record.message for record in caplog.records)


def test_validate_activity_file_e2e(tmp_path: Path):
    """Verify file-based ingestion, cleaning, reporting, and CSV writing."""
    raw_csv = tmp_path / "raw_activity.csv"
    proc_csv = tmp_path / "processed" / "cleaned_activity.csv"

    # Generate 15 sample rows with 1 corrupted row
    df = generate_campus_activity_data(
        start_date="2023-01-01",
        end_date="2023-03-01",
        num_buildings=5,
        seed=42,
    )
    df.loc[3, "waste_kg"] = -10.0  # Corrupted row
    df.to_csv(raw_csv, index=False)

    cleaned_df, report = validate_activity_file(
        input_path=raw_csv,
        output_path=proc_csv,
    )

    assert proc_csv.exists()
    assert report.total_records == 15
    assert report.valid_records == 14
    assert report.rejected_records == 1
    assert report.completeness_percentage == round(14 / 15 * 100.0, 2)

    loaded_df = pd.read_csv(proc_csv)
    assert len(loaded_df) == 14
    assert (loaded_df["waste_kg"] >= 0).all()
