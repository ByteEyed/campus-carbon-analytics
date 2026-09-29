"""
Campus Activity Data Validation Pipeline
========================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

This module implements a rigorous, transparent data validation pipeline for
campus activity data. In compliance with PROJECT_SPEC.md:
- Detects missing required columns, missing values, duplicate records,
  invalid dates, negative activity values, invalid building values, and invalid numeric values.
- Never silently repairs, clips, or imputes suspicious/corrupted values.
- Explicitly logs all validation decisions and rejections with row-level details.
- Produces a comprehensive ValidationReport tracking total, valid, rejected records,
  completeness percentage, and errors by type.
- Writes validated clean data to data/processed/cleaned_activity.csv.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# Configure module-level logger
logger = logging.getLogger("campus_carbon.data_validation")

# Standard required columns for campus activity data
REQUIRED_COLUMNS: list[str] = [
    "date",
    "building",
    "electricity_kwh",
    "travel_km",
    "waste_kg",
    "procurement_inr",
    "student_count",
    "staff_count",
]

# Numeric continuous activity columns
NUMERIC_FLOAT_COLUMNS: list[str] = [
    "electricity_kwh",
    "travel_km",
    "waste_kg",
    "procurement_inr",
]

# Numeric discrete count columns
NUMERIC_COUNT_COLUMNS: list[str] = [
    "student_count",
    "staff_count",
]

ALL_NUMERIC_COLUMNS: list[str] = NUMERIC_FLOAT_COLUMNS + NUMERIC_COUNT_COLUMNS

# Standard campus building archetypes
DEFAULT_ALLOWED_BUILDINGS: set[str] = {
    "Science & Engineering Complex",
    "Academic Block - Arts & Humanities",
    "Central Library & Student Hub",
    "Administrative Headquarters",
    "Hostel & Dining Complex",
}

# Standard validation error categories
ERROR_TYPE_MISSING_COLUMNS = "missing_required_columns"
ERROR_TYPE_MISSING_VALUES = "missing_values"
ERROR_TYPE_DUPLICATES = "duplicate_records"
ERROR_TYPE_INVALID_DATES = "invalid_dates"
ERROR_TYPE_NEGATIVE_VALUES = "negative_activity_values"
ERROR_TYPE_INVALID_BUILDING = "invalid_building_values"
ERROR_TYPE_INVALID_NUMERIC = "invalid_numeric_values"

ALL_ERROR_TYPES: list[str] = [
    ERROR_TYPE_MISSING_COLUMNS,
    ERROR_TYPE_MISSING_VALUES,
    ERROR_TYPE_DUPLICATES,
    ERROR_TYPE_INVALID_DATES,
    ERROR_TYPE_NEGATIVE_VALUES,
    ERROR_TYPE_INVALID_BUILDING,
    ERROR_TYPE_INVALID_NUMERIC,
]

DATE_REGEX = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")


class DataValidationError(Exception):
    """Exception raised when a critical dataset schema error prevents validation."""


@dataclass
class ValidationErrorDetail:
    """Detailed record of a validation failure."""

    row_index: int
    column: str
    error_type: str
    message: str
    raw_value: Any = None
    date: Any = None
    building: Any = None


@dataclass
class ValidationReport:
    """Summary and audit report of data validation results."""

    total_records: int
    valid_records: int
    rejected_records: int
    completeness_percentage: float
    errors_by_type: dict[str, int] = field(default_factory=dict)
    error_details: list[ValidationErrorDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "rejected_records": self.rejected_records,
            "completeness_percentage": self.completeness_percentage,
            "errors_by_type": dict(self.errors_by_type),
            "total_errors": len(self.error_details),
            "error_details": [asdict(e) for e in self.error_details],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return human-readable multi-line summary of validation results."""
        lines = [
            "=================================================================",
            "DATA VALIDATION REPORT",
            "=================================================================",
            f"Total Records Evaluated : {self.total_records}",
            f"Valid Records Accepted  : {self.valid_records}",
            f"Rejected Records        : {self.rejected_records}",
            f"Completeness Percentage : {self.completeness_percentage:.2f}%",
            "-----------------------------------------------------------------",
            "Validation Errors by Type:",
        ]
        if not self.errors_by_type or sum(self.errors_by_type.values()) == 0:
            lines.append("  (No errors detected)")
        else:
            for err_type, count in sorted(self.errors_by_type.items()):
                lines.append(f"  - {err_type}: {count}")

        if self.error_details:
            lines.append("-----------------------------------------------------------------")
            lines.append(f"Sample Rejected Records (First {min(5, len(self.error_details))} of {len(self.error_details)}):")
            for detail in self.error_details[:5]:
                lines.append(
                    f"  [Row {detail.row_index}] Col: '{detail.column}' | "
                    f"Type: {detail.error_type} | Val: {detail.raw_value!r} | "
                    f"Reason: {detail.message}"
                )
        lines.append("=================================================================")
        return "\n".join(lines)


def _is_empty_or_null(val: Any) -> bool:
    """Check if value is null, NaN, or whitespace-only string."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, str) and val.strip().lower() in ("nan", "none", "null", "nat"):
        return True
    return False


def _validate_date_value(val: Any) -> tuple[bool, str]:
    """Validate date format (YYYY-MM-DD) and logical calendar bounds."""
    if _is_empty_or_null(val):
        return False, "Date value is missing or empty."

    val_str = str(val).strip()
    if not DATE_REGEX.match(val_str):
        return False, f"Date '{val_str}' does not strictly match ISO YYYY-MM-DD pattern."

    try:
        parsed = pd.to_datetime(val_str, format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        return False, f"Date '{val_str}' cannot be parsed into a calendar date: {exc}"

    if parsed.year < 2000 or parsed.year > 2100:
        return False, f"Date year {parsed.year} is outside logical campus operational range (2000-2100)."

    return True, ""


def validate_activity_dataframe(
    df: pd.DataFrame,
    allowed_buildings: Iterable[str] | None = None,
    enforce_known_buildings: bool = True,
    raise_on_missing_columns: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate campus activity data and partition into accepted clean records and audit report.

    Suspicious or corrupted records are NEVER silently repaired (no clipping, no imputation).
    Every rejection is logged with exact row indices and diagnostic reasons.

    Parameters
    ----------
    df : pd.DataFrame
        Input campus activity dataframe.
    allowed_buildings : Iterable[str] | None, default None
        Set or list of valid building names. Defaults to DEFAULT_ALLOWED_BUILDINGS.
    enforce_known_buildings : bool, default True
        If True, rejects building names not in allowed_buildings.
        If False, accepts any non-empty string building name.
    raise_on_missing_columns : bool, default False
        If True, immediately raises DataValidationError when required columns are absent.

    Returns
    -------
    tuple[pd.DataFrame, ValidationReport]
        - Cleaned, validated DataFrame containing strictly valid records.
        - ValidationReport detailing metrics, error frequencies, and rejection audit logs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    logger.info("Starting campus activity data validation pipeline on %d records.", len(df))

    # Initialize error counters
    errors_by_type: dict[str, int] = {err_type: 0 for err_type in ALL_ERROR_TYPES}
    error_details: list[ValidationErrorDetail] = []
    rejected_row_indices: set[int] = set()

    total_records = len(df)

    # 1. Check Missing Required Columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Dataset is missing required column(s): {missing_cols}"
        logger.error(msg)
        errors_by_type[ERROR_TYPE_MISSING_COLUMNS] += len(missing_cols)
        for col in missing_cols:
            error_details.append(
                ValidationErrorDetail(
                    row_index=-1,
                    column=col,
                    error_type=ERROR_TYPE_MISSING_COLUMNS,
                    message=f"Required column '{col}' is entirely absent from dataset schema.",
                )
            )
        if raise_on_missing_columns:
            raise DataValidationError(msg)

        # If schema is broken, all rows are rejected
        empty_cleaned = pd.DataFrame(columns=[c for c in REQUIRED_COLUMNS if c in df.columns])
        report = ValidationReport(
            total_records=total_records,
            valid_records=0,
            rejected_records=total_records,
            completeness_percentage=0.0,
            errors_by_type=errors_by_type,
            error_details=error_details,
        )
        return empty_cleaned, report

    # Resolve allowed buildings
    valid_building_set: set[str] = (
        set(allowed_buildings) if allowed_buildings is not None else DEFAULT_ALLOWED_BUILDINGS
    )

    # 2. Check for Duplicate Records across (date, building)
    # The first occurrence is preserved; subsequent occurrences are marked duplicate.
    duplicate_mask = df.duplicated(subset=["date", "building"], keep="first")
    for idx in df[duplicate_mask].index:
        rejected_row_indices.add(idx)
        errors_by_type[ERROR_TYPE_DUPLICATES] += 1
        d_val = df.loc[idx, "date"]
        b_val = df.loc[idx, "building"]
        msg = f"Duplicate record detected for date '{d_val}' and building '{b_val}'."
        logger.warning("Row %d REJECTED: [duplicate_records] %s", idx, msg)
        error_details.append(
            ValidationErrorDetail(
                row_index=idx,
                column="date,building",
                error_type=ERROR_TYPE_DUPLICATES,
                message=msg,
                raw_value=(d_val, b_val),
                date=d_val,
                building=b_val,
            )
        )

    # 3. Row-by-Row Schema and Constraint Validation
    for idx, row in df.iterrows():
        row_has_error = idx in rejected_row_indices
        date_raw = row["date"]
        building_raw = row["building"]

        # 3a. Check Building Validity
        if _is_empty_or_null(building_raw):
            row_has_error = True
            errors_by_type[ERROR_TYPE_INVALID_BUILDING] += 1
            msg = "Building value is null, missing, or empty."
            logger.warning("Row %d REJECTED: [invalid_building_values] %s", idx, msg)
            error_details.append(
                ValidationErrorDetail(
                    row_index=idx,
                    column="building",
                    error_type=ERROR_TYPE_INVALID_BUILDING,
                    message=msg,
                    raw_value=building_raw,
                    date=date_raw,
                    building=building_raw,
                )
            )
        else:
            b_str = str(building_raw).strip()
            # Must not be purely numeric
            if b_str.isdigit():
                row_has_error = True
                errors_by_type[ERROR_TYPE_INVALID_BUILDING] += 1
                msg = f"Building value '{b_str}' is purely numeric; expected institutional facility name."
                logger.warning("Row %d REJECTED: [invalid_building_values] %s", idx, msg)
                error_details.append(
                    ValidationErrorDetail(
                        row_index=idx,
                        column="building",
                        error_type=ERROR_TYPE_INVALID_BUILDING,
                        message=msg,
                        raw_value=building_raw,
                        date=date_raw,
                        building=b_str,
                    )
                )
            elif enforce_known_buildings and b_str not in valid_building_set:
                row_has_error = True
                errors_by_type[ERROR_TYPE_INVALID_BUILDING] += 1
                msg = f"Building '{b_str}' is not in the recognized campus facilities: {sorted(valid_building_set)}"
                logger.warning("Row %d REJECTED: [invalid_building_values] %s", idx, msg)
                error_details.append(
                    ValidationErrorDetail(
                        row_index=idx,
                        column="building",
                        error_type=ERROR_TYPE_INVALID_BUILDING,
                        message=msg,
                        raw_value=building_raw,
                        date=date_raw,
                        building=b_str,
                    )
                )

        # 3b. Check Date Validity
        is_date_valid, date_err_msg = _validate_date_value(date_raw)
        if not is_date_valid:
            row_has_error = True
            if _is_empty_or_null(date_raw):
                errors_by_type[ERROR_TYPE_MISSING_VALUES] += 1
                err_kind = ERROR_TYPE_MISSING_VALUES
            else:
                errors_by_type[ERROR_TYPE_INVALID_DATES] += 1
                err_kind = ERROR_TYPE_INVALID_DATES
            logger.warning("Row %d REJECTED: [%s] %s", idx, err_kind, date_err_msg)
            error_details.append(
                ValidationErrorDetail(
                    row_index=idx,
                    column="date",
                    error_type=err_kind,
                    message=date_err_msg,
                    raw_value=date_raw,
                    date=date_raw,
                    building=building_raw,
                )
            )

        # 3c. Check Numeric Columns
        for num_col in ALL_NUMERIC_COLUMNS:
            raw_num = row[num_col]

            # Missing value check
            if _is_empty_or_null(raw_num):
                row_has_error = True
                errors_by_type[ERROR_TYPE_MISSING_VALUES] += 1
                msg = f"Missing value in required numeric column '{num_col}'."
                logger.warning("Row %d REJECTED: [missing_values] %s", idx, msg)
                error_details.append(
                    ValidationErrorDetail(
                        row_index=idx,
                        column=num_col,
                        error_type=ERROR_TYPE_MISSING_VALUES,
                        message=msg,
                        raw_value=raw_num,
                        date=date_raw,
                        building=building_raw,
                    )
                )
                continue

            # Parsing to float
            try:
                num_val = float(raw_num)
            except (ValueError, TypeError):
                row_has_error = True
                errors_by_type[ERROR_TYPE_INVALID_NUMERIC] += 1
                msg = f"Value '{raw_num}' in column '{num_col}' cannot be parsed as a numeric float."
                logger.warning("Row %d REJECTED: [invalid_numeric_values] %s", idx, msg)
                error_details.append(
                    ValidationErrorDetail(
                        row_index=idx,
                        column=num_col,
                        error_type=ERROR_TYPE_INVALID_NUMERIC,
                        message=msg,
                        raw_value=raw_num,
                        date=date_raw,
                        building=building_raw,
                    )
                )
                continue

            # Infinity check
            if np.isinf(num_val) or np.isnan(num_val):
                row_has_error = True
                errors_by_type[ERROR_TYPE_INVALID_NUMERIC] += 1
                msg = f"Non-finite value '{raw_num}' in column '{num_col}'."
                logger.warning("Row %d REJECTED: [invalid_numeric_values] %s", idx, msg)
                error_details.append(
                    ValidationErrorDetail(
                        row_index=idx,
                        column=num_col,
                        error_type=ERROR_TYPE_INVALID_NUMERIC,
                        message=msg,
                        raw_value=raw_num,
                        date=date_raw,
                        building=building_raw,
                    )
                )
                continue

            # Negative value check
            if num_val < 0:
                row_has_error = True
                errors_by_type[ERROR_TYPE_NEGATIVE_VALUES] += 1
                msg = (
                    f"Physical non-negativity violation: column '{num_col}' has "
                    f"negative value {num_val}. Suspicious values are NOT repaired."
                )
                logger.warning("Row %d REJECTED: [negative_activity_values] %s", idx, msg)
                error_details.append(
                    ValidationErrorDetail(
                        row_index=idx,
                        column=num_col,
                        error_type=ERROR_TYPE_NEGATIVE_VALUES,
                        message=msg,
                        raw_value=raw_num,
                        date=date_raw,
                        building=building_raw,
                    )
                )

            # Integer constraint for headcount columns
            if num_col in NUMERIC_COUNT_COLUMNS:
                if num_val % 1 != 0:
                    row_has_error = True
                    errors_by_type[ERROR_TYPE_INVALID_NUMERIC] += 1
                    msg = (
                        f"Headcount column '{num_col}' must be a whole integer, "
                        f"got fractional value {num_val}."
                    )
                    logger.warning("Row %d REJECTED: [invalid_numeric_values] %s", idx, msg)
                    error_details.append(
                        ValidationErrorDetail(
                            row_index=idx,
                            column=num_col,
                            error_type=ERROR_TYPE_INVALID_NUMERIC,
                            message=msg,
                            raw_value=raw_num,
                            date=date_raw,
                            building=building_raw,
                        )
                    )

        if row_has_error:
            rejected_row_indices.add(idx)

    # 4. Construct Cleaned DataFrame
    valid_indices = [i for i in df.index if i not in rejected_row_indices]
    cleaned_df = df.loc[valid_indices].copy()

    # Enforce correct data types on clean dataframe without altering values
    if not cleaned_df.empty:
        cleaned_df["date"] = cleaned_df["date"].astype(str)
        cleaned_df["building"] = cleaned_df["building"].astype(str)
        for c in NUMERIC_FLOAT_COLUMNS:
            cleaned_df[c] = cleaned_df[c].astype(float)
        for c in NUMERIC_COUNT_COLUMNS:
            cleaned_df[c] = cleaned_df[c].astype(int)

        # Sort cleanly by date and building
        cleaned_df = cleaned_df.sort_values(by=["date", "building"]).reset_index(drop=True)

    # 5. Compile Validation Report
    valid_count = len(cleaned_df)
    rejected_count = total_records - valid_count
    completeness_pct = (valid_count / total_records * 100.0) if total_records > 0 else 0.0

    report = ValidationReport(
        total_records=total_records,
        valid_records=valid_count,
        rejected_records=rejected_count,
        completeness_percentage=round(completeness_pct, 2),
        errors_by_type=errors_by_type,
        error_details=error_details,
    )

    logger.info(
        "Validation complete: %d valid records, %d rejected records (%.2f%% completeness).",
        valid_count,
        rejected_count,
        completeness_pct,
    )

    return cleaned_df, report


def validate_activity_file(
    input_path: str | Path = "data/raw/campus_activity.csv",
    output_path: str | Path | None = "data/processed/cleaned_activity.csv",
    allowed_buildings: Iterable[str] | None = None,
    enforce_known_buildings: bool = True,
    raise_on_missing_columns: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Ingest, validate, and export clean campus activity data from CSV.

    Parameters
    ----------
    input_path : str | Path
        Path to raw CSV file.
    output_path : str | Path | None
        Optional destination for cleaned CSV file. If None, file is not written.
    allowed_buildings : Iterable[str] | None
        Optional collection of accepted building names.
    enforce_known_buildings : bool
        Whether to enforce recognized building names.
    raise_on_missing_columns : bool
        Whether to raise an exception on missing columns.

    Returns
    -------
    tuple[pd.DataFrame, ValidationReport]
        Cleaned dataframe and detailed validation report.
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found at: {in_path.resolve()}")

    logger.info("Loading activity data from %s", in_path.resolve())
    df = pd.read_csv(in_path)

    cleaned_df, report = validate_activity_dataframe(
        df=df,
        allowed_buildings=allowed_buildings,
        enforce_known_buildings=enforce_known_buildings,
        raise_on_missing_columns=raise_on_missing_columns,
    )

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(out_path, index=False)
        logger.info("Saved %d cleaned records to %s", len(cleaned_df), out_path.resolve())

    return cleaned_df, report


def main() -> None:
    """CLI entrypoint for campus activity data validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Validate and clean campus activity data for BDS-36 Carbon Analytics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/campus_activity.csv",
        help="Path to input raw CSV file (default: data/raw/campus_activity.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/cleaned_activity.csv",
        help="Path to output cleaned CSV file (default: data/processed/cleaned_activity.csv)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure console logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=================================================================")
    print("CAMPUS ACTIVITY DATA VALIDATION PIPELINE")
    print("=================================================================")
    print(f"Source Input : {args.input}")
    print(f"Target Output: {args.output}")
    print("Policy       : Suspicious values are NOT silently repaired.")
    print("-----------------------------------------------------------------")

    cleaned_df, report = validate_activity_file(
        input_path=args.input,
        output_path=args.output,
    )

    print("\n" + report.summary())
    print(f"\nCleaned dataset saved with {len(cleaned_df)} records to: {args.output}")


if __name__ == "__main__":
    main()
