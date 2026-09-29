"""
Campus Carbon Accounting Engine
===============================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

This module implements the carbon accounting engine and emission factor registry.
In accordance with PROJECT_SPEC.md:
- Emission factors are documented, externally sourced, and strictly separated from simulated activity data.
- Pure function `calculate_emissions(activity_value, emission_factor)` calculates kgCO2e.
- Category-level aggregations produce:
  1. Emissions by date (monthly campus time series)
  2. Emissions by category
  3. Total emissions (kgCO2e & MTCO2e / metric tonnes)
  4. Emissions per student
  5. Category contribution percentages
- All emission factors remain fully configurable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("campus_carbon.accounting")

# Standard primary categories
CATEGORY_ELECTRICITY = "Electricity"
CATEGORY_TRAVEL = "Travel"
CATEGORY_WASTE = "Waste"
CATEGORY_PROCUREMENT = "Procurement"

PRIMARY_CATEGORIES: list[str] = [
    CATEGORY_ELECTRICITY,
    CATEGORY_TRAVEL,
    CATEGORY_WASTE,
    CATEGORY_PROCUREMENT,
]

# Mapping between category and default activity column name
CATEGORY_TO_ACTIVITY_COL: dict[str, str] = {
    CATEGORY_ELECTRICITY: "electricity_kwh",
    CATEGORY_TRAVEL: "travel_km",
    CATEGORY_WASTE: "waste_kg",
    CATEGORY_PROCUREMENT: "procurement_inr",
}

ACTIVITY_COL_TO_CATEGORY: dict[str, str] = {
    v: k for k, v in CATEGORY_TO_ACTIVITY_COL.items()
}

# Recognized standard units per category to prevent 1000x magnitude mismatches
EXPECTED_CATEGORY_UNITS: dict[str, str] = {
    CATEGORY_ELECTRICITY: "kgCO2e/kWh",
    CATEGORY_TRAVEL: "kgCO2e/km",
    CATEGORY_WASTE: "kgCO2e/kg",
    CATEGORY_PROCUREMENT: "kgCO2e/INR",
}

# Canonical path to emission factors CSV
DEFAULT_FACTORS_CSV_PATH = Path("data/emission_factors.csv")


@dataclass(frozen=True)
class EmissionFactor:
    """Documented emission factor record with genuine source provenance."""

    category: str
    activity_type: str
    unit: str
    factor: float
    source: str
    version: str

    def __post_init__(self) -> None:
        """Validate emission factor fields and physical constraints."""
        if not self.category or not str(self.category).strip():
            raise ValueError("EmissionFactor 'category' cannot be empty.")
        if not self.activity_type or not str(self.activity_type).strip():
            raise ValueError("EmissionFactor 'activity_type' cannot be empty.")
        if not self.unit or not str(self.unit).strip():
            raise ValueError("EmissionFactor 'unit' cannot be empty.")
        if not self.source or not str(self.source).strip():
            raise ValueError("EmissionFactor 'source' cannot be empty.")
        if not self.version or not str(self.version).strip():
            raise ValueError("EmissionFactor 'version' cannot be empty.")

        try:
            num_factor = float(self.factor)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Emission factor value {self.factor!r} is not a valid float.") from exc

        if np.isnan(num_factor) or np.isinf(num_factor):
            raise ValueError(f"Emission factor value cannot be NaN or Infinite, got {self.factor}.")
        if num_factor < 0:
            raise ValueError(f"Emission factor cannot be negative, got {self.factor}.")

        # Check unit matching for standard categories
        norm_cat = str(self.category).strip().title()
        if norm_cat in EXPECTED_CATEGORY_UNITS:
            expected_unit = EXPECTED_CATEGORY_UNITS[norm_cat]
            if str(self.unit).strip() != expected_unit:
                raise ValueError(
                    f"Unit mismatch for category '{self.category}': "
                    f"expected '{expected_unit}', got '{self.unit}'."
                )

        if num_factor == 0.0:
            logger.warning(
                "EmissionFactor for category '%s' is registered as 0.0. "
                "This implies zero emissions for this activity.",
                self.category,
            )


class EmissionFactorRegistry:
    """
    Configurable registry storing and managing documented emission factors.

    Note on Unit Validation:
    Unit validation between registered emission factor units (e.g. kgCO2e/kWh)
    and activity data columns is currently the responsibility of the caller
    and relies on the strict schema defined in DATA_DICTIONARY.md.
    """

    def __init__(self, factors: Sequence[EmissionFactor] | None = None) -> None:
        self._factors_by_key: dict[str, EmissionFactor] = {}
        if factors is not None:
            for factor in factors:
                self.register(factor)

    def register(self, factor: EmissionFactor) -> None:
        """Register or update an emission factor."""
        if not isinstance(factor, EmissionFactor):
            raise TypeError(f"Expected EmissionFactor instance, got {type(factor).__name__}")
        # Index by both normalized category and activity_type for flexible lookup
        cat_key = factor.category.strip().lower()
        act_key = factor.activity_type.strip().lower()
        self._factors_by_key[cat_key] = factor
        self._factors_by_key[act_key] = factor

    def get_factor(self, key: str) -> EmissionFactor:
        """Retrieve an emission factor by category or activity type."""
        lookup_key = str(key).strip().lower()
        if lookup_key not in self._factors_by_key:
            available = sorted(set(f.category for f in self._factors_by_key.values()))
            raise KeyError(
                f"Emission factor for '{key}' is missing or not registered. "
                f"Available registered categories: {available}"
            )
        return self._factors_by_key[lookup_key]

    def get_factor_value(self, key: str) -> float:
        """Retrieve the numerical factor value by category or activity type."""
        return self.get_factor(key).factor

    def list_factors(self) -> list[EmissionFactor]:
        """Return unique list of registered emission factors."""
        unique_factors: dict[tuple[str, str], EmissionFactor] = {}
        for f in self._factors_by_key.values():
            unique_factors[(f.category, f.activity_type)] = f
        return list(unique_factors.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Export registry as a pandas DataFrame."""
        records = [asdict(f) for f in self.list_factors()]
        return pd.DataFrame(records)

    def to_csv(self, path: str | Path) -> Path:
        """Save emission factors registry to CSV."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(out_path, index=False)
        return out_path

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> EmissionFactorRegistry:
        """Instantiate registry from a DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("Emission factor dataframe cannot be empty.")

        required = {"category", "activity_type", "unit", "factor", "source", "version"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Emission factor dataframe is missing required column(s): {missing}")

        factors = []
        for _, row in df.iterrows():
            factor = EmissionFactor(
                category=str(row["category"]),
                activity_type=str(row["activity_type"]),
                unit=str(row["unit"]),
                factor=float(row["factor"]),
                source=str(row["source"]),
                version=str(row["version"]),
            )
            factors.append(factor)
        return cls(factors)

    @classmethod
    def from_csv(cls, path: str | Path) -> EmissionFactorRegistry:
        """Load registry from an external CSV file."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Emission factors file not found at: {csv_path.resolve()}")
        df = pd.read_csv(csv_path)
        return cls.from_dataframe(df)

    @classmethod
    def default(cls) -> EmissionFactorRegistry:
        """
        Create registry initialized with documented emission factors loaded directly from CSV.

        Raises
        ------
        FileNotFoundError
            If the canonical emission factors file (data/emission_factors.csv) is not found.
        """
        candidates = [
            DEFAULT_FACTORS_CSV_PATH,
            Path(__file__).resolve().parent.parent / "data" / "emission_factors.csv",
        ]
        for p in candidates:
            if p.exists():
                return cls.from_csv(p)

        raise FileNotFoundError(
            f"Default emission factors CSV file not found at '{DEFAULT_FACTORS_CSV_PATH}'. "
            "Ensure data/emission_factors.csv exists."
        )


def calculate_emissions(activity_value: float, emission_factor: float) -> float:
    """
    Calculate carbon emissions in kgCO2e for an activity value and emission factor.

    Formula:
        emissions (kgCO2e) = activity_value * emission_factor

    Parameters
    ----------
    activity_value : float
        The numerical amount of the activity (e.g. kWh, km, kg, INR).
        Must be a non-negative number.
    emission_factor : float
        The emission factor in kgCO2e per activity unit.
        Must be a non-negative, finite number.

    Returns
    -------
    float
        Calculated greenhouse gas emissions in kgCO2e.

    Raises
    ------
    ValueError
        If activity_value or emission_factor is negative, missing, or non-finite.
    """
    if activity_value is None or emission_factor is None:
        raise ValueError("activity_value and emission_factor cannot be None.")

    try:
        act = float(activity_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"activity_value {activity_value!r} is not a valid number.") from exc

    try:
        ef = float(emission_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"emission_factor {emission_factor!r} is not a valid number.") from exc

    if np.isnan(act) or np.isinf(act):
        raise ValueError(f"activity_value cannot be NaN or Infinite, got {activity_value}.")
    if np.isnan(ef) or np.isinf(ef):
        raise ValueError(f"emission_factor cannot be NaN or Infinite, got {emission_factor}.")

    if act < 0:
        raise ValueError(f"Negative activity value is invalid: {act}. Emissions cannot be calculated.")
    if ef < 0:
        raise ValueError(f"Negative emission factor is invalid: {ef}.")

    if act == 0.0 or ef == 0.0:
        return 0.0

    return round(act * ef, 4)


def calculate_campus_emissions(
    df: pd.DataFrame,
    registry: EmissionFactorRegistry | None = None,
) -> pd.DataFrame:
    """
    Apply emission factors to cleaned activity data to compute category-level and total emissions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned activity DataFrame containing activity columns:
        ['electricity_kwh', 'travel_km', 'waste_kg', 'procurement_inr', 'student_count', 'staff_count'].
    registry : EmissionFactorRegistry | None, default None
        Registry supplying documented emission factors. Defaults to EmissionFactorRegistry.default().

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with emissions columns:
        - electricity_emissions_kg
        - travel_emissions_kg
        - waste_emissions_kg
        - procurement_emissions_kg
        - total_emissions_kg
        - total_emissions_mt (metric tonnes)
        - emissions_per_student_kg (for that specific facility/month)
    """
    if registry is None:
        registry = EmissionFactorRegistry.default()

    # 1. Guard against missing required activity columns
    required_cols = list(CATEGORY_TO_ACTIVITY_COL.values())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Activity DataFrame is missing required column(s): {missing}")

    emissions_df = df.copy()

    # 2. Strict validation: enforce calculate_emissions guards on DataFrame columns
    for col in required_cols:
        series = emissions_df[col]
        if series.isna().any():
            null_count = int(series.isna().sum())
            raise ValueError(f"Activity column '{col}' contains {null_count} NaN/null values.")

        try:
            numeric_series = pd.to_numeric(series, errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Activity column '{col}' contains non-numeric values: {exc}") from exc

        if np.isinf(numeric_series).any():
            raise ValueError(f"Activity column '{col}' contains Infinite values.")

        if (numeric_series < 0).any():
            neg_vals = numeric_series[numeric_series < 0].head().tolist()
            raise ValueError(f"Activity column '{col}' contains negative values: {neg_vals}")

    # 3. Retrieve factors from registry
    ef_elec = registry.get_factor_value("electricity_kwh")
    ef_travel = registry.get_factor_value("travel_km")
    ef_waste = registry.get_factor_value("waste_kg")
    ef_proc = registry.get_factor_value("procurement_inr")

    # 4. Vectorized calculation adhering to calculate_emissions formula
    emissions_df["electricity_emissions_kg"] = (
        emissions_df["electricity_kwh"].astype(float) * ef_elec
    ).round(4)
    emissions_df["travel_emissions_kg"] = (
        emissions_df["travel_km"].astype(float) * ef_travel
    ).round(4)
    emissions_df["waste_emissions_kg"] = (
        emissions_df["waste_kg"].astype(float) * ef_waste
    ).round(4)
    emissions_df["procurement_emissions_kg"] = (
        emissions_df["procurement_inr"].astype(float) * ef_proc
    ).round(4)

    emissions_df["total_emissions_kg"] = (
        emissions_df["electricity_emissions_kg"]
        + emissions_df["travel_emissions_kg"]
        + emissions_df["waste_emissions_kg"]
        + emissions_df["procurement_emissions_kg"]
    ).round(4)

    # 1 metric tonne (t) = 1,000 kg
    emissions_df["total_emissions_mt"] = (
        emissions_df["total_emissions_kg"] / 1000.0
    ).round(4)

    # Building-level emissions per student (safe division: zero if student_count is 0)
    if "student_count" in emissions_df.columns:
        students = emissions_df["student_count"].astype(float)
        emissions_df["emissions_per_student_kg"] = np.where(
            students > 0,
            (emissions_df["total_emissions_kg"] / students).round(4),
            0.0,
        )

    return emissions_df


def aggregate_emissions_by_date(emissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce monthly campus-wide emissions time series aggregated across all facilities.

    Note on per-student metrics: Student headcounts are NOT summed across buildings
    at each date, because students utilize multiple buildings (a sum would double-count
    the student denominator and dilute per-student emissions). Per-student intensity
    is computed at the campus summary level via calculate_emissions_per_student().

    Parameters
    ----------
    emissions_df : pd.DataFrame
        Emissions DataFrame produced by calculate_campus_emissions.

    Returns
    -------
    pd.DataFrame
        Aggregated monthly emissions by date.
    """
    agg_dict: dict[str, str] = {
        "electricity_emissions_kg": "sum",
        "travel_emissions_kg": "sum",
        "waste_emissions_kg": "sum",
        "procurement_emissions_kg": "sum",
        "total_emissions_kg": "sum",
        "total_emissions_mt": "sum",
    }

    by_date = emissions_df.groupby("date", as_index=False).agg(agg_dict)

    # Round numeric columns cleanly to 2 decimal places
    for c in agg_dict:
        by_date[c] = by_date[c].round(2)

    return by_date.sort_values(by="date").reset_index(drop=True)


def aggregate_emissions_by_category(emissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce total emissions and contribution percentages grouped by activity category.

    Parameters
    ----------
    emissions_df : pd.DataFrame
        Emissions DataFrame produced by calculate_campus_emissions.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: ['category', 'emissions_kg', 'emissions_mt', 'contribution_percentage'].
    """
    elec_tot = float(emissions_df["electricity_emissions_kg"].sum())
    travel_tot = float(emissions_df["travel_emissions_kg"].sum())
    waste_tot = float(emissions_df["waste_emissions_kg"].sum())
    proc_tot = float(emissions_df["procurement_emissions_kg"].sum())

    grand_total = float(emissions_df["total_emissions_kg"].sum())

    categories = [CATEGORY_ELECTRICITY, CATEGORY_TRAVEL, CATEGORY_WASTE, CATEGORY_PROCUREMENT]
    totals = [elec_tot, travel_tot, waste_tot, proc_tot]

    records = []
    for cat, tot in zip(categories, totals):
        pct = (tot / grand_total * 100.0) if grand_total > 0 else 0.0
        records.append({
            "category": cat,
            "emissions_kg": round(tot, 2),
            "emissions_mt": round(tot / 1000.0, 2),
            "contribution_percentage": round(pct, 2),
        })

    cat_df = pd.DataFrame(records)
    return cat_df.sort_values(by="emissions_kg", ascending=False).reset_index(drop=True)


def calculate_total_emissions(emissions_df: pd.DataFrame) -> float:
    """
    Calculate the grand total carbon emissions across all categories in kgCO2e.

    Parameters
    ----------
    emissions_df : pd.DataFrame
        Emissions DataFrame produced by calculate_campus_emissions.

    Returns
    -------
    float
        Grand total emissions in kgCO2e.
    """
    return round(float(emissions_df["total_emissions_kg"].sum()), 2)


def calculate_emissions_per_student(
    emissions_df: pd.DataFrame,
    unique_students: int | None = None,
) -> float:
    """
    Calculate overall campus emissions per student across the dataset.

    Formula:
        - If unique_students is provided: Total emissions (kgCO2e) / unique_students
        - Otherwise: Total emissions (kgCO2e) / Total student-months recorded

    Parameters
    ----------
    emissions_df : pd.DataFrame
        Emissions DataFrame containing 'total_emissions_kg' and 'student_count'.
    unique_students : int | None, default None
        Optional actual unique student headcount of the campus institution.

    Returns
    -------
    float
        Emissions per student in kgCO2e.
    """
    total_emissions = float(emissions_df["total_emissions_kg"].sum())
    if unique_students is not None:
        if unique_students <= 0:
            return 0.0
        return round(total_emissions / unique_students, 2)

    total_students = float(emissions_df["student_count"].sum())
    if total_students == 0:
        return 0.0
    return round(total_emissions / total_students, 2)


def calculate_category_contributions(emissions_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate percentage contribution of each category towards total emissions.

    Parameters
    ----------
    emissions_df : pd.DataFrame
        Emissions DataFrame produced by calculate_campus_emissions.

    Returns
    -------
    dict[str, float]
        Dictionary mapping category names to their percentage share (0.0 to 100.0%).
    """
    cat_df = aggregate_emissions_by_category(emissions_df)
    return dict(zip(cat_df["category"], cat_df["contribution_percentage"]))


@dataclass
class CarbonAccountingSummary:
    """
    Consolidated summary report of campus carbon accounting analysis.

    Note on Units:
    In this summary and across the analytics pipeline, 'MT' and 'MTCO2e' strictly
    denote Metric Tonnes of CO2 equivalent (1 Metric Tonne = 1,000 kgCO2e),
    avoiding any confusion with 'Million Tonnes'.
    """

    total_emissions_kg: float
    total_emissions_mt: float
    emissions_per_student_kg: float
    category_contributions: dict[str, float]
    emissions_by_category: pd.DataFrame
    emissions_by_date: pd.DataFrame

    def summary(self) -> str:
        """Return human-readable summary of carbon accounting results."""
        lines = [
            "=================================================================",
            "CAMPUS CARBON ACCOUNTING SUMMARY",
            "=================================================================",
            f"Total Historical Emissions : {self.total_emissions_kg:,.2f} kgCO2e ({self.total_emissions_mt:,.2f} MTCO2e / metric tonnes)",
            f"Emissions per Student      : {self.emissions_per_student_kg:,.2f} kgCO2e/student-month",
            "-----------------------------------------------------------------",
            "Category Contribution Breakdown:",
        ]
        for _, row in self.emissions_by_category.iterrows():
            lines.append(
                f"  - {row['category']:<12}: {row['emissions_kg']:>12,.2f} kgCO2e "
                f"({row['emissions_mt']:>8,.2f} MT) | {row['contribution_percentage']:>5.2f}%"
            )
        lines.append("=================================================================")
        return "\n".join(lines)


def run_carbon_accounting(
    activity_path: str | Path = "data/processed/cleaned_activity.csv",
    factors_path: str | Path = "data/emission_factors.csv",
    output_path: str | Path | None = "data/processed/campus_emissions.csv",
    strict_factors: bool = False,
) -> tuple[pd.DataFrame, CarbonAccountingSummary]:
    """
    End-to-end execution of carbon accounting pipeline.

    Loads cleaned activity data, applies documented emission factors,
    computes category and date aggregations, and saves the result.

    Parameters
    ----------
    activity_path : str | Path
        Path to processed cleaned activity CSV.
    factors_path : str | Path
        Path to emission factors CSV registry.
    output_path : str | Path | None
        Optional destination for calculated emissions CSV.
    strict_factors : bool, default False
        If True, raises FileNotFoundError if factors_path does not exist.
        If False, logs warning and falls back to default documented factors.

    Returns
    -------
    tuple[pd.DataFrame, CarbonAccountingSummary]
        Calculated emissions DataFrame and summary analysis object.
    """
    act_path = Path(activity_path)
    if not act_path.exists():
        raise FileNotFoundError(f"Cleaned activity file not found at: {act_path.resolve()}")

    f_path = Path(factors_path)
    if f_path.exists():
        registry = EmissionFactorRegistry.from_csv(f_path)
    else:
        if strict_factors:
            raise FileNotFoundError(f"Emission factors file required but not found: {f_path.resolve()}")
        logger.warning("Factors file not found at %s. Falling back to default documented factors.", f_path)
        registry = EmissionFactorRegistry.default()

    activity_df = pd.read_csv(act_path)
    emissions_df = calculate_campus_emissions(activity_df, registry=registry)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        emissions_df.to_csv(out_p, index=False)
        logger.info("Saved campus emissions dataset to %s", out_p.resolve())

    summary = CarbonAccountingSummary(
        total_emissions_kg=calculate_total_emissions(emissions_df),
        total_emissions_mt=round(calculate_total_emissions(emissions_df) / 1000.0, 2),
        emissions_per_student_kg=calculate_emissions_per_student(emissions_df),
        category_contributions=calculate_category_contributions(emissions_df),
        emissions_by_category=aggregate_emissions_by_category(emissions_df),
        emissions_by_date=aggregate_emissions_by_date(emissions_df),
    )

    return emissions_df, summary


def main() -> None:
    """CLI entrypoint for campus carbon accounting module."""
    parser = argparse.ArgumentParser(
        description="Compute carbon emissions from campus activity data using documented factors."
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="data/processed/cleaned_activity.csv",
        help="Path to cleaned activity CSV (default: data/processed/cleaned_activity.csv)",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default="data/emission_factors.csv",
        help="Path to emission factors registry CSV (default: data/emission_factors.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/campus_emissions.csv",
        help="Path to output emissions CSV (default: data/processed/campus_emissions.csv)",
    )
    parser.add_argument(
        "--strict-factors",
        action="store_true",
        help="Raise error if emission factors file is missing instead of falling back to default.",
    )

    args = parser.parse_args()

    print("=================================================================")
    print("CAMPUS CARBON ACCOUNTING PIPELINE")
    print("=================================================================")
    print(f"Activity Data Source : {args.activity}")
    print(f"Emission Factors Path: {args.factors}")
    print(f"Output Destination   : {args.output}")
    print("-----------------------------------------------------------------")

    emissions_df, summary = run_carbon_accounting(
        activity_path=args.activity,
        factors_path=args.factors,
        output_path=args.output,
        strict_factors=args.strict_factors,
    )

    print("\n" + summary.summary())

    print("\n--- Emissions by Date (First 6 Months) ---")
    print(summary.emissions_by_date.head(6).to_string(index=False))

    print("\n--- Emissions by Category ---")
    print(summary.emissions_by_category.to_string(index=False))

    print(f"\nComprehensive emissions dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
