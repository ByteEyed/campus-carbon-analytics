"""
Campus Decarbonization Intervention Scenario Library
====================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 13: Intervention Scenario Library
Defines decarbonization interventions and their functional activity calculation contracts:
1. INT-001: LED Lighting Retrofit (Multiplicative Electricity Reduction)
2. INT-002: Rooftop Solar Installation (Additive Behind-the-Meter Generation)
3. INT-003: AC / HVAC Optimization (Multiplicative Seasonal Electricity Reduction)
4. INT-004: Waste Segregation & Composting (Multiplicative Waste Diversion)
5. INT-005: Low-Carbon / Sustainable Transport (Multiplicative Travel Substitution)

Paradigm:
Interventions act as pure functional transformations on the Activity DataFrame,
reducing physical consumption before carbon accounting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.carbon_accounting import (
    CATEGORY_ELECTRICITY,
    CATEGORY_PROCUREMENT,
    CATEGORY_TO_ACTIVITY_COL,
    CATEGORY_TRAVEL,
    CATEGORY_WASTE,
    PRIMARY_CATEGORIES,
)


class InterventionMechanism(str, Enum):
    """Classification of intervention transformation mechanism."""

    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"


@dataclass
class BaseIntervention(ABC):
    """
    Abstract Base Class defining the contract for all decarbonization interventions.

    Interventions transform raw/cleaned activity DataFrames into post-intervention
    activity DataFrames without modifying carbon accounting logic.

    Attributes
    ----------
    id : str
        Unique intervention identifier (e.g., 'INT-001').
    name : str
        Human-readable name of the intervention.
    category : str
        Target emission category ('Electricity', 'Travel', 'Waste', 'Procurement').
    target_column : str
        Target activity column in the DataFrame ('electricity_kwh', etc.).
    capital_cost : float
        Initial implementation cost in INR (₹).
    lifespan_years : int
        Expected operational lifespan of the project in years.
    is_simulated_assumption : bool, default True
        Explicit flag indicating simulated/assumed parameter provenance.
    mechanism_type : str, default 'multiplicative'
        Transformation mechanism ('multiplicative' or 'additive').
    priority : int, default 1
        Execution priority order (1 for multiplicative, 2 for additive).
    description : str, default ""
        Detailed description of the intervention project.
    """

    id: str
    name: str
    category: str
    target_column: str
    capital_cost: float
    lifespan_years: int
    is_simulated_assumption: bool = True
    mechanism_type: str = InterventionMechanism.MULTIPLICATIVE.value
    priority: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        """Validate core schema and physical constraints."""
        self.validate()

    def validate(self) -> None:
        """Validate schema, parameter bounds, and physical non-negativity."""
        # 1. Identifier and naming
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"Intervention id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Intervention name must be a non-empty string, got {self.name!r}")

        # 2. Category and target column compatibility
        norm_cat = self.category.strip().title()
        if norm_cat not in PRIMARY_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. Must be one of {PRIMARY_CATEGORIES}."
            )

        expected_col = CATEGORY_TO_ACTIVITY_COL[norm_cat]
        if self.target_column != expected_col:
            raise ValueError(
                f"Target column mismatch for category '{self.category}': "
                f"expected '{expected_col}', got '{self.target_column}'."
            )

        # 3. Capital cost validation
        try:
            cost = float(self.capital_cost)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"capital_cost {self.capital_cost!r} is not a valid float.") from exc
        if math.isnan(cost) or math.isinf(cost):
            raise ValueError(f"capital_cost must be finite, got {self.capital_cost}.")
        if cost < 0.0:
            raise ValueError(f"capital_cost cannot be negative, got {self.capital_cost}.")

        # 4. Lifespan validation
        if not isinstance(self.lifespan_years, (int, np.integer)):
            raise TypeError(f"lifespan_years must be an integer, got {type(self.lifespan_years).__name__}")
        if self.lifespan_years < 1:
            raise ValueError(f"lifespan_years must be at least 1, got {self.lifespan_years}.")

        # 5. Mechanism and priority validation
        valid_mechanisms = {
            InterventionMechanism.MULTIPLICATIVE.value,
            InterventionMechanism.ADDITIVE.value,
        }
        if self.mechanism_type not in valid_mechanisms:
            raise ValueError(
                f"Invalid mechanism_type '{self.mechanism_type}'. Expected one of {valid_mechanisms}."
            )

        if not isinstance(self.priority, (int, np.integer)) or self.priority < 1:
            raise ValueError(f"priority must be an integer >= 1, got {self.priority}.")

    @abstractmethod
    def _transform_activity(self, df: pd.DataFrame) -> pd.Series | np.ndarray:
        """Internal computation transforming the target activity column."""
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the intervention transformation to an activity DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Activity DataFrame containing self.target_column.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with updated target_column and untouched collateral columns.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        ValueError
            If df is empty, missing target_column, or contains invalid data.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("Activity DataFrame cannot be empty.")
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame columns: {list(df.columns)}"
            )

        # Validate input column data
        in_col = df[self.target_column]
        if in_col.isna().any():
            raise ValueError(f"Activity column '{self.target_column}' contains NaN values.")
        try:
            num_in = pd.to_numeric(in_col, errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Activity column '{self.target_column}' contains non-numeric data: {exc}") from exc
        if np.isinf(num_in).any():
            raise ValueError(f"Activity column '{self.target_column}' contains infinite values.")
        if (num_in < 0.0).any():
            raise ValueError(f"Activity column '{self.target_column}' contains negative values.")

        out_df = df.copy()
        transformed = self._transform_activity(out_df)

        # Enforce physical non-negativity and boundary constraints
        transformed = np.maximum(0.0, np.asarray(transformed, dtype=float))

        if np.isnan(transformed).any():
            raise ValueError(f"Transformation produced NaN values in '{self.target_column}'.")
        if np.isinf(transformed).any():
            raise ValueError(f"Transformation produced infinite values in '{self.target_column}'.")

        out_df[self.target_column] = np.round(transformed, 2)
        return out_df

    def to_dict(self) -> dict[str, Any]:
        """Serialize intervention to dictionary."""
        return asdict(self)


# =====================================================================
# 1. INT-001: LED Lighting Retrofit
# =====================================================================

@dataclass
class LEDLightingRetrofit(BaseIntervention):
    """
    INT-001: LED Lighting Retrofit.

    Mechanism: Multiplicative activity reduction.
    Formula: X_new = X_old * (1 - r_led)

    Parameters
    ----------
    reduction_fraction : float, default 0.12
        Fractional reduction in electrical load (e.g. 0.12 for 12% reduction).
    capital_cost : float, default 500000.0
        Simulated capital expenditure (₹5,00,000 INR).
    lifespan_years : int, default 10
        Expected operating lifespan in years.
    """

    id: str = "INT-001"
    name: str = "LED Lighting Retrofit"
    category: str = CATEGORY_ELECTRICITY
    target_column: str = "electricity_kwh"
    capital_cost: float = 500_000.0
    lifespan_years: int = 10
    reduction_fraction: float = 0.12
    is_simulated_assumption: bool = True
    mechanism_type: str = InterventionMechanism.MULTIPLICATIVE.value
    priority: int = 1
    description: str = (
        "Campus-wide retrofit of conventional fluorescent lamps to high-efficiency LEDs, "
        "reducing base-load lighting electricity consumption."
    )

    def validate(self) -> None:
        super().validate()
        try:
            r = float(self.reduction_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reduction_fraction {self.reduction_fraction!r} is not a valid float.") from exc
        if math.isnan(r) or math.isinf(r):
            raise ValueError(f"reduction_fraction must be finite, got {self.reduction_fraction}.")
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"reduction_fraction must be between 0.0 and 1.0, got {self.reduction_fraction}.")

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        vals = df[self.target_column].to_numpy(dtype=float)
        return vals * (1.0 - float(self.reduction_fraction))


# =====================================================================
# 2. INT-002: Rooftop Solar Installation
# =====================================================================

@dataclass
class RooftopSolarInstallation(BaseIntervention):
    """
    INT-002: Rooftop Solar Installation.

    Mechanism: Additive behind-the-meter generation subtraction.
    Formula: X_new = max(0, X_old - G_solar)

    Parameters
    ----------
    monthly_generation_kwh : float, default 5000.0
        Monthly solar photovoltaic energy generation in kWh.
    target_building : str | None, default None
        Specific facility to allocate solar generation to.
        If None and multiple buildings exist per date, generation is allocated
        proportionally across building electricity demands.
    capital_cost : float, default 2500000.0
        Simulated capital expenditure (₹25,00,000 INR).
    lifespan_years : int, default 25
        Expected operating lifespan in years.
    """

    id: str = "INT-002"
    name: str = "Rooftop Solar Installation"
    category: str = CATEGORY_ELECTRICITY
    target_column: str = "electricity_kwh"
    capital_cost: float = 2_500_000.0
    lifespan_years: int = 25
    monthly_generation_kwh: float = 5000.0
    target_building: str | None = None
    is_simulated_assumption: bool = True
    mechanism_type: str = InterventionMechanism.ADDITIVE.value
    priority: int = 2
    description: str = (
        "Installation of rooftop solar photovoltaic system generating clean electricity, "
        "subtracting directly from campus grid consumption."
    )

    def validate(self) -> None:
        super().validate()
        try:
            g = float(self.monthly_generation_kwh)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"monthly_generation_kwh {self.monthly_generation_kwh!r} is not a valid float.") from exc
        if math.isnan(g) or math.isinf(g):
            raise ValueError(f"monthly_generation_kwh must be finite, got {self.monthly_generation_kwh}.")
        if g < 0.0:
            raise ValueError(f"monthly_generation_kwh cannot be negative, got {self.monthly_generation_kwh}.")

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        vals = df[self.target_column].to_numpy(dtype=float).copy()
        g_solar = float(self.monthly_generation_kwh)

        # Case 1: Specific target building requested
        if self.target_building is not None and "building" in df.columns:
            mask = df["building"].astype(str).str.strip().str.lower() == self.target_building.strip().lower()
            vals[mask] = np.maximum(0.0, vals[mask] - g_solar)
            return vals

        # Case 2: Multi-building monthly data (allocate generation across facilities per date)
        if "building" in df.columns and "date" in df.columns and df["building"].nunique() > 1:
            series = df[self.target_column].copy().astype(float)
            for _, group_indices in df.groupby("date").groups.items():
                date_vals = series.loc[group_indices]
                tot_load = float(date_vals.sum())
                if tot_load > 0.0:
                    allocations = g_solar * (date_vals / tot_load)
                    series.loc[group_indices] = np.maximum(0.0, date_vals - allocations)
                else:
                    series.loc[group_indices] = 0.0
            return series.to_numpy()

        # Case 3: Campus-aggregate series or single-facility series
        return np.maximum(0.0, vals - g_solar)


# =====================================================================
# 3. INT-003: AC / HVAC Optimization
# =====================================================================

@dataclass
class ACOptimization(BaseIntervention):
    """
    INT-003: AC / HVAC Optimization.

    Mechanism: Multiplicative activity reduction with seasonal summer masking.
    Formula: X_new = X_old * (1 - r_hvac * I_summer)

    Parameters
    ----------
    reduction_fraction : float, default 0.15
        Fractional reduction during active cooling months (e.g. 0.15 for 15% reduction).
    active_months : tuple[int, ...], default (4, 5, 6, 7, 8)
        Calendar months when cooling curtailment is active (April through August).
    capital_cost : float, default 300000.0
        Simulated capital expenditure (₹3,00,000 INR).
    lifespan_years : int, default 7
        Expected operating lifespan in years.
    """

    id: str = "INT-003"
    name: str = "AC / HVAC Optimization"
    category: str = CATEGORY_ELECTRICITY
    target_column: str = "electricity_kwh"
    capital_cost: float = 300_000.0
    lifespan_years: int = 7
    reduction_fraction: float = 0.15
    active_months: tuple[int, ...] = (4, 5, 6, 7, 8)
    is_simulated_assumption: bool = True
    mechanism_type: str = InterventionMechanism.MULTIPLICATIVE.value
    priority: int = 1
    description: str = (
        "Setpoint thermostat optimization, duct sealing, and scheduled compressor cycling "
        "curtailing cooling loads during peak summer months."
    )

    def validate(self) -> None:
        super().validate()
        try:
            r = float(self.reduction_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reduction_fraction {self.reduction_fraction!r} is not a valid float.") from exc
        if math.isnan(r) or math.isinf(r):
            raise ValueError(f"reduction_fraction must be finite, got {self.reduction_fraction}.")
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"reduction_fraction must be between 0.0 and 1.0, got {self.reduction_fraction}.")

        if not hasattr(self.active_months, "__iter__"):
            raise TypeError("active_months must be an iterable of integer months.")
        if len(self.active_months) == 0:
            raise ValueError("active_months cannot be empty.")
        for m in self.active_months:
            if not isinstance(m, (int, np.integer)) or not (1 <= m <= 12):
                raise ValueError(f"active_months must contain integers from 1 to 12, got {m}.")

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        vals = df[self.target_column].to_numpy(dtype=float).copy()
        r = float(self.reduction_fraction)

        # 1. Determine months from 'date' column
        if "date" in df.columns:
            try:
                months = pd.to_datetime(df["date"]).dt.month.to_numpy()
            except Exception as exc:
                raise ValueError(f"Failed to parse dates in 'date' column: {exc}") from exc
        # 2. Alternatively check for an explicit 'month' column
        elif "month" in df.columns:
            months = pd.to_numeric(df["month"], errors="raise").to_numpy(dtype=int)
        else:
            raise ValueError(
                "ACOptimization requires a 'date' or 'month' column in the DataFrame "
                "to evaluate seasonal active months."
            )

        is_active = np.isin(months, self.active_months)
        vals[is_active] = vals[is_active] * (1.0 - r)
        return vals


# =====================================================================
# 4. INT-004: Waste Segregation & Composting
# =====================================================================

@dataclass
class WasteSegregation(BaseIntervention):
    """
    INT-004: Waste Segregation & Composting.

    Mechanism: Multiplicative activity diversion from municipal landfill.
    Formula: X_new = X_old * (1 - r_diverted)

    Parameters
    ----------
    diversion_fraction : float, default 0.35
        Fraction of solid waste diverted to organic composting (e.g. 0.35 for 35% diversion).
    capital_cost : float, default 150000.0
        Simulated capital expenditure (₹1,50,000 INR).
    lifespan_years : int, default 5
        Expected operating lifespan in years.
    """

    id: str = "INT-004"
    name: str = "Waste Segregation & Composting"
    category: str = CATEGORY_WASTE
    target_column: str = "waste_kg"
    capital_cost: float = 150_000.0
    lifespan_years: int = 5
    diversion_fraction: float = 0.35
    is_simulated_assumption: bool = True
    mechanism_type: str = InterventionMechanism.MULTIPLICATIVE.value
    priority: int = 1
    description: str = (
        "Decentralized organic waste segregation and on-campus aerobic composting, "
        "diverting solid waste away from municipal landfills."
    )

    def validate(self) -> None:
        super().validate()
        try:
            d = float(self.diversion_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"diversion_fraction {self.diversion_fraction!r} is not a valid float.") from exc
        if math.isnan(d) or math.isinf(d):
            raise ValueError(f"diversion_fraction must be finite, got {self.diversion_fraction}.")
        if not (0.0 <= d <= 1.0):
            raise ValueError(f"diversion_fraction must be between 0.0 and 1.0, got {self.diversion_fraction}.")

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        vals = df[self.target_column].to_numpy(dtype=float)
        return vals * (1.0 - float(self.diversion_fraction))


# =====================================================================
# 5. INT-005: Low-Carbon / Sustainable Transport
# =====================================================================

@dataclass
class LowCarbonTransport(BaseIntervention):
    """
    INT-005: Low-Carbon / Sustainable Transport.

    Mechanism: Multiplicative substitution of combustion-vehicle travel distance.
    Formula: X_new = X_old * (1 - r_ev)

    Parameters
    ----------
    ev_adoption_fraction : float, default 0.20
        Fraction of commuter distance transitioned to electric shuttles / transit (e.g. 0.20).
    capital_cost : float, default 1800000.0
        Simulated capital expenditure (₹18,00,000 INR).
    lifespan_years : int, default 8
        Expected operating lifespan in years.
    """

    id: str = "INT-005"
    name: str = "Low-Carbon / Sustainable Transport"
    category: str = CATEGORY_TRAVEL
    target_column: str = "travel_km"
    capital_cost: float = 1_800_000.0
    lifespan_years: int = 8
    ev_adoption_fraction: float = 0.20
    is_simulated_assumption: bool = True
    mechanism_type: str = InterventionMechanism.MULTIPLICATIVE.value
    priority: int = 1
    description: str = (
        "Deployment of campus electric vehicle (EV) shuttles and commuter EV charging incentives, "
        "substituting combustion-engine passenger travel distance."
    )

    def validate(self) -> None:
        super().validate()
        try:
            ev = float(self.ev_adoption_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ev_adoption_fraction {self.ev_adoption_fraction!r} is not a valid float.") from exc
        if math.isnan(ev) or math.isinf(ev):
            raise ValueError(f"ev_adoption_fraction must be finite, got {self.ev_adoption_fraction}.")
        if not (0.0 <= ev <= 1.0):
            raise ValueError(f"ev_adoption_fraction must be between 0.0 and 1.0, got {self.ev_adoption_fraction}.")

    def _transform_activity(self, df: pd.DataFrame) -> np.ndarray:
        vals = df[self.target_column].to_numpy(dtype=float)
        return vals * (1.0 - float(self.ev_adoption_fraction))


# =====================================================================
# Canonical Aliases
# =====================================================================
LEDLighting = LEDLightingRetrofit
SolarInstallation = RooftopSolarInstallation
HVACOptimization = ACOptimization
WasteSegregationComposting = WasteSegregation
SustainableEVTransport = LowCarbonTransport

# Registry mapping intervention IDs to canonical classes
INTERVENTION_CATALOG: dict[str, type[BaseIntervention]] = {
    "INT-001": LEDLightingRetrofit,
    "INT-002": RooftopSolarInstallation,
    "INT-003": ACOptimization,
    "INT-004": WasteSegregation,
    "INT-005": LowCarbonTransport,
}


def get_default_interventions() -> list[BaseIntervention]:
    """Return new instances of all five canonical interventions with default simulated parameters."""
    return [cls() for cls in INTERVENTION_CATALOG.values()]


def sort_interventions(interventions: Sequence[BaseIntervention]) -> list[BaseIntervention]:
    """
    Sort interventions strictly by physical interaction priority order.

    Rule: Multiplicative interventions (priority=1, e.g. LED, HVAC) MUST be applied
    before Additive interventions (priority=2, e.g. Rooftop Solar).
    Preserves original insertion order among interventions with equal priority.

    Parameters
    ----------
    interventions : Sequence[BaseIntervention]
        Unordered list of interventions.

    Returns
    -------
    list[BaseIntervention]
        Sorted list conforming to physical interaction rules.
    """
    return sorted(interventions, key=lambda i: i.priority)


def intervention_from_dict(data: dict[str, Any]) -> BaseIntervention:
    """
    Instantiate appropriate intervention subclass from serialized dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary representation containing 'id'.

    Returns
    -------
    BaseIntervention
        Instantiated concrete intervention subclass.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")
    int_id = str(data.get("id", "")).strip().upper()
    if int_id not in INTERVENTION_CATALOG:
        raise ValueError(
            f"Unknown intervention id '{int_id}'. Available: {list(INTERVENTION_CATALOG.keys())}"
        )
    cls = INTERVENTION_CATALOG[int_id]

    # Convert active_months back to tuple if it was serialized as a list
    clean_data = dict(data)
    if "active_months" in clean_data and isinstance(clean_data["active_months"], list):
        clean_data["active_months"] = tuple(clean_data["active_months"])

    return cls(**clean_data)
