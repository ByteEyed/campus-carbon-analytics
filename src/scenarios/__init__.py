"""
Campus Carbon Analytics - Scenario & Intervention Library Module
================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 13: Intervention Scenario Library
Defines decarbonization interventions and their functional activity calculation contracts:
1. INT-001: LED Lighting Retrofit
2. INT-002: Rooftop Solar Installation
3. INT-003: AC / HVAC Optimization
4. INT-004: Waste Segregation & Composting
5. INT-005: Low-Carbon / Sustainable Transport
"""

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
    sort_interventions,
)

__all__ = [
    "ACOptimization",
    "BaseIntervention",
    "HVACOptimization",
    "INTERVENTION_CATALOG",
    "InterventionMechanism",
    "LEDLighting",
    "LEDLightingRetrofit",
    "LowCarbonTransport",
    "RooftopSolarInstallation",
    "SolarInstallation",
    "SustainableEVTransport",
    "WasteSegregation",
    "WasteSegregationComposting",
    "get_default_interventions",
    "sort_interventions",
]
