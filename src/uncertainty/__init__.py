"""
Campus Carbon Analytics - Uncertainty Analysis Module
=====================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 12: Uncertainty Analysis
Implements Monte Carlo pipeline wrapper and One-at-a-Time (OAT) variance decomposition
for forecast, activity-data, and emission-factor uncertainties.
"""

from src.uncertainty.monte_carlo import (
    DEFAULT_ACTIVITY_RSD,
    DEFAULT_EF_RSD,
    UncertaintyParameters,
    VarianceDecomposition,
    decompose_all_targets,
    decompose_uncertainty,
    generate_perturbed_inputs,
    run_monte_carlo_pipeline,
)

__all__ = [
    "DEFAULT_ACTIVITY_RSD",
    "DEFAULT_EF_RSD",
    "UncertaintyParameters",
    "VarianceDecomposition",
    "decompose_all_targets",
    "decompose_uncertainty",
    "generate_perturbed_inputs",
    "run_monte_carlo_pipeline",
]
