"""
Monte Carlo Uncertainty Quantification and Sensitivity Analysis
================================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 12: Uncertainty Analysis
Engine for quantifying and decomposing carbon forecast uncertainty into:
1. Forecast Uncertainty (V_forecast): Intrinsic time-series state-space error.
2. Activity-Data Uncertainty (V_act): Historical measurement and survey error.
3. Emission-Factor Uncertainty (V_ef): Grid and emissions conversion factor error.

Decomposition uses a Monte Carlo One-at-a-Time (OAT) variance framework,
enforcing physical non-negativity (X >= 0) via Truncated Normal distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.carbon_accounting import (
    CATEGORY_TO_ACTIVITY_COL,
    EmissionFactor,
    EmissionFactorRegistry,
    aggregate_emissions_by_date,
    calculate_campus_emissions,
)
from src.forecasting.holt_winters import HoltWintersForecaster

logger = logging.getLogger("campus_carbon.uncertainty")

# Default relative standard deviations (simulated prototype assumptions)
DEFAULT_ACTIVITY_RSD: dict[str, float] = {
    "electricity_kwh": 0.02,
    "travel_km": 0.15,
    "waste_kg": 0.10,
    "procurement_inr": 0.10,
}

DEFAULT_EF_RSD: dict[str, float] = {
    "electricity_kwh": 0.05,
    "travel_km": 0.20,
    "waste_kg": 0.15,
    "procurement_inr": 0.15,
}

DEFAULT_RAW_DATA_PATH = Path("data/raw/campus_activity.csv")


@dataclass(frozen=True)
class UncertaintyParameters:
    """
    Configuration parameters for Monte Carlo uncertainty analysis.

    Parameters
    ----------
    activity_rsd : dict[str, float]
        Relative standard deviations for activity data categories.
    ef_rsd : dict[str, float]
        Relative standard deviations for emission factor categories.
    seed : int, default 42
        Master random seed for reproducibility.
    n_simulations : int, default 1000
        Number of Monte Carlo iterations.
    """

    activity_rsd: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTIVITY_RSD))
    ef_rsd: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EF_RSD))
    seed: int = 42
    n_simulations: int = 1000

    def __post_init__(self) -> None:
        """Validate uncertainty parameters and enforce physical non-negativity."""
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {type(self.seed).__name__}")
        if not isinstance(self.n_simulations, (int, np.integer)):
            raise TypeError(f"n_simulations must be an integer, got {type(self.n_simulations).__name__}")
        if self.n_simulations <= 0:
            raise ValueError(f"n_simulations must be a positive integer, got {self.n_simulations}")

        if not isinstance(self.activity_rsd, dict):
            raise TypeError(f"activity_rsd must be a dictionary, got {type(self.activity_rsd).__name__}")
        if not isinstance(self.ef_rsd, dict):
            raise TypeError(f"ef_rsd must be a dictionary, got {type(self.ef_rsd).__name__}")

        for k, v in self.activity_rsd.items():
            try:
                num_v = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Activity RSD for '{k}' is not a valid float: {v!r}") from exc
            if math.isnan(num_v) or math.isinf(num_v):
                raise ValueError(f"Activity RSD for '{k}' must be finite, got {num_v}")
            if num_v < 0.0:
                raise ValueError(f"Relative standard deviation cannot be negative, got {num_v} for '{k}'")

        for k, v in self.ef_rsd.items():
            try:
                num_v = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Emission factor RSD for '{k}' is not a valid float: {v!r}") from exc
            if math.isnan(num_v) or math.isinf(num_v):
                raise ValueError(f"Emission factor RSD for '{k}' must be finite, got {num_v}")
            if num_v < 0.0:
                raise ValueError(f"Relative standard deviation cannot be negative, got {num_v} for '{k}'")


@dataclass(frozen=True)
class VarianceDecomposition:
    """
    Decomposition of total forecast variance into constituent uncertainty sources.

    Parameters
    ----------
    forecast_variance_pct : float
        Percentage contribution of Holt-Winters forecast path uncertainty (0 to 100).
    activity_variance_pct : float
        Percentage contribution of activity data measurement uncertainty (0 to 100).
    ef_variance_pct : float
        Percentage contribution of emission factor uncertainty (0 to 100).
    forecast_variance : float, default 0.0
        Absolute variance from forecast state-space simulation (kgCO2e^2).
    activity_variance : float, default 0.0
        Absolute variance from activity data perturbations (kgCO2e^2).
    ef_variance : float, default 0.0
        Absolute variance from emission factor perturbations (kgCO2e^2).
    total_variance : float, default 0.0
        Sum of isolated component variances (kgCO2e^2).
    target : str, default "total_emissions_kg"
        Forecast target name.
    """

    forecast_variance_pct: float
    activity_variance_pct: float
    ef_variance_pct: float
    forecast_variance: float = 0.0
    activity_variance: float = 0.0
    ef_variance: float = 0.0
    total_variance: float = 0.0
    target: str = "total_emissions_kg"


def _lookup_rsd(rsd_dict: dict[str, float], key: str) -> float:
    """
    Look up relative standard deviation by column name or category name (case-insensitive).
    """
    key_clean = key.strip().lower()
    if key in rsd_dict:
        return float(rsd_dict[key])
    for k, v in rsd_dict.items():
        if k.strip().lower() == key_clean:
            return float(v)
    for cat, col in CATEGORY_TO_ACTIVITY_COL.items():
        if key_clean in (cat.lower(), col.lower()):
            for k, v in rsd_dict.items():
                if k.strip().lower() in (cat.lower(), col.lower()):
                    return float(v)
    return 0.0


class _FastFitContext:
    """
    Context manager configuring statsmodels ExponentialSmoothing.fit
    for accelerated Monte Carlo iterations (bypasses full brute grid search).
    """

    def __enter__(self) -> _FastFitContext:
        self._orig_fit = ExponentialSmoothing.fit

        def _fast_fit(model_self: ExponentialSmoothing, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("use_brute", False)
            kwargs.setdefault("method", "L-BFGS-B")
            return self._orig_fit(model_self, *args, **kwargs)

        ExponentialSmoothing.fit = _fast_fit
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ExponentialSmoothing.fit = self._orig_fit


def generate_perturbed_inputs(
    df: pd.DataFrame,
    registry: EmissionFactorRegistry | None = None,
    params: UncertaintyParameters | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, EmissionFactorRegistry]:
    """
    Generate perturbed activity data and emission factor registry using Truncated Normal distributions.

    Enforces physical non-negativity (X >= 0) by bounding perturbed values strictly at 0.0.

    Parameters
    ----------
    df : pd.DataFrame
        Baseline raw or cleaned campus activity DataFrame.
    registry : EmissionFactorRegistry | None, default None
        Baseline emission factor registry. Defaults to EmissionFactorRegistry.default().
    params : UncertaintyParameters | None, default None
        Uncertainty configuration parameters. Defaults to UncertaintyParameters().
    rng : np.random.Generator | None, default None
        NumPy random number generator. Defaults to np.random.default_rng(params.seed).

    Returns
    -------
    tuple[pd.DataFrame, EmissionFactorRegistry]
        Tuple of (perturbed_activity_df, perturbed_emission_factor_registry).
    """
    if params is None:
        params = UncertaintyParameters()
    if registry is None:
        registry = EmissionFactorRegistry.default()
    if rng is None:
        rng = np.random.default_rng(params.seed)

    # 1. Perturb Activity Data (row-level perturbations bounded at 0.0)
    p_df = df.copy()
    for cat, col in CATEGORY_TO_ACTIVITY_COL.items():
        if col in p_df.columns:
            rsd = _lookup_rsd(params.activity_rsd, col)
            if rsd > 0.0:
                vals = p_df[col].to_numpy(dtype=float)
                noise = rng.normal(0.0, rsd, size=len(vals))
                perturbed = np.maximum(0.0, vals * (1.0 + noise))
                p_df[col] = np.round(perturbed, 4)

    # 2. Perturb Emission Factors (category-level factors bounded at 0.0)
    factors = registry.list_factors()
    new_factors: list[EmissionFactor] = []
    for f in factors:
        rsd = _lookup_rsd(params.ef_rsd, f.activity_type)
        factor_val = float(f.factor)
        if rsd > 0.0:
            noise = float(rng.normal(0.0, rsd))
            factor_val = max(0.0, factor_val * (1.0 + noise))
            factor_val = round(factor_val, 6)

        new_f = EmissionFactor(
            category=f.category,
            activity_type=f.activity_type,
            unit=f.unit,
            factor=factor_val,
            source=f.source,
            version=f.version,
        )
        new_factors.append(new_f)

    p_registry = EmissionFactorRegistry(new_factors)
    return p_df, p_registry


def run_monte_carlo_pipeline(
    df: pd.DataFrame | None = None,
    registry: EmissionFactorRegistry | None = None,
    params: UncertaintyParameters | None = None,
    target: str = "total_emissions_kg",
    train_months: int = 24,
    forecast_horizon: int = 12,
    mode: str = "all",
    include_forecast_noise: bool = False,
) -> pd.DataFrame:
    """
    Run end-to-end Monte Carlo simulation wrapper across carbon accounting and Holt-Winters forecasting.

    Parameters
    ----------
    df : pd.DataFrame | None, default None
        Raw campus activity DataFrame. If None, loaded from data/raw/campus_activity.csv.
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry. If None, EmissionFactorRegistry.default() is used.
    params : UncertaintyParameters | None, default None
        Uncertainty parameters. If None, UncertaintyParameters() is used.
    target : str, default "total_emissions_kg"
        Target emission series to forecast.
    train_months : int, default 24
        Number of historical months used for model fitting.
    forecast_horizon : int, default 12
        Number of future steps to forecast.
    mode : str, default "all"
        Uncertainty simulation mode:
        - "all": Perturb both activity data and emission factors.
        - "activity_only": Perturb activity data only (ef_rsd forced to 0).
        - "ef_only": Perturb emission factors only (activity_rsd forced to 0).
    include_forecast_noise : bool, default False
        Whether to add Holt-Winters state-space simulation error on top of input perturbations.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns:
        ['date', 'point_forecast', 'mean', 'std', 'ci_lower_95', 'ci_upper_95', 'relative_uncertainty_pct'].
    """
    if params is None:
        params = UncertaintyParameters()
    if df is None:
        if not DEFAULT_RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw activity data not found at {DEFAULT_RAW_DATA_PATH.resolve()}")
        df = pd.read_csv(DEFAULT_RAW_DATA_PATH)
    if registry is None:
        registry = EmissionFactorRegistry.default()

    # 1. Unperturbed baseline emissions and deterministic point forecast
    base_emissions = calculate_campus_emissions(df, registry)
    base_monthly = aggregate_emissions_by_date(base_emissions)

    if target not in base_monthly.columns:
        raise ValueError(
            f"Target '{target}' not found in aggregated monthly emissions. "
            f"Available: {[c for c in base_monthly.columns if c != 'date']}"
        )

    base_y = base_monthly[target].iloc[:train_months].to_numpy(dtype=float)
    base_hw = HoltWintersForecaster().fit(base_y)
    det_point_forecast = base_hw.forecast(steps=forecast_horizon)

    # Forecast dates from monthly sequence
    if len(base_monthly) >= train_months + forecast_horizon:
        forecast_dates = base_monthly["date"].iloc[train_months : train_months + forecast_horizon].tolist()
    else:
        last_date = pd.to_datetime(base_monthly["date"].iloc[train_months - 1])
        forecast_dates = [
            (last_date + pd.DateOffset(months=m)).strftime("%Y-%m-%d")
            for m in range(1, forecast_horizon + 1)
        ]

    # 2. Check for zero-variance flag: if all RSDs are 0 and no forecast noise is requested
    is_zero_variance = (
        all(v == 0.0 for v in params.activity_rsd.values())
        and all(v == 0.0 for v in params.ef_rsd.values())
        and not include_forecast_noise
    )

    if is_zero_variance:
        return pd.DataFrame({
            "date": forecast_dates,
            "point_forecast": np.round(det_point_forecast, 2),
            "mean": np.round(det_point_forecast, 2),
            "std": np.zeros(forecast_horizon, dtype=float),
            "ci_lower_95": np.round(det_point_forecast, 2),
            "ci_upper_95": np.round(det_point_forecast, 2),
            "relative_uncertainty_pct": np.zeros(forecast_horizon, dtype=float),
        })

    # 3. Configure effective parameters per mode
    mode_lower = mode.strip().lower()
    if mode_lower == "activity_only":
        eff_params = UncertaintyParameters(
            activity_rsd=params.activity_rsd,
            ef_rsd={k: 0.0 for k in params.ef_rsd},
            seed=params.seed,
            n_simulations=params.n_simulations,
        )
    elif mode_lower == "ef_only":
        eff_params = UncertaintyParameters(
            activity_rsd={k: 0.0 for k in params.activity_rsd},
            ef_rsd=params.ef_rsd,
            seed=params.seed,
            n_simulations=params.n_simulations,
        )
    elif mode_lower == "all":
        eff_params = params
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'all', 'activity_only', or 'ef_only'.")

    # 4. Execute Monte Carlo iterations
    n_sims = eff_params.n_simulations
    sim_paths = np.empty((n_sims, forecast_horizon), dtype=float)
    rng = np.random.default_rng(eff_params.seed)

    with _FastFitContext(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for i in range(n_sims):
            p_df, p_reg = generate_perturbed_inputs(df, registry, eff_params, rng=rng)
            p_emissions = calculate_campus_emissions(p_df, p_reg)
            p_monthly = aggregate_emissions_by_date(p_emissions)
            y_train = p_monthly[target].iloc[:train_months].to_numpy(dtype=float)

            hw_model = HoltWintersForecaster().fit(y_train)

            if include_forecast_noise:
                path = hw_model.fitted_model.simulate(
                    nsimulations=forecast_horizon,
                    repetitions=1,
                    error="add",
                    rng=rng,
                ).ravel()
                sim_paths[i, :] = path
            else:
                sim_paths[i, :] = hw_model.forecast(steps=forecast_horizon)

    # 5. Compute summary statistics
    mean_forecast = np.mean(sim_paths, axis=0)
    std_forecast = np.std(sim_paths, axis=0, ddof=1)
    lower_95 = np.percentile(sim_paths, 2.5, axis=0)
    upper_95 = np.percentile(sim_paths, 97.5, axis=0)

    # Enforce physical non-negativity
    lower_95 = np.maximum(0.0, lower_95)

    # Relative uncertainty width: (CI width / Point Forecast) * 100
    rel_unc = np.where(
        det_point_forecast > 0,
        ((upper_95 - lower_95) / det_point_forecast) * 100.0,
        0.0,
    )

    result_df = pd.DataFrame({
        "date": forecast_dates,
        "point_forecast": np.round(det_point_forecast, 2),
        "mean": np.round(mean_forecast, 2),
        "std": np.round(std_forecast, 2),
        "ci_lower_95": np.round(lower_95, 2),
        "ci_upper_95": np.round(upper_95, 2),
        "relative_uncertainty_pct": np.round(rel_unc, 2),
    })
    result_df.attrs["simulations"] = sim_paths
    return result_df


def decompose_uncertainty(
    df: pd.DataFrame | None = None,
    registry: EmissionFactorRegistry | None = None,
    params: UncertaintyParameters | None = None,
    target: str = "total_emissions_kg",
    train_months: int = 24,
    forecast_horizon: int = 12,
) -> VarianceDecomposition:
    """
    Decompose total forecast variance using One-at-a-Time (OAT) variance sensitivity.

    Steps:
    1. Baseline Forecast Variance (V_forecast): Holt-Winters state-space simulations on unperturbed series.
    2. Activity Variance (V_act): N Monte Carlo simulations perturbing only activity data (point forecasts).
    3. Emission-Factor Variance (V_ef): N Monte Carlo simulations perturbing only emission factors (point forecasts).
    4. Normalization: Sum variances and scale percentages strictly to 100.0%.

    Parameters
    ----------
    df : pd.DataFrame | None, default None
        Raw campus activity DataFrame. If None, loaded from data/raw/campus_activity.csv.
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry. If None, EmissionFactorRegistry.default() is used.
    params : UncertaintyParameters | None, default None
        Uncertainty parameters. If None, UncertaintyParameters() is used.
    target : str, default "total_emissions_kg"
        Target emission series to decompose.
    train_months : int, default 24
        Number of historical training months.
    forecast_horizon : int, default 12
        Forecast horizon length.

    Returns
    -------
    VarianceDecomposition
        Normalized percentage contributions and absolute variances.
    """
    if params is None:
        params = UncertaintyParameters()
    if df is None:
        if not DEFAULT_RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw activity data not found at {DEFAULT_RAW_DATA_PATH.resolve()}")
        df = pd.read_csv(DEFAULT_RAW_DATA_PATH)
    if registry is None:
        registry = EmissionFactorRegistry.default()

    # 1. Unperturbed baseline emissions
    base_emissions = calculate_campus_emissions(df, registry)
    base_monthly = aggregate_emissions_by_date(base_emissions)

    if target not in base_monthly.columns:
        raise ValueError(f"Target '{target}' not found in monthly emissions.")

    train_y = base_monthly[target].iloc[:train_months].to_numpy(dtype=float)
    n_sims = params.n_simulations

    # ---------------------------------------------------------
    # STEP 1: Baseline Forecast Variance (V_forecast)
    # ---------------------------------------------------------
    base_hw = HoltWintersForecaster(random_state=params.seed, simulation_repetitions=n_sims).fit(train_y)
    rng_fc = np.random.default_rng(params.seed)
    fc_sim = base_hw.fitted_model.simulate(
        nsimulations=forecast_horizon,
        repetitions=n_sims,
        error="add",
        rng=rng_fc,
    )
    # Variance across simulated paths per forecast step, then mean across horizon
    v_forecast_steps = np.var(fc_sim, axis=1, ddof=1)
    v_forecast = float(np.mean(v_forecast_steps))

    # ---------------------------------------------------------
    # STEP 2: Activity Variance (V_act) - Freeze forecast paths
    # ---------------------------------------------------------
    act_pts = np.empty((n_sims, forecast_horizon), dtype=float)
    rng_act = np.random.default_rng(params.seed)

    act_params = UncertaintyParameters(
        activity_rsd=params.activity_rsd,
        ef_rsd={k: 0.0 for k in params.ef_rsd},
        seed=params.seed,
        n_simulations=n_sims,
    )

    with _FastFitContext(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for i in range(n_sims):
            p_df, _ = generate_perturbed_inputs(df, registry, act_params, rng=rng_act)
            p_em = calculate_campus_emissions(p_df, registry)
            p_m = aggregate_emissions_by_date(p_em)
            y_i = p_m[target].iloc[:train_months].to_numpy(dtype=float)
            hw_i = HoltWintersForecaster().fit(y_i)
            act_pts[i, :] = hw_i.forecast(steps=forecast_horizon)

    v_act_steps = np.var(act_pts, axis=0, ddof=1)
    v_act = float(np.mean(v_act_steps))

    # ---------------------------------------------------------
    # STEP 3: Emission-Factor Variance (V_ef) - Freeze forecast paths
    # ---------------------------------------------------------
    ef_pts = np.empty((n_sims, forecast_horizon), dtype=float)
    rng_ef = np.random.default_rng(params.seed)

    ef_params = UncertaintyParameters(
        activity_rsd={k: 0.0 for k in params.activity_rsd},
        ef_rsd=params.ef_rsd,
        seed=params.seed,
        n_simulations=n_sims,
    )

    with _FastFitContext(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for i in range(n_sims):
            _, p_reg = generate_perturbed_inputs(df, registry, ef_params, rng=rng_ef)
            p_em = calculate_campus_emissions(df, p_reg)
            p_m = aggregate_emissions_by_date(p_em)
            y_i = p_m[target].iloc[:train_months].to_numpy(dtype=float)
            hw_i = HoltWintersForecaster().fit(y_i)
            ef_pts[i, :] = hw_i.forecast(steps=forecast_horizon)

    v_ef_steps = np.var(ef_pts, axis=0, ddof=1)
    v_ef = float(np.mean(v_ef_steps))

    # ---------------------------------------------------------
    # STEP 4: Decomposition and Normalization
    # ---------------------------------------------------------
    total_var = v_forecast + v_act + v_ef

    if total_var > 0:
        pct_fc = (v_forecast / total_var) * 100.0
        pct_act = (v_act / total_var) * 100.0
        pct_ef = (v_ef / total_var) * 100.0

        # Exact normalization to 100.00%
        r_fc = round(pct_fc, 2)
        r_act = round(pct_act, 2)
        r_ef = round(100.0 - (r_fc + r_act), 2)
    else:
        r_fc, r_act, r_ef = 0.0, 0.0, 0.0

    return VarianceDecomposition(
        forecast_variance_pct=r_fc,
        activity_variance_pct=r_act,
        ef_variance_pct=r_ef,
        forecast_variance=round(v_forecast, 2),
        activity_variance=round(v_act, 2),
        ef_variance=round(v_ef, 2),
        total_variance=round(total_var, 2),
        target=target,
    )


def decompose_all_targets(
    df: pd.DataFrame | None = None,
    registry: EmissionFactorRegistry | None = None,
    params: UncertaintyParameters | None = None,
    targets: Sequence[str] | None = None,
    train_months: int = 24,
    forecast_horizon: int = 12,
) -> tuple[pd.DataFrame, dict[str, VarianceDecomposition]]:
    """
    Execute variance decomposition across all standard forecast targets.

    Parameters
    ----------
    df : pd.DataFrame | None, default None
        Raw campus activity DataFrame.
    registry : EmissionFactorRegistry | None, default None
        Emission factor registry.
    params : UncertaintyParameters | None, default None
        Uncertainty parameters.
    targets : Sequence[str] | None, default None
        Target emission columns. Defaults to Total, Electricity, and Travel.
    train_months : int, default 24
        Training history length.
    forecast_horizon : int, default 12
        Forecast horizon.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, VarianceDecomposition]]
        Summary DataFrame and mapping from target name to VarianceDecomposition.
    """
    if targets is None:
        targets = [
            "total_emissions_kg",
            "electricity_emissions_kg",
            "travel_emissions_kg",
        ]

    results: dict[str, VarianceDecomposition] = {}
    records: list[dict[str, Any]] = []

    for tgt in targets:
        logger.info("Decomposing uncertainty for target: %s", tgt)
        decomp = decompose_uncertainty(
            df=df,
            registry=registry,
            params=params,
            target=tgt,
            train_months=train_months,
            forecast_horizon=forecast_horizon,
        )
        results[tgt] = decomp
        records.append({
            "target": tgt,
            "forecast_variance_kg2": decomp.forecast_variance,
            "forecast_variance_pct": decomp.forecast_variance_pct,
            "activity_variance_kg2": decomp.activity_variance,
            "activity_variance_pct": decomp.activity_variance_pct,
            "ef_variance_kg2": decomp.ef_variance,
            "ef_variance_pct": decomp.ef_variance_pct,
            "total_variance_kg2": decomp.total_variance,
        })

    summary_df = pd.DataFrame(records)
    return summary_df, results


def run_uncertainty_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute complete Phase 12 uncertainty analysis workflow.

    1. Executes OAT variance decomposition for Total, Electricity, and Travel targets.
    2. Runs end-to-end Monte Carlo simulation for 2025 forecast confidence intervals.
    3. Exports summary CSV artifacts to data/processed/.
    """
    print("=" * 75)
    print("PHASE 12: UNCERTAINTY ANALYSIS & VARIANCE DECOMPOSITION")
    print("Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)")
    print("=" * 75)

    params = UncertaintyParameters(seed=42, n_simulations=1000)
    print(f"Monte Carlo Simulations: N = {params.n_simulations}")
    print(f"Random Seed            : {params.seed}")
    print(f"Activity Data RSDs     : {params.activity_rsd}")
    print(f"Emission Factor RSDs   : {params.ef_rsd}")
    print("-" * 75)

    # 1. Variance Decomposition across all targets
    print("\nExecuting One-at-a-Time (OAT) Variance Decomposition across targets...")
    summary_df, decomps = decompose_all_targets(params=params)

    print("\n--- VARIANCE DECOMPOSITION SUMMARY TABLE ---")
    display_cols = [
        "target",
        "forecast_variance_pct",
        "activity_variance_pct",
        "ef_variance_pct",
        "total_variance_kg2",
    ]
    print(summary_df[display_cols].to_string(index=False))

    # Export Decomposition CSV
    out_decomp_csv = Path("data/processed/uncertainty_variance_decomposition.csv")
    out_decomp_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_decomp_csv, index=False)
    print(f"\n[Artifact] Exported decomposition CSV: {out_decomp_csv.resolve()}")

    # 2. Monte Carlo 2025 Forecast Intervals for Total Emissions
    print("\nRunning End-to-End Monte Carlo Forecast Wrapper for Total Emissions...")
    mc_intervals_df = run_monte_carlo_pipeline(params=params, target="total_emissions_kg")

    print("\n--- 2025 MONTE CARLO FORECAST INTERVALS (TOTAL EMISSIONS) ---")
    print(mc_intervals_df.to_string(index=False))

    # Export Intervals CSV
    out_mc_csv = Path("data/processed/monte_carlo_forecast_intervals_2025.csv")
    mc_intervals_df.to_csv(out_mc_csv, index=False)
    print(f"\n[Artifact] Exported forecast intervals CSV: {out_mc_csv.resolve()}")

    print("=" * 75)
    return summary_df, mc_intervals_df


if __name__ == "__main__":
    run_uncertainty_analysis()
