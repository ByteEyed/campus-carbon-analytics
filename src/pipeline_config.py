"""
Campus Carbon Analytics - End-to-End Pipeline Configuration
===========================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 16: Pipeline Configuration Schema
Defines strictly-typed configuration parameters for end-to-end execution
from raw activity data validation through constrained budget optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration specification for the integrated end-to-end pipeline.

    Attributes
    ----------
    activity_data_path : Path, default Path("data/raw/campus_activity.csv")
        Path to raw campus activity CSV.
    emission_factors_path : Path, default Path("data/emission_factors.csv")
        Path to documented emission factors CSV.
    output_dir : Path, default Path("data/processed/pipeline_run")
        Destination directory for exported execution artifacts.
    optimization_budget_inr : float, default 2_500_000.0
        Capital expenditure budget for Phase 15 optimization in INR (₹).
    random_seed : int, default 42
        Master random seed controlling all stochastic components.
    train_months : int, default 24
        Number of historical months allocated for training.
    forecast_horizon_months : int, default 12
        Number of future months to project (Phase 11 & Phase 12).
    uncertainty_simulations : int, default 1000
        Number of Monte Carlo iterations for Phase 12 uncertainty quantification.
    baseline_year : int, default 2025
        Target calendar year for forward-looking scenario baseline evaluation.
    fail_fast : bool, default True
        If True, rejects corrupted data immediately rather than attempting recovery.
    export_artifacts_on_finish : bool, default True
        If True, serializes all pipeline dataframes and summary to output_dir.
    """

    activity_data_path: Path = Path("data/raw/campus_activity.csv")
    emission_factors_path: Path = Path("data/emission_factors.csv")
    output_dir: Path = Path("data/processed/pipeline_run")
    optimization_budget_inr: float = 2_500_000.0
    random_seed: int = 42
    train_months: int = 24
    forecast_horizon_months: int = 12
    uncertainty_simulations: int = 1000
    baseline_year: int = 2025
    fail_fast: bool = True
    export_artifacts_on_finish: bool = True

    def __post_init__(self) -> None:
        """Validate parameter boundaries and ensure path typing."""
        # Normalize paths
        act_p = Path(self.activity_data_path)
        ef_p = Path(self.emission_factors_path)
        out_p = Path(self.output_dir)

        object.__setattr__(self, "activity_data_path", act_p)
        object.__setattr__(self, "emission_factors_path", ef_p)
        object.__setattr__(self, "output_dir", out_p)

        # Path traversal and boundary security validation
        for field_name, p in [
            ("activity_data_path", act_p),
            ("emission_factors_path", ef_p),
            ("output_dir", out_p),
        ]:
            path_str = str(p).replace("\\", "/")
            if ".." in path_str.split("/"):
                raise ValueError(
                    f"Security violation: path traversal sequence ('..') detected in {field_name}: {p}"
                )

        resolved_out = out_p.resolve()
        sensitive_roots = [
            Path("C:/Windows"),
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path("/etc"),
            Path("/bin"),
            Path("/sbin"),
            Path("/root"),
            Path("/sys"),
            Path("/proc"),
        ]
        for sroot in sensitive_roots:
            is_sub = False
            try:
                sroot_res = sroot.resolve()
                if resolved_out == sroot_res or resolved_out.is_relative_to(sroot_res):
                    is_sub = True
            except ValueError:
                # is_relative_to raises ValueError if on different drives on Windows
                is_sub = False
            if is_sub:
                raise ValueError(
                    f"Security violation: output_dir points to sensitive system directory: {self.output_dir}"
                )

        # Budget constraint validation
        if self.optimization_budget_inr < 0.0:
            raise ValueError(
                f"optimization_budget_inr cannot be negative, got {self.optimization_budget_inr}"
            )

        # Time series constraints
        if self.train_months <= 0:
            raise ValueError(f"train_months must be positive, got {self.train_months}")
        if self.forecast_horizon_months <= 0:
            raise ValueError(
                f"forecast_horizon_months must be positive, got {self.forecast_horizon_months}"
            )

        # Simulation constraints
        if self.uncertainty_simulations <= 0:
            raise ValueError(
                f"uncertainty_simulations must be positive, got {self.uncertainty_simulations}"
            )

        # Seed typing
        if not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be an integer, got {type(self.random_seed).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        d = asdict(self)
        d["activity_data_path"] = str(self.activity_data_path)
        d["emission_factors_path"] = str(self.emission_factors_path)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create PipelineConfig instance from dictionary."""
        data_copy = dict(data)
        if "activity_data_path" in data_copy:
            data_copy["activity_data_path"] = Path(data_copy["activity_data_path"])
        if "emission_factors_path" in data_copy:
            data_copy["emission_factors_path"] = Path(data_copy["emission_factors_path"])
        if "output_dir" in data_copy:
            data_copy["output_dir"] = Path(data_copy["output_dir"])
        return cls(**data_copy)

    @classmethod
    def from_json(cls, json_path: str | Path) -> PipelineConfig:
        """Load PipelineConfig instance from JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
