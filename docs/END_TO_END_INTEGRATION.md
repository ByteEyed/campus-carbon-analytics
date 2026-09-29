# Phase 16: End-to-End Pipeline Integration

**Academic Project:** BDS-36 (T.Y. B.Sc. Data Science, Semester V)  
**Academic Year:** 2026-27  
**Module:** [`src.pipeline`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/pipeline.py)  
**Configuration:** [`src.pipeline_config`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/pipeline_config.py)  
**Test Suite:** [`tests/test_pipeline_integration.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/tests/test_pipeline_integration.py)  
**Output Artifacts Directory:** [`data/processed/pipeline_run/`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/pipeline_run)

---

## 1. Overview & Architectural Role

Phase 16 implements the **Pipeline Orchestrator** for the Campus Carbon Analytics platform. It functions as a conductor, binding together the independent modules built across Phases 7 through 15 into a single, automated, reproducible pipeline.

The orchestrator adheres strictly to the **zero duplication rule**: it does not duplicate any domain formulas, accounting logic, forecasting parameters, or optimization routines. Instead, it coordinates data flow via strictly-typed in-memory Python objects (`pd.DataFrame`, `EmissionFactorRegistry`, `PortfolioResult`), validating data contracts at each boundary.

```
+-------------------------------------------------------------------------------+
|                       PHASE 16 PIPELINE ORCHESTRATION                         |
+-------------------------------------------------------------------------------+
                                       │
                         [Stage 1: Load Configuration]
                         src.pipeline_config.PipelineConfig
                                       │
                                       ▼
                       [Stage 2: Activity Data Validation]
                     src.data_validation.validate_activity_dataframe
                                       │
                                       ▼
                         [Stage 3: Carbon Accounting]
                   src.carbon_accounting.calculate_campus_emissions
                   src.carbon_accounting.aggregate_emissions_by_date
                                       │
                                       ▼
                         [Stage 4: Primary Forecasting]
                 src.forecasting.holt_winters.HoltWintersForecaster
                                       │
                                       ▼
                    [Stage 5: Uncertainty Quantification]
                 src.uncertainty.monte_carlo.decompose_all_targets
                 src.uncertainty.monte_carlo.run_monte_carlo_pipeline
                                       │
                                       ▼
                    [Stage 6: Baseline Horizon Extraction]
                     src.scenarios.evaluation.get_12m_baseline
               src.scenarios.evaluation.generate_baseline_activity_bounds
                                       │
                                       ▼
                     [Stage 7: Scenario Impact Evaluation]
                src.scenarios.evaluation.evaluate_all_interventions
                                       │
                                       ▼
                    [Stage 8: Constrained Budget Optimization]
                   src.scenarios.optimization.optimize_portfolio
                                       │
                                       ▼
                  [Stage 9: Integrated Assembly & Serialization]
                      src.pipeline.PipelineResult.export_artifacts
```

---

## 2. Pipeline Stages & Execution Flow

### Stage 1: Load Configuration & Inputs
- Resolves filesystem paths (`activity_data_path`, `emission_factors_path`, `output_dir`).
- Enforces boundary constraints ($B \ge 0$, positive horizon, positive training history).
- Halts immediately with `FileNotFoundError` if input datasets are absent.

### Stage 2: Activity Data Validation (Phase 7)
- Evaluates raw campus activity against [`DATA_DICTIONARY.md`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/DATA_DICTIONARY.md).
- Enforces required columns, building naming consistency, positive numbers, and date formats.
- Generates [`ValidationReport`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/data_validation.py#L110-L150). If `fail_fast=True` and invalid records are detected, execution halts with `PipelineContractError`.

### Stage 3: Carbon Accounting Engine (Phase 8)
- Calculates category emissions (`electricity_emissions_kg`, `travel_emissions_kg`, `waste_emissions_kg`, `procurement_emissions_kg`, `total_emissions_kg`).
- Aggregates monthly campus emissions time series via `aggregate_emissions_by_date()`.
- Verifies time-series continuity: enforces minimum historical length ($\ge 36\text{ months}$) and strictly zero calendar gaps.

### Stage 4: Primary Holt-Winters Forecasting (Phase 11)
- Fits the primary Holt-Winters model (additive trend, additive seasonality $m = 12$) on all 36 available historical months to maximize statistical signal and produce the authoritative forward projections for 2026.
- Projects the 12-month forward horizon (2026) for Total Emissions, Electricity Emissions, and Travel Emissions.
- Evaluates 95% state-space simulation prediction intervals using a standard 2,000 Monte Carlo repetitions, enforcing interval sandwiching: $\text{lower\_95} \le \text{point\_forecast} \le \text{upper\_95}$.

### Stage 5: Uncertainty Quantification & Variance Decomposition (Phase 12)
> [!NOTE] Methodological Training Window Distinction (Stage 4 vs. Stage 5)
> While Stage 4 fits on the complete 36-month timeline to generate forward projections, Stage 5 trains on a 24-month subset (2023–2024) to forecast the 12-month 2025 period. This allows One-at-a-Time (OAT) Monte Carlo variance decomposition against an empirical, observed evaluation window rather than an unobserved future, enabling quantitative assessment of how activity data and emission factor uncertainties contribute to forecast variance.

- Decomposes forecast uncertainty into Forecast error ($V_{\text{fc}}$), Activity measurement error ($V_{\text{act}}$), and Emission-Factor uncertainty ($V_{\text{ef}}$).
- Runs Monte Carlo simulations ($N = 1000$ default) to derive the empirical 95% confidence interval trajectory.

### Stage 6: Baseline Horizon Extraction (Phase 12 & 14)
- Extracts the forward 12-month baseline Activity DataFrame (2025: 60 records across 5 campus buildings).
- Derives activity sensitivity bounds ($\pm 1.96 \times \text{RSD}$) for downstream sensitivity propagation.

### Stage 7: Isolated Scenario Evaluation (Phase 14)
- Evaluates all five Phase 13 interventions in strict isolation (One-At-a-Time).
- Computes avoided carbon ($\text{kgCO}_2\text{e}$), relative percentage reduction (%), and Marginal Abatement Cost ($\text{INR/kgCO}_2\text{e}$).

### Stage 8: Constrained Portfolio Budget Optimization (Phase 15)
- Evaluates the complete discrete powerset ($2^5 = 32$ combinations).
- Sequentially applies multiplicative and additive transformations to eliminate double-counting.
- Enforces budget constraint ($\sum C_i \le B$) and identifies the optimal portfolio via the 4-level deterministic tie-breaker.

### Stage 9: Integrated Assembly & Serialization
- Packages all data structures into [`PipelineResult`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/pipeline.py#L180-L280).
- Flushes 9 processed CSV, JSON, and report artifacts to `config.output_dir`.

---

## 3. Data Contracts & Invariants

| Contract Boundary | Input Type | Invariants Enforced | Failure Mode |
| :--- | :--- | :--- | :--- |
| **Activity Input** | `pd.DataFrame` | Contains 8 required columns; non-empty; non-negative numbers | `PipelineContractError` |
| **Emissions Output** | `pd.DataFrame` | All 4 category emissions present; `total_emissions_kg >= 0`; zero NaNs | `PipelineContractError` |
| **Monthly Timeline** | `pd.DataFrame` | Contiguous monthly date range; length $\ge \text{train} + \text{test}$ (36m) | `PipelineContractError` |
| **Forecast Table** | `pd.DataFrame` | $\text{lower} \le \text{point} \le \text{upper}$; $\text{point} \ge 0$; dates monotonically increasing | `PipelineContractError` |
| **Uncertainty Table** | `pd.DataFrame` | Variance percentages sum to $100.0\% \pm 0.1\%$; $\text{ci\_lower\_95} \ge 0$ | `PipelineContractError` |
| **Scenario Table** | `list[ScenarioResult]` | Exactly 5 canonical initiatives; MAC finite or $+\infty$ | `PipelineContractError` |
| **Budget Optimization** | `PortfolioResult` | Powerset count $= 32$; cost $\le$ budget; tie-breaker determinism | `PipelineContractError` |

---

## 4. Configuration Reference: `PipelineConfig`

The pipeline is configured via [`PipelineConfig`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/pipeline_config.py):

```python
@dataclass(frozen=True)
class PipelineConfig:
    activity_data_path: Path = Path("data/raw/campus_activity.csv")
    emission_factors_path: Path = Path("data/emission_factors.csv")
    output_dir: Path = Path("data/processed/pipeline_run")
    optimization_budget_inr: float = 2_500_000.0   # ₹25 Lakhs default
    random_seed: int = 42
    train_months: int = 24
    forecast_horizon_months: int = 12
    uncertainty_simulations: int = 1000
    baseline_year: int = 2025
    fail_fast: bool = True
    export_artifacts_on_finish: bool = True
```

### JSON Configuration Example
```json
{
  "activity_data_path": "data/raw/campus_activity.csv",
  "emission_factors_path": "data/emission_factors.csv",
  "output_dir": "data/processed/pipeline_run",
  "optimization_budget_inr": 2500000.0,
  "random_seed": 42,
  "train_months": 24,
  "forecast_horizon_months": 12,
  "uncertainty_simulations": 1000,
  "baseline_year": 2025,
  "fail_fast": true,
  "export_artifacts_on_finish": true
}
```

---

## 5. Execution Instructions

### Option A: Command-Line Interface (CLI)

```bash
# 1. Full standard run (1000 Monte Carlo simulations, ₹25 Lakh budget)
python -m src.pipeline

# 2. Fast verification run (50 Monte Carlo simulations, takes ~15 seconds)
python -m src.pipeline --simulations 50

# 3. Custom budget run (₹10 Lakhs efficiency budget)
python -m src.pipeline --budget 1000000 --simulations 100

# 4. Custom JSON configuration run
python -m src.pipeline --config config/my_pipeline_config.json
```

### Option B: Python Programmatic API

```python
from pathlib import Path
from src.pipeline import run_pipeline
from src.pipeline_config import PipelineConfig

# 1. Define configuration
config = PipelineConfig(
    optimization_budget_inr=1_000_000.0,  # ₹10 Lakhs
    uncertainty_simulations=100,           # Fast run
    random_seed=42,
)

# 2. Execute pipeline
result = run_pipeline(config)

# 3. Access results
print("Total Historical:", result.accounting_summary.total_emissions_kg, "kgCO2e")
print("Optimal Portfolio:", result.optimal_portfolio.portfolio_id)
print("Projects:", result.optimal_portfolio.selected_interventions)
print("Carbon Avoided:", result.optimal_portfolio.total_absolute_reduction_kg, "kgCO2e")
print("Total Cost: INR", result.optimal_portfolio.total_cost_inr)
```

### Option C: Running Integration & Unit Tests

```bash
# Run Phase 16 integration tests
pytest tests/test_pipeline_integration.py -v

# Run the complete test suite across all 16 phases
pytest tests/ -v
```

---

## 6. Output Artifacts Directory Structure

When `export_artifacts_on_finish=True` (default), the pipeline flushes the following files to `output_dir`:

```
data/processed/pipeline_run/
├── cleaned_activity.csv                  # Validated activity records (180 rows)
├── campus_emissions.csv                  # Record-level calculated GHG emissions
├── historical_monthly_emissions.csv      # Monthly aggregated campus emissions
├── forecast_projections.csv              # 12-month projections with 95% PI bounds
├── uncertainty_variance_decomposition.csv # OAT variance shares (Forecast vs Act vs EF)
├── uncertainty_monte_carlo_simulation.csv # 12-month MC simulation confidence interval
├── scenario_evaluations.csv              # Phase 14 One-At-a-Time intervention impacts
├── portfolio_optimization_all_32.csv     # Complete 32-portfolio solution space
└── pipeline_summary.json                 # Consolidated metadata and optimal portfolio
```

---

## 7. Reproducibility & Determinism

The pipeline guarantees strict, byte-for-byte reproducibility:
1. **Centrally Injected Seed:** `config.random_seed` is passed explicitly to `HoltWintersForecaster(random_state=seed)` and `UncertaintyParameters(seed=seed)`.
2. **Deterministic Powerset Ordering:** Portfolios are generated via `itertools.combinations` in fixed order ($0$ to $5$ projects).
3. **Deterministic 4-Level Tie-Breaker:** Resolves optimization conflicts monotonically (Reduction $\downarrow$, Cost $\uparrow$, Projects $\uparrow$, Lexicographical ID $\uparrow$).
4. **No Current-Time Dependency:** All horizon filtering explicitly keys off the historical dataset's timestamps and configured target year (`baseline_year=2025`), ensuring identical results regardless of system clock.

---

## 8. Limitations & Scope Boundaries

1. **Academic Prototype Assumptions:** Activity relative standard deviations (RSD) and capital costs are based on approved prototype assumptions.
2. **Computational Scaling:** Running 1,000 Monte Carlo iterations across 3 targets requires fitting 3,000 Holt-Winters models ($\sim 6\text{ minutes}$ in single-threaded Python). Fast runs (`--simulations 50` or `100`) are provided for routine verification.
3. **Out of Scope (Reserved for Phase 17+):** Streamlit web dashboard visualizers, cloud database adapters, and interactive toggle UI components.
