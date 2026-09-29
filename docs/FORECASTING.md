# Primary Forecasting Model Documentation: Holt-Winters Exponential Smoothing

> **Academic Project**: BDS-36 (T.Y. B.Sc. Data Science, Semester V)  
> **Academic Year**: 2026–27  
> **Course**: BDS-36 Data Science Project  
> **Phase**: Phase 11 — Primary Forecasting Model Implementation  
> **Implementation**: [`src/forecasting/holt_winters.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/forecasting/holt_winters.py)  
> **Test Suite**: [`tests/test_primary_model.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/tests/test_primary_model.py) (11 Phase 11-specific tests in test_primary_model.py (78 total forecasting tests across all test modules), 100% pass)  
> **Evaluation Artifacts**:
> - [`data/processed/primary_model_evaluation_summary.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/primary_model_evaluation_summary.csv)
> - [`data/processed/primary_model_predictions_2025.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/primary_model_predictions_2025.csv)
> - [`reports/figures/11_primary_vs_baseline_comparison.png`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/reports/figures/11_primary_vs_baseline_comparison.png)

---

## 1. Model Description

The primary forecasting engine for the Campus Carbon Analytics system is **Exponential Smoothing (Holt-Winters) with Additive Trend and Additive Seasonality ($m=12$)**.

The model is designed to generate monthly point forecasts and calibrated 95% prediction intervals across a 12-month horizon ($H=12$) for three operational targets:
1. **Total Campus Emissions** (`total_emissions_kg`): The aggregate institutional decarbonization KPI.
2. **Electricity Emissions** (`electricity_emissions_kg`): The largest single source (~78% of gross footprint) exhibiting severe summer cooling seasonality.
3. **Travel Emissions** (`travel_emissions_kg`): The commuter footprint (~16% of gross footprint) tightly coupled to the academic calendar.

Waste and Procurement emissions are not forecasted independently because their combined share represents <6% of total emissions; their minor secular contributions are modeled within the aggregate Total series.

---

## 2. Justification & Model Selection vs. Alternatives

The choice of Holt-Winters with additive trend and seasonality is grounded in the empirical statistical findings of Phase 9 (Exploratory Data Analysis) and Phase 10 (Baseline Forecasting):

1. **Dominant Annual Seasonality ($m=12$)**:
   Phase 10 proved that seasonal replication captures over 94% of the series variance, dropping MAPE from ~40% (Standard Flat Naive) to 1.95% (Seasonal Naive). A model without explicit annual seasonal periodicity cannot capture campus carbon dynamics.
2. **Correction for Secular Growth Drift**:
   Phase 10 identified that the Seasonal Naive model systematically under-predicted the 2025 test series (mean residual: $+2,006.42\text{ kgCO}_2\text{e}$ for Total, $+1,405.54\text{ kgCO}_2\text{e}$ for Electricity) because it assumed zero year-over-year drift. Phase 9 EDA uncovered a secular growth rate of approximately $\sim 2.5\%$ annually (driven by institutional headcount and facility expansion). Holt-Winters incorporates an explicit linear trend component ($\beta$) to absorb this drift.
3. **Small Sample Size ($N=36$, $N_{\text{train}}=24$)**:
   With only 24 historical training observations (2 complete annual cycles), complex models suffer from severe parameter over-determination:
   - **SARIMA $(p,d,q)(P,D,Q)_{12}$**: Requires estimating 5–8 parameters with only 2 seasonal cycles, leading to optimization instability, non-invertible MA polynomials, or catastrophic overfitting.
   - **Prophet / Neural Networks / LSTMs**: Require hundreds of observations and complex priors; LSTMs overfit instantly on $N=24$.
   - **Holt-Winters**: Solves exactly 3 smoothing parameters ($\alpha, \beta, \gamma$) via constrained maximum likelihood / SSE optimization. It is mathematically well-conditioned for $N=24$.
4. **Physical Operational Interpretability**:
   Decomposition into Level ($l_t$), Trend ($b_t$), and Seasonality ($s_t$) provides direct operational visibility into whether emission shifts are driven by baseline campus growth or cyclic calendar swings.

---

## 3. Mathematical Formulation

The additive Holt-Winters model equations for seasonal period $m=12$ are formulated as:

$$\begin{aligned}
\text{Level Equation:} \quad & l_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + b_{t-1}) \\
\text{Trend Equation:} \quad & b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1} \\
\text{Seasonal Equation:} \quad & s_t = \gamma (y_t - l_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m} \\
\text{Point Forecast:} \quad & \hat{y}_{t+h|t} = l_t + h \cdot b_t + s_{t+h - m(k+1)}
\end{aligned}$$

where:
- $y_t$ is the observed emission value at month $t$.
- $l_t$ is the estimated smoothed series level at month $t$.
- $b_t$ is the estimated linear secular trend (drift) at month $t$.
- $s_t$ is the estimated seasonal component at month $t$, constrained such that $\sum_{i=1}^m s_{t-i} \approx 0$.
- $h \in \{1, 2, \dots, 12\}$ is the forecast horizon.
- $k = \lfloor (h - 1) / m \rfloor = 0$ for a 12-month horizon.
- $\alpha, \beta, \gamma \in [0, 1]$ are the smoothing parameters for level, trend, and seasonality, respectively.

### Physical Constraints Enforced
- **Physical Non-Negativity**: Gross carbon emissions cannot be physically negative:
  $$\hat{y}_{t+h|t} \ge 0, \quad \text{lower\_bound}_{t+h} \ge 0$$
- **Interval Consistency**:
  $$\text{lower\_bound}_{t+h} \le \hat{y}_{t+h|t} \le \text{upper\_bound}_{t+h}$$

---

## 4. Parameter Estimation & Optimal Values

Parameters were estimated on the 24-month training series (`2023-01-01` to `2024-12-01`) using Nelder-Mead / L-BFGS-B bounded optimization minimizing Sum of Squared Errors (SSE):

| Target Series | Smoothing Level ($\alpha$) | Smoothing Trend ($\beta$) | Smoothing Seasonal ($\gamma$) | Optimization Method |
|---|---|---|---|---|
| **Total Campus Emissions** (`total_emissions_kg`) | `0.5575` | `0.0000` | `0.0000` | L-BFGS-B (SSE) |
| **Electricity Emissions** (`electricity_emissions_kg`) | `0.5606` | `0.0000` | `0.0000` | L-BFGS-B (SSE) |
| **Travel Emissions** (`travel_emissions_kg`) | `1.4901e-08` | `0.0000` | `0.0000` | L-BFGS-B (SSE) |

### Parameter Diagnostics
- **Level Smoothing ($\alpha \approx 0.56$)**: Total and Electricity update their baseline level moderately quickly, reflecting responsive tracking of annual load shifts.
- **Trend Smoothing ($\beta = 0.0$)**: The trend parameter settled to zero, indicating that the initial estimated linear drift ($b_0$) is constant and preserved across the forecast horizon rather than fluctuating wildly on short training noise.
- **Seasonal Smoothing ($\gamma = 0.0$)**: The seasonal indices established over the two baseline years are deterministic and stable, consistent with fixed academic semesters and annual climate cycles.
- **Travel Series Parameter Collapse ($\alpha \approx 1.49 \times 10^{-8} \approx 0, \beta = 0, \gamma = 0$)**: The optimization routine converged with the travel series smoothing parameters effectively at zero ($\alpha \approx 0$). This behavior confirms that the travel series is highly stable, with its dynamics driven almost entirely by the academic calendar. The initial seasonal decomposition captures the cyclical commuter patterns and baseline drift so comprehensively that iteration-to-iteration state updates are unnecessary. The model functions as a fixed annual seasonal pattern with preserved baseline drift, delivering a 41.42% MAE improvement over Seasonal Naive without requiring adaptive level recalculation.

---

## 5. Training & Testing Methodology

To guarantee rigorous evaluation and eliminate temporal data leakage:
- **Historical Data**: Exactly 36 consecutive monthly observations (`2023-01-01` to `2025-12-01`).
- **Training Set (Months 1–24)**: `2023-01-01` to `2024-12-01` ($N_{\text{train}} = 24$).
- **Test Set (Months 25–36)**: `2025-01-01` to `2025-12-01` ($N_{\text{test}} = 12$, Horizon $H=12$).
- **Split Property**: Strict chronological ordering ($t_{\text{train}} < t_{\text{test}}$). No random shuffling, k-fold cross-validation, or future lookahead.
- **Leakage Prevention Verification**: Formally verified in unit test `test_leakage_prevention`, confirming that corrupting future test observations produces zero change in fitted training parameters or forecasts.

---

## 6. Empirical Performance Comparison Table

All models were evaluated on the identical 12-month test set (`2025-01-01` to `2025-12-01`):

| Target Series | Model | MAE ($\text{kgCO}_2\text{e}$) | RMSE ($\text{kgCO}_2\text{e}$) | MAPE (%) | 95% PI Coverage (%) | MAE Improvement vs. Seasonal Naive (%) |
|---|---|---|---|---|---|---|
| **Total Campus Emissions** | Standard Naive (Flat) | 50,329.83 | 55,002.17 | 39.50% | — | — |
| | Seasonal Naive ($m=12$) | 2,378.85 | 2,794.15 | 1.95% | — | 0.00% (Baseline) |
| | **Holt-Winters (Primary)** | **1,548.20** | **1,975.93** | **1.27%** | **100.0%** | **+34.92%** |
| **Electricity Emissions** | Standard Naive (Flat) | 36,244.20 | 39,323.57 | 36.07% | — | — |
| | Seasonal Naive ($m=12$) | 1,683.76 | 2,109.91 | 1.72% | — | 0.00% (Baseline) |
| | **Holt-Winters (Primary)** | **1,319.22** | **1,597.13** | **1.38%** | **100.0%** | **+21.65%** |
| **Travel Emissions** | Standard Naive (Flat) | 11,774.88 | 13,197.81 | 61.25% | — | — |
| | Seasonal Naive ($m=12$) | 620.31 | 748.56 | 3.70% | — | 0.00% (Baseline) |
| | **Holt-Winters (Primary)** | **363.36** | **440.91** | **2.26%** | **100.0%** | **+41.42%** |

---

## 7. Detailed 2025 Monthly Forecasts & Uncertainty Intervals

Below are the exact empirical values for 2025 generated by the Holt-Winters primary model:

### Total Campus Emissions (`total_emissions_kg`)
| Month | Actual ($\text{kgCO}_2\text{e}$) | Forecast ($\text{kgCO}_2\text{e}$) | 95% Lower Bound | 95% Upper Bound | Residual $e_t$ ($\text{kg}$) | In 95% PI? |
|---|---|---|---|---|---|---|
| **2025-01-01** | 115,886.37 | 118,026.23 | 114,770.99 | 120,982.24 | -2,139.86 | Yes |
| **2025-02-01** | 126,506.04 | 125,579.99 | 121,957.89 | 129,180.40 | +926.05 | Yes |
| **2025-03-01** | 134,953.99 | 133,785.58 | 129,869.44 | 137,881.57 | +1,168.41 | Yes |
| **2025-04-01** | 137,491.41 | 138,524.59 | 134,155.62 | 142,753.78 | -1,033.18 | Yes |
| **2025-05-01** | 106,775.01 | 105,855.52 | 101,242.92 | 110,452.95 | +919.49 | Yes |
| **2025-06-01** | 78,757.79 | 79,603.98 | 74,791.92 | 84,422.98 | -846.19 | Yes |
| **2025-07-01** | 119,601.35 | 117,542.15 | 112,341.49 | 122,667.99 | +2,059.19 | Yes |
| **2025-08-01** | 141,335.00 | 137,002.32 | 131,703.16 | 142,429.66 | +4,332.68 | Yes |
| **2025-09-01** | 135,072.70 | 135,628.12 | 129,996.93 | 141,567.53 | -555.42 | Yes |
| **2025-10-01** | 137,694.01 | 134,016.04 | 128,168.76 | 139,883.94 | +3,677.97 | Yes |
| **2025-11-01** | 124,497.51 | 124,684.22 | 118,791.47 | 131,095.83 | -186.71 | Yes |
| **2025-12-01** | 70,654.65 | 71,387.91 | 65,333.87 | 78,107.84 | -733.26 | Yes |

*Empirical 95% Coverage*: **12 / 12 months = 100.0%**.

---

## 8. Error Analysis & Systematic Bias Reduction

A fundamental weakness discovered in the Phase 10 Seasonal Naive baseline was **one-sided positive residual bias**: because the baseline replicates year $t-1$ without drift, it systematically under-predicted every single test month.

Holt-Winters successfully remediated this bias:

| Target Series | Seasonal Naive Mean Residual ($\text{kgCO}_2\text{e}$) | Holt-Winters Mean Residual ($\text{kgCO}_2\text{e}$) | Systematic Bias Reduction (%) | Directional Distribution of Errors |
|---|---|---|---|---|
| **Total Campus Emissions** | $+2,006.42$ | **$+632.43$** | **-68.48%** | 6 positive, 6 negative (balanced) |
| **Electricity Emissions** | $+1,405.54$ | **$+38.88$** | **-97.23%** | 4 positive, 8 negative (near-zero mean) |
| **Travel Emissions** | $+504.15$ | **$+226.71$** | **-55.03%** | 8 positive, 4 negative (centered) |

### Residual Properties
- Holt-Winters residuals fluctuate symmetrically around zero, indicating that the additive trend component effectively absorbed the secular campus expansion rate.
- For electricity, the mean error collapsed from $+1,405.54\text{ kg}$ to just $+38.88\text{ kg}$ (a 97.23% elimination of bias).

---

## 9. Peak Error Analysis & Operational Explanations

The largest single absolute errors were identified and correlated with campus operational schedules:

### 1. Total Campus Emissions Peak Error: August 2025 (`2025-08-01`)
- **Actual**: $141,335.00\text{ kgCO}_2\text{e}$
- **Forecast**: $137,002.32\text{ kgCO}_2\text{e}$
- **Error**: $+4,332.68\text{ kgCO}_2\text{e}$ (Percentage error: $+3.07\%$)
- **Context**: August represents the annual apex of chiller plant and HVAC load combined with administrative ramp-up and new student orientation prior to the fall semester. Cooling degree days (CDD) in late August drive nonlinear electricity consumption that exceeds a purely linear trend.

### 2. Electricity Emissions Peak Error: August 2025 (`2025-08-01`)
- **Actual**: $111,588.39\text{ kgCO}_2\text{e}$
- **Forecast**: $108,369.53\text{ kgCO}_2\text{e}$
- **Error**: $+3,218.86\text{ kgCO}_2\text{e}$ (Percentage error: $+2.88\%$)
- **Context**: Electricity accounts for $74.3\%$ of the total August forecast error. The surge reflects peak ambient temperatures across laboratory and dining facilities.

### 3. Travel Emissions Peak Error: October 2025 (`2025-10-01`)
- **Actual**: $22,025.52\text{ kgCO}_2\text{e}$
- **Forecast**: $21,176.01\text{ kgCO}_2\text{e}$
- **Error**: $+849.51\text{ kgCO}_2\text{e}$ (Percentage error: $+3.86\%$)
- **Context**: October is the month of highest commuter transit density (midterm examination period with 100% campus attendance and no scheduled holidays or term breaks).

---

## 10. Prediction Interval Validation & Calibration

Prediction intervals were generated using state-space simulation ($B=2000$ simulated paths) reflecting accumulated variance across the 12-month horizon:

- **Nominal Confidence**: 95% ($\alpha = 0.05$).
- **Empirical Coverage Observed**: **100.0%** across all three series (12 of 12 test months fell strictly inside $[\text{lower\_bound}, \text{upper\_bound}]$).
- **Physical Sandwiches**: $\text{lower\_bound}_t \le \hat{y}_t \le \text{upper\_bound}_t$ verified for all steps.
- **Physical Non-Negativity**: $\min(\text{lower\_bound}_t) = 4,052.77\text{ kgCO}_2\text{e} > 0$, confirming no physical violations.
- **Uncertainty Expansion**: Interval width naturally expands over the horizon (e.g. for Total: $\pm 3,105\text{ kg}$ in month 1 expanding to $\pm 6,386\text{ kg}$ in month 12), accurately reflecting increasing multi-step forecast uncertainty.

---

## 11. Methodological Limitations & Future Scope

While Holt-Winters achieves superior accuracy on this dataset, several technical limitations must be documented:

1. **Absence of Exogenous Weather Regressors**:
   Holt-Winters relies strictly on endogenous historical values. It cannot incorporate actual meteorological variables (e.g., Cooling Degree Days, Heating Degree Days, or wet-bulb temperature). An unusually hot summer or unseasonal cold wave cannot be anticipated until it appears in historical lags.
2. **Fixed Calendar Assumption**:
   Academic holidays and examination periods that shift by a week or two from year to year (e.g., lunar calendar holidays or varying semester start dates) are treated as identical 12-month cycles.
3. **Transition to SARIMAX / ARIMAX**:
   When $\ge 48$ months of monthly data become available (4 complete annual cycles), the model should be upgraded to **SARIMAX with exogenous temperature covariates** or **Dynamic Harmonic Regression** to capture nonlinear climatic anomalies.
4. **Intervention Scenario Engine Requirement**:
   Holt-Winters forecasts a "business-as-usual" (BAU) trajectory under secular growth. It does not simulate hypothetical structural interventions (e.g., rooftop solar photovoltaic installations, fleet electrification, or HVAC setpoint adjustments). That capability will be provided by the Phase 12 Scenario Analysis Engine.

---

## 12. Verification & Reproducibility Command

To reproduce the complete Phase 11 experimental results, generate the comparison visualization, and update all exported artifacts:

```powershell
python -m src.forecasting.holt_winters
pytest tests/test_primary_model.py -v
```
