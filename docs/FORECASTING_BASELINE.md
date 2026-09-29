# Phase 10: Forecasting Baseline Design & Evaluation

**Academic Project:** BDS-36 (T.Y. B.Sc. Data Science, Semester V)  
**Academic Year:** 2026-27  
**Module:** `src/forecasting/`  
**Execution Date:** September 29, 2026  

---

## 1. Overview & Objective

Phase 10 establishes a rigorous, academically defensible baseline benchmarking framework for campus greenhouse gas emissions time-series forecasting. The goal is to determine the lower bound of predictability on historical monthly campus emissions before implementing higher-complexity statistical or machine learning models in Phase 11.

---

## 2. Forecast Targets

Three target series are forecasted independently:

1. **Total Campus Emissions (`total_emissions_kg`):**
   - The primary institutional macro-KPI for campus sustainability planning and net-zero roadmaps.
2. **Electricity Emissions (`electricity_emissions_kg`):**
   - The largest emissions category (~79.8% of historical emissions). Exhibits severe weather-dependent seasonality (summer cooling air-conditioning loads) and academic operational swings.
3. **Travel Emissions (`travel_emissions_kg`):**
   - The second largest contributor (~14.2% of emissions). Directly correlated with commuter presence during academic teaching semesters versus recess periods.

> [!NOTE]
> **Scope Exclusion:** Waste and Procurement are not forecasted individually. Procurement exhibits stochastic year-end fiscal expenditure spikes that resist simple time-series modeling, while Waste represents a minor share (~5.2%). Both are fully captured within the aggregate "Total Campus Emissions" target.

---

## 3. Granularity, Horizon & Chronological Splitting

### 3.1 Time Granularity
- **Granularity:** Monthly (`MS` frequency, start of month).
- **Justification:** Institutional activity data, electricity utility meter reads, and accounting invoices arrive on monthly billing cycles. Weekly or daily downsampling would require synthetic interpolation not grounded in the underlying records.

### 3.2 Forecast Horizon
- **Horizon ($H$):** 12 Months.
- **Justification:** 12 months covers one complete academic and climatic cycle, aligning with campus annual budgeting, capital planning, and decarbonization intervention timelines.

### 3.3 Strict Chronological Train/Test Split
- **Dataset Span:** 36 continuous months (2023-01-01 to 2025-12-01).
- **Training Set:** First 24 months (January 2023 – December 2024; $N_{\text{train}} = 24$).
- **Testing Set:** Final 12 months (January 2025 – December 2025; $N_{\text{test}} = 12$).

```
Timeline (36 Months Total):
[2023-01-01 .................... 2024-12-01] | [2025-01-01 .......... 2025-12-01]
               Training Set                  |               Test Set
             (24 Observations)               |           (12 Observations)
```

> [!IMPORTANT]
> **Why No Random Split?**
> Time-series observations possess strong autocorrelation and secular trends. Randomly partitioning points (e.g. via standard k-fold or random shuffling) leaks future information into training, invalidating causality and producing artificially inflated performance metrics that do not generalize to forward-looking deployments.

---

## 4. Baseline Model Formulations

### 4.1 Baseline 1: Standard Naive Model (Flat Persistence)
The standard naive baseline assumes future emissions remain identical to the final observed historical value:
$$y_{T+h} = y_T \quad \text{for } h = 1, 2, \dots, H$$
- **Behavior:** Takes Month 24 (December 2024) and projects it as a flat horizontal line across all 12 test months of 2025.
- **Rationale:** Establishes the absolute lower bound of predictive performance.

### 4.2 Baseline 2: Seasonal Naive Model ($m=12$)
The seasonal naive baseline projects future values equal to the corresponding month from the prior annual cycle:
$$y_{T+h} = y_{T - m + ((h - 1) \pmod m)} \quad \text{where } m = 12$$
- **Behavior:** For each month of 2025, the forecast equals the observed emissions from that same calendar month in 2024 (e.g., Forecast Jan 2025 = Actual Jan 2024).
- **Rationale:** Captures recurring institutional cycles (exam periods, summer vacation dips, semester starts) without estimating parameters.

---

## 5. Evaluation Metrics

Model performance is evaluated on the 12-month test set ($n = 12$) using:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|$$
   - Units: $\text{kgCO}_2\text{e}$. Direct physical interpretability for campus facility managers.

2. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}$$
   - Units: $\text{kgCO}_2\text{e}$. Penalizes large peak misses more heavily than small off-peak discrepancies.

3. **Mean Absolute Percentage Error (MAPE):**
   $$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$
   - Units: $\%$. Relative accuracy metric. Protected by explicit zero-value guardrails in `src/forecasting/metrics.py`.

4. **Relative Seasonal Improvement:**
   $$\text{Improvement} = \frac{\text{Metric}_{\text{Naive}} - \text{Metric}_{\text{Seasonal Naive}}}{\text{Metric}_{\text{Naive}}} \times 100\%$$

---

## 6. Leakage Prevention Controls

To prevent data contamination:
1. **Isolated Interfaces:** The model fitting methods (`NaiveBaseline.fit` and `SeasonalNaiveBaseline.fit`) receive **strictly** the 24-month training series. The test dataset is never referenced during model fitting.
2. **Chronological Aggregations:** Monthly facility aggregations are computed per timestamp. No global scaling, normalizations, or parameters are computed across the full 36-month timeline prior to splitting.
3. **Automated Leakage Testing:** `tests/test_baselines.py` and `tests/test_experiment.py` contain dedicated leakage unit tests verifying that mutating future test observations results in **zero change** to model predictions.

---

## 7. Actual Experimental Results

The baseline evaluation pipeline was executed on the validated dataset via `src/forecasting/experiment.py`:

| Target Series | Model | MAE (kg) | RMSE (kg) | MAPE (%) | MAE Improvement vs Flat Naive |
|---|---|---|---|---|---|
| **Total Campus Emissions** | Standard Naive (Flat) | 50,329.83 | 55,002.17 | 39.50% | — |
| | **Seasonal Naive ($m=12$)** | **2,378.85** | **2,794.15** | **1.95%** | **+95.27%** |
| **Electricity Emissions** | Standard Naive (Flat) | 36,244.20 | 39,323.57 | 36.07% | — |
| | **Seasonal Naive ($m=12$)** | **1,683.76** | **2,109.91** | **1.72%** | **+95.35%** |
| **Travel Emissions** | Standard Naive (Flat) | 11,774.88 | 13,197.81 | 61.25% | — |
| | **Seasonal Naive ($m=12$)** | **620.31** | **748.56** | **3.70%** | **+94.73%** |

*All results recorded from physical pipeline execution on 2026-09-29.*

### Empirical Insights
1. **Catastrophic Failure of Standard Naive:**
   - Standard Naive projects Month 24 (December 2024, a vacation trough of ~75,000 kg) across the entirety of 2025. It fails to anticipate the spring/summer peak loads (>130,000 kg), producing massive errors (~50,330 kg MAE, 39.5% MAPE).
2. **Dominance of Seasonality:**
   - Seasonal Naive achieves an extraordinary **>94.7% error reduction** across all categories, reducing Total Emissions MAPE to **1.95%**. This quantitatively confirms the Phase 9 EDA conclusion that the academic and climate calendar drives primary variance.

---

## 8. Artifacts Generated

1. **Summary Metrics Table:**
   - [`data/processed/baseline_evaluation_summary.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/baseline_evaluation_summary.csv)
2. **Comparison Visualization:**
   - [`reports/figures/10_baseline_model_comparison.png`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/reports/figures/10_baseline_model_comparison.png) (3 panels showing historical training data, actual test observations, standard naive flat projections, and seasonal naive lag curves).

---

## 9. Baseline Limitations & Recommendations for Phase 11

### 9.1 Known Limitations
- **Growth Trend Underestimation:**
  Seasonal Naive captures cyclical swings but assumes zero net year-over-year change ($y_t = y_{t-12}$). Because the campus experiences ~2.5% secular growth (increasing enrollment and research activity), Seasonal Naive systematically under-predicts the 2025 actuals.
- **Uncertainty Quantification Absence:**
  Point baseline models provide no formal confidence or prediction intervals for institutional risk planning.

### 9.2 Recommendation for Phase 11
The benchmark established by Phase 10 is:
$$\text{Target to Beat: Seasonal Naive MAE} = 2,378.85\text{ kg (1.95\% MAPE)}$$
To surpass this benchmark in Phase 11, the primary statistical model must simultaneously model:
1. **Additive Seasonal Cycle ($m=12$):** To capture semester and cooling swings.
2. **Additive Trend ($\beta$):** To overcome the secular under-prediction of ~2.5% annual growth.
3. **Uncertainty Intervals:** 95% state-space prediction intervals.
