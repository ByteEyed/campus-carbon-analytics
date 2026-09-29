# Phase 10: Actual Baseline Execution Output & Verification Record

**Academic Context:** BDS-36 (T.Y. B.Sc. Data Science, Semester V)  
**Execution Timestamp:** 2026-09-29  
**Source Dataset:** [`data/processed/campus_emissions.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/campus_emissions.csv)  
**Execution Script:** [`src/forecasting/experiment.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/forecasting/experiment.py)  

---

## 1. Experimental Setup & Temporal Isolation

- **Time Granularity:** Monthly (`MS` frequency)
- **Forecast Horizon:** 12 Months ($h = 12$)
- **Total Dataset Observations:** 36 monthly records (2023-01-01 to 2025-12-01)
- **Training Period:** Months 1–24 (`2023-01-01` to `2024-12-01`, 24 observations)
- **Testing Period:** Months 25–36 (`2025-01-01` to `2025-12-01`, 12 observations)
- **Strict Chronological Condition:**
  - Latest Train Date: `2024-12-01`
  - Earliest Test Date: `2025-01-01`
  - Condition: `2025-01-01 > 2024-12-01` $\rightarrow$ **TRUE** (Zero temporal overlap / zero leakage)
- **Runtime Errors / Warnings:** **None (Exit Code 0)**

---

## 2. Summary Metrics Table

| Target Variable | Model | MAE (kg) | RMSE (kg) | MAPE (%) | MAE Improvement vs Flat Naive |
|---|---|---|---|---|---|
| **Total Campus Emissions** | Standard Naive (Flat) | 50,329.83 | 55,002.17 | 39.50% | — |
| (`total_emissions_kg`) | **Seasonal Naive ($m=12$)** | **2,378.85** | **2,794.15** | **1.95%** | **+95.27%** |
| **Electricity Emissions** | Standard Naive (Flat) | 36,244.20 | 39,323.57 | 36.07% | — |
| (`electricity_emissions_kg`) | **Seasonal Naive ($m=12$)** | **1,683.76** | **2,109.91** | **1.72%** | **+95.35%** |
| **Travel Emissions** | Standard Naive (Flat) | 11,774.88 | 13,197.81 | 61.25% | — |
| (`travel_emissions_kg`) | **Seasonal Naive ($m=12$)** | **620.31** | **748.56** | **3.70%** | **+94.73%** |

---

## 3. Month-by-Month Actual vs. Predicted Values (2025 Test Set)

### A. Total Campus Emissions (`total_emissions_kg`)

| Date | Actual (kg) | Standard Naive (kg) | Seasonal Naive (kg) | Error (Std Naive) | Error (Seasonal Naive) | APE (Std Naive) | APE (Seasonal Naive) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2025-01-01** | 115,886.37 | 68,772.32 | 116,666.97 | +47,114.04 | -780.61 | 40.66% | 0.67% |
| **2025-02-01** | 126,506.04 | 68,772.32 | 124,201.14 | +57,733.71 | +2,304.90 | 45.64% | 1.82% |
| **2025-03-01** | 134,953.99 | 68,772.32 | 131,930.82 | +66,181.66 | +3,023.16 | 49.04% | 2.24% |
| **2025-04-01** | 137,491.41 | 68,772.32 | 138,945.38 | +68,719.09 | -1,453.97 | 49.98% | 1.06% |
| **2025-05-01** | 106,775.01 | 68,772.32 | 105,135.49 | +38,002.68 | +1,639.52 | 35.59% | 1.54% |
| **2025-06-01** | 78,757.79 | 68,772.32 | 78,115.16 | +9,985.47 | +642.63 | 12.68% | 0.82% |
| **2025-07-01** | 119,601.35 | 68,772.32 | 116,779.16 | +50,829.03 | +2,822.18 | 42.50% | 2.36% |
| **2025-08-01** | 141,335.00 | 68,772.32 | 136,565.15 | +72,562.68 | +4,769.85 | 51.34% | 3.37% |
| **2025-09-01** | 135,072.70 | 68,772.32 | 134,011.43 | +66,300.38 | +1,061.27 | 49.08% | 0.79% |
| **2025-10-01** | 137,694.01 | 68,772.32 | 132,080.53 | +68,921.68 | +5,613.48 | 50.05% | 4.08% |
| **2025-11-01** | 124,497.51 | 68,772.32 | 121,945.21 | +55,725.19 | +2,552.30 | 44.76% | 2.05% |
| **2025-12-01** | 70,654.65 | 68,772.32 | 68,772.32 | +1,882.33 | +1,882.33 | 2.66% | 2.66% |

---

### B. Electricity Emissions (`electricity_emissions_kg`)

| Date | Actual (kg) | Standard Naive (kg) | Seasonal Naive (kg) | Error (Std Naive) | Error (Seasonal Naive) | APE (Std Naive) | APE (Seasonal Naive) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2025-01-01** | 88,755.91 | 58,794.75 | 89,131.63 | +29,961.16 | -375.71 | 33.76% | 0.42% |
| **2025-02-01** | 96,259.39 | 58,794.75 | 95,286.84 | +37,464.64 | +972.55 | 38.92% | 1.01% |
| **2025-03-01** | 105,527.14 | 58,794.75 | 103,389.64 | +46,732.39 | +2,137.50 | 44.28% | 2.03% |
| **2025-04-01** | 109,799.17 | 58,794.75 | 111,092.77 | +51,004.42 | -1,293.60 | 46.45% | 1.18% |
| **2025-05-01** | 90,090.45 | 58,794.75 | 88,994.62 | +31,295.70 | +1,095.83 | 34.74% | 1.22% |
| **2025-06-01** | 70,081.99 | 58,794.75 | 69,434.24 | +11,287.24 | +647.75 | 16.11% | 0.92% |
| **2025-07-01** | 98,846.44 | 58,794.75 | 96,658.06 | +40,051.69 | +2,188.38 | 40.52% | 2.21% |
| **2025-08-01** | 111,588.39 | 58,794.75 | 107,746.01 | +52,793.64 | +3,842.39 | 47.31% | 3.44% |
| **2025-09-01** | 106,316.65 | 58,794.75 | 105,851.97 | +47,521.90 | +464.68 | 44.70% | 0.44% |
| **2025-10-01** | 106,936.63 | 58,794.75 | 102,308.44 | +48,141.88 | +4,628.19 | 45.02% | 4.33% |
| **2025-11-01** | 96,125.31 | 58,794.75 | 94,911.94 | +37,330.56 | +1,213.36 | 38.84% | 1.26% |
| **2025-12-01** | 60,139.96 | 58,794.75 | 58,794.75 | +1,345.21 | +1,345.21 | 2.24% | 2.24% |

---

### C. Travel Emissions (`travel_emissions_kg`)

| Date | Actual (kg) | Standard Naive (kg) | Seasonal Naive (kg) | Error (Std Naive) | Error (Seasonal Naive) | APE (Std Naive) | APE (Seasonal Naive) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2025-01-01** | 19,735.16 | 5,241.31 | 20,249.79 | +14,493.84 | -514.64 | 73.44% | 2.61% |
| **2025-02-01** | 21,259.24 | 5,241.31 | 20,206.45 | +16,017.92 | +1,052.78 | 75.35% | 4.95% |
| **2025-03-01** | 21,007.03 | 5,241.31 | 20,272.08 | +15,765.72 | +734.95 | 75.05% | 3.50% |
| **2025-04-01** | 20,102.17 | 5,241.31 | 20,284.50 | +14,860.86 | -182.33 | 73.93% | 0.91% |
| **2025-05-01** | 11,397.84 | 5,241.31 | 11,122.46 | +6,156.53 | +275.38 | 54.01% | 2.42% |
| **2025-06-01** | 4,730.69 | 5,241.31 | 4,730.48 | -510.62 | +0.21 | 10.79% | 0.00% |
| **2025-07-01** | 13,504.33 | 5,241.31 | 12,981.11 | +8,263.02 | +523.22 | 61.19% | 3.87% |
| **2025-08-01** | 21,703.97 | 5,241.31 | 21,113.10 | +16,462.66 | +590.87 | 75.85% | 2.72% |
| **2025-09-01** | 20,947.59 | 5,241.31 | 20,609.09 | +15,706.28 | +338.49 | 74.98% | 1.62% |
| **2025-10-01** | 22,025.52 | 5,241.31 | 20,805.40 | +16,784.21 | +1,220.12 | 76.20% | 5.54% |
| **2025-11-01** | 20,989.36 | 5,241.31 | 19,507.51 | +15,748.04 | +1,481.85 | 75.03% | 7.06% |
| **2025-12-01** | 5,770.21 | 5,241.31 | 5,241.31 | +528.90 | +528.90 | 9.17% | 9.17% |

---

## 4. Persisted File Paths for Future Retrieval

- **Detailed Monthly CSV:** [`data/processed/baseline_predictions_2025.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/baseline_predictions_2025.csv)
- **Summary Metrics CSV:** [`data/processed/baseline_evaluation_summary.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/baseline_evaluation_summary.csv)
- **Complete JSON Artifact:** [`reports/phase10_baseline_actual_output.json`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/reports/phase10_baseline_actual_output.json)
- **Visual Chart (300 DPI):** [`reports/figures/10_baseline_model_comparison.png`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/reports/figures/10_baseline_model_comparison.png)
