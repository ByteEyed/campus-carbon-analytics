# Forecasting Methodology & Experiment Design

## 1. Overview
The forecasting module is designed to predict future campus emissions (primarily from Electricity and Travel) based on historical synthetic data. The synthetic dataset consists of 36 months of data (3 years) with pronounced monthly seasonality driven by the academic calendar and a modest secular growth trend.

This document outlines a simple, academically defensible time-series experiment designed to evaluate our models before using them for future projections.

## 2. Data Splitting: Time-Based Split
**Methodology:** Strict chronological split. 
- **Training Set:** First 24 months (e.g., Jan 2023 – Dec 2024).
- **Testing Set:** Final 12 months (e.g., Jan 2025 – Dec 2025).

**Why No Random Split?** 
Time-series data exhibits temporal dependence (autocorrelation). Randomly splitting the data (e.g., using `train_test_split` from `scikit-learn` without `shuffle=False`) would leak future information into the training set. This violates causality and results in artificially inflated performance metrics that will not generalize to actual future forecasting.

## 3. Naive Baseline Model
**Model:** Seasonal Naive Forecast
- **Methodology:** A standard naive model ($y_t = y_{t-1}$) assumes the next month equals the last month. However, given our strong academic seasonality, this would perform poorly. Instead, we use a **Seasonal Naive Baseline**, where the forecast for any given month is exactly equal to the observed value from the same month in the previous year ($y_t = y_{t-12}$).
- **Inputs:** The last 12 months of the training set.
- **Outputs:** Point forecasts for the 12-month test set.

## 4. Primary Forecasting Model Recommendation
**Recommendation:** **Exponential Smoothing (Holt-Winters)**

**Justification vs. SARIMA:**
1. **Dataset Size:** The dataset spans exactly 36 months (3 seasonal cycles). SARIMA requires estimating many parameters (p, d, q, P, D, Q), and applying seasonal differencing consumes a full year of data, leaving very few points for stable parameter estimation.
2. **Complexity:** Holt-Winters (Exponential Smoothing with Trend and Seasonality) is mathematically simpler, robust on short seasonal datasets, and aligns perfectly with the project's engineering rule: *"Prefer simple, understandable implementations over unnecessary complexity."*

**Methodology:**
- We will fit an Exponential Smoothing model using `statsmodels.tsa.holtwinters.ExponentialSmoothing`.
- **Trend:** Additive (to capture the ~2.5% simulated annual growth).
- **Seasonality:** Additive or Multiplicative, with a seasonal period of $m=12$.

## 5. Forecast Intervals (Uncertainty)
Academic forecasting requires quantifying uncertainty rather than just providing point estimates.
- **Methodology:** We will generate 95% Prediction Intervals (PI). 
- **Outputs:** For every predicted point, the model will output a `lower_bound` and `upper_bound`. When visualized on the Streamlit dashboard, this will appear as a shaded confidence band around the forecast line, allowing planners to see the best-case and worst-case emission scenarios.

## 6. Evaluation Metrics
The models will be evaluated on the 12-month test set using the following metrics:
- **MAE (Mean Absolute Error):** Provides the average error in actual physical units (e.g., kgCO2e). It is easy for stakeholders to understand (e.g., "We are off by 500 kg per month on average").
- **RMSE (Root Mean Square Error):** Penalizes larger errors more heavily than MAE. This is crucial for campus planning, as underestimating a massive peak month (like the start of a semester) is worse than making small errors during vacation months.
- **MAPE (Mean Absolute Percentage Error):** Provides a relative error metric (e.g., "The forecast is 4% off"). *Note: MAPE is appropriate here because our synthetic data generation explicitly prevents zero or negative values, avoiding the division-by-zero flaw of MAPE.*

## 7. Error Analysis Methodology
Beyond raw metrics, the experiment will include an error analysis phase:
1. **Baseline Comparison:** Calculate the percentage improvement of Holt-Winters over the Seasonal Naive baseline. If the complex model cannot beat the naive baseline, the naive baseline should be used.
2. **Residual Plotting:** Plot the residuals (Actual Test Values minus Forecasted Values) over time to ensure there is no systematic bias (e.g., the model shouldn't consistently under-predict summer months).
3. **Coverage Probability:** Verify what percentage of the actual test data points fall within the 95% prediction intervals. Ideally, ~11 out of the 12 test months should fall within the bounds.
