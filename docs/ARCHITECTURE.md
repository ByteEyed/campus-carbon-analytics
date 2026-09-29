# System Architecture: Campus Carbon Analytics

## 3. System Architecture

The project will be structured as a modular, monolithic Python application. It emphasizes simplicity, maintainability, and reproducibility for a two-person academic team.

**Core Stack:**
- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Forecasting:** `statsmodels` (ARIMA/SARIMA) or `prophet` (alongside naive baselines)
- **Optimization:** `scipy.optimize` or `pulp` (for linear/integer programming)
- **Frontend:** `streamlit`
- **Testing:** `pytest`

The architecture relies on static file ingestion (CSV/Excel) rather than live databases, avoiding unnecessary complexity like microservices or cloud infrastructure.

## 4. Data Flow

1. **Ingestion Layer:** The system ingests two types of CSV files:
   - Synthetic campus activity data (electricity, travel, waste, procurement).
   - Documented emission factors.
2. **Validation Layer:** Data is validated for missing values, correct data types, and logical constraints (e.g., no negative energy usage).
3. **Accounting Engine:** Activity data is multiplied by appropriate emission factors to calculate historical CO2e (carbon dioxide equivalent) emissions.
4. **Analytical Engines:**
   - **Forecasting Engine:** Consumes historical emission/activity data to train models and project future emissions with uncertainty bounds.
   - **Optimization Engine:** Consumes proposed decarbonization interventions and user-defined budget constraints to compute the optimal portfolio of interventions.
5. **Presentation Layer:** The Streamlit dashboard visualizes the outputs of the Accounting, Forecasting, and Optimization engines.

## 5. Module Boundaries

The codebase will be divided into the following loosely coupled modules:

- `src/data_pipeline/`: Responsible for loading, cleaning, and validating CSV datasets.
- `src/accounting/`: Contains pure functions for carbon calculations and unit conversions.
- `src/forecasting/`: Houses the forecasting models (naive and primary), evaluation metrics (MAE, RMSE, MAPE), and prediction logic.
- `src/optimization/`: Implements the knapsack solver for budget-constrained scenario selection.
- `src/app/`: The Streamlit application, containing UI components, routing, and state management.
- `tests/`: Unit and integration tests mirroring the `src/` structure.

## 10. Dashboard Structure

The Streamlit dashboard will feature a multi-page or tabbed layout to guide the user through the analytical process:

1. **Overview & Data Ingestion:**
   - Upload UI for activity data and emission factors.
   - High-level KPIs (Total Emissions, Top Emission Sources).
   - Historical emission trends (Bar/Line charts).
2. **Forecasting & Projections:**
   - Visual comparison of historical data vs. forecasts.
   - Toggle between Naive Baseline and Primary Model.
   - Display of forecast uncertainty (confidence intervals) and evaluation metrics (MAE, RMSE).
3. **Scenario Simulation & Optimization:**
   - Input controls for budget constraints.
   - Interactive table of available decarbonization interventions.
   - Results of the optimization engine: Selected portfolio, expected emission reductions, and budget utilization.
4. **Methodology & Assumptions:**
   - Transparent display of emission factors used.
   - Explicit disclaimers regarding synthetic data and model assumptions.

## 12. Security & Privacy Considerations

Given the scope restrictions and academic nature of this prototype:
- **Local Execution:** The dashboard is designed to run locally or in a sandbox environment. No exposure to the public internet is required.
- **No PII:** The synthetic campus data must be generated without Personally Identifiable Information (PII). Travel data should be aggregated rather than tied to individual students or staff.
- **Input Sanitization:** The data ingestion pipeline will include basic sanitization to handle malformed CSV files gracefully and prevent code injection via Pandas `eval` or similar unsafe functions (which will be strictly avoided).
- **No Complex Auth:** Authentication will not be implemented, adhering strictly to the scope exclusions.
