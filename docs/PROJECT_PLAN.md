# Project Plan: Campus Carbon Analytics

## 1. Functional Requirements
The system must:
- Accept and parse static campus activity data (CSV format) encompassing electricity, travel, waste, and procurement.
- Validate and clean the ingested data, handling missing values and enforcing logical constraints.
- Calculate historical carbon emissions using clearly documented, externally sourced emission factors.
- Perform historical analysis and aggregate emissions by category and time period.
- Forecast future emissions for energy and travel using a naive baseline and one primary statistical model.
- Provide uncertainty estimates (confidence intervals) for forecasted values.
- Simulate decarbonization interventions (e.g., solar panel installation, EV fleet adoption).
- Compare and select an optimal portfolio of interventions that maximizes carbon reduction subject to a user-defined financial budget.
- Present all results, metrics, and interactive scenario planning tools via a Streamlit dashboard.

## 2. Non-Functional Requirements
- **Simplicity:** Code must be simple, readable, and prioritize straightforward implementations over complex design patterns.
- **Reproducibility:** Provide a reproducible setup environment (e.g., `requirements.txt` or `environment.yml`).
- **Transparency:** Clearly separate and mark simulated synthetic data versus externally documented real-world emission factors.
- **Logging:** Implement structured logging to track data ingestion, model execution, and errors.
- **Performance:** Calculations and optimizations must run interactively (sub-second or few seconds) within the Streamlit UI.

## 8. Forecasting Methodology
- **Scope:** Forecasting will primarily target Electricity and Travel emissions.
- **Naive Baseline:** A simple baseline model (e.g., forecasting that next year's emissions will equal this year's, or a simple moving average).
- **Primary Model:** A standard statistical time-series model suitable for academic demonstration, such as SARIMA (via `statsmodels`) or Exponential Smoothing.
- **Uncertainty:** The primary model must output confidence intervals (e.g., 95% CI) to quantify forecast uncertainty.

## 9. Scenario & Optimization Methodology
- **Scenario Simulation:** Interventions will be defined with estimated capital costs and projected annual emission reductions.
- **Optimization Strategy:** The problem will be framed as a 0/1 Knapsack Problem.
- **Solver:** We will use Integer Linear Programming (ILP) via a library like `PuLP` or `scipy.optimize`.
- **Objective:** Maximize total emissions reduced.
- **Constraint:** Total cost of selected interventions $\le$ User-defined Budget.

## 13. Evaluation Metrics
- **Forecasting Metrics:**
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- **Optimization Metrics:**
  - Budget utilization percentage.
  - Total projected CO2e reduction.

## 14. Git & Branching Strategy
For a two-person team, a simplified Git Flow is recommended:
- `main` branch: Stable, production-ready code.
- `dev` branch: Integration branch for ongoing work.
- Feature branches: Created from `dev` (e.g., `feature/data-loader`, `feature/forecasting`).
- **Process:** Peer review via Pull Requests before merging feature branches into `dev`. Code must pass unit tests before merging.

## 15. Development Milestones
- **Milestone 1:** Data Pipeline & Carbon Accounting Engine
- **Milestone 2:** Forecasting Module
- **Milestone 3:** Optimization Engine
- **Milestone 4:** Streamlit Dashboard Integration
- **Milestone 5:** Final Testing, Documentation, & Handoff

## 16. Acceptance Criteria
- **Milestone 1:** System successfully ingests synthetic data, applies emission factors, and outputs correct historical CO2e totals. Data validation handles malformed files without crashing.
- **Milestone 2:** Both naive and primary forecasting models output future projections. MAE, RMSE, and MAPE are calculated. Uncertainty bounds are visible.
- **Milestone 3:** Knapsack optimizer selects the highest impact interventions without exceeding the inputted budget.
- **Milestone 4:** Streamlit app runs locally, allowing users to upload data, view forecasts, and toggle budget constraints interactively.
- **Milestone 5:** Unit tests pass, `pytest` coverage is adequate, structured logging works, and all documentation is finalized.

## 17. Risks & Scope Exclusions
**Risks:**
- Synthetic data may lack realistic seasonality, leading to trivial or erratic forecasting results.
- Over-engineering the optimization algorithm beyond the required scope.

**Scope Exclusions:**
- Real-time IoT data ingestion.
- Live campus API integrations.
- Complex user authentication.
- Microservices, Kubernetes, or large-scale cloud deployments.
- Custom React frontend (strictly using Streamlit).
- Production-grade financial planning systems.
