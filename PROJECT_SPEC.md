# Campus Carbon Forecasting and Decarbonization Scenario Analytics

## Project Code
BDS-36

## Academic Context
T.Y. B.Sc. Data Science Semester V
Academic Year 2026-27

## Project Goal
Build an academic prototype that:
1. Accepts campus activity data
2. Validates and cleans the data
3. Calculates carbon emissions using documented emission factors
4. Performs historical emissions analysis
5. Forecasts future energy/travel emissions
6. Provides uncertainty estimates
7. Simulates decarbonization interventions
8. Compares intervention portfolios under a budget constraint
9. Presents results through an interactive Streamlit dashboard

## Data
Use realistic synthetic campus data.
Clearly distinguish simulated data from externally sourced emission factors.

## Primary Categories
- Electricity
- Travel
- Waste
- Procurement

## Forecasting
Use:
- Naive baseline
- One primary statistical forecasting model

Evaluate using MAE, RMSE and/or MAPE.

## Optimization
Maximize estimated emissions reduction subject to a user-defined budget.

## Dashboard
Use Streamlit.

## Required Evidence
- Data validation
- Carbon accounting
- Forecast baseline comparison
- Forecast uncertainty
- Scenario comparison
- Budget-constrained portfolio selection
- Failure-mode testing
- Unit tests
- Reproducible setup
- Structured logging
- Security/privacy evidence
- Documentation
- Individual contribution evidence

## Scope Restrictions
Do not implement unless explicitly requested:
- Real-time IoT
- Live campus integrations
- Complex authentication
- Microservices
- Kubernetes
- React frontend
- Production financial systems
- Large-scale cloud infrastructure

## Engineering Rule
Prefer simple, understandable implementations over unnecessary complexity.

## AI Coding Rule
Never silently invent domain facts, emission factors, costs or experimental results.
Mark simulated assumptions explicitly.
Never claim an experiment was performed unless it was actually executed.