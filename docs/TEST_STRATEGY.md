# Test Strategy: Campus Carbon Analytics

## 11. Testing Strategy

Given the scope of a two-person academic prototype, the testing strategy focuses on correctness of the mathematical logic, data integrity, and basic end-to-end functionality, rather than exhaustive UI automation.

**Framework:** `pytest`

### A. Unit Testing
- **Accounting Engine:** Tests to ensure `quantity * emission_factor` calculations are strictly correct. Includes tests for unit mismatches (e.g., ensuring a clear error is raised if units do not match).
- **Forecasting Models:** Mock historical arrays to verify that the models fit without syntax errors and return predictions with the expected shape and data types (point forecasts + uncertainty bounds).
- **Optimization Engine:** Provide known "toy" knapsack problems to the solver. Verify that it selects the mathematically optimal portfolio and strictly respects the budget constraint.

### B. Failure-Mode Testing
The system must gracefully handle bad data without crashing the Streamlit UI. Tests will inject:
- Missing critical columns (e.g., missing `quantity` or `emission_factor`).
- Negative values where physically impossible (e.g., negative energy consumption).
- Unrecognized subcategories lacking corresponding emission factors.
- Zero or negative budgets in the optimization engine.

### C. Integration Testing
- Verify the data flow from CSV ingestion $\rightarrow$ Data Cleaning $\rightarrow$ Carbon Accounting Engine.
- Ensure that the resulting DataFrame matches expected totals.

### D. Reproducibility & Environment Testing
- Ensure the application runs successfully from a clean virtual environment using the provided `requirements.txt`.
- Structured logs must capture the testing and startup sequence, verifying that the logging configuration functions correctly.

### E. Manual UI Testing (Dashboard)
- Developers will manually verify Streamlit dashboard responsiveness, ensuring charts render correctly and the application state does not break when toggling between models or adjusting budget sliders.
