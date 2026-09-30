# Campus Carbon Forecasting and Decarbonization Scenario Analytics (BDS-36)

This project is an academic prototype designed to perform comprehensive carbon accounting, forecasting, and decarbonization scenario analytics for campus environments. 

## Features
- **Carbon Accounting & Validation**: Accepts, cleans, and validates realistic synthetic campus activity data across primary categories (Electricity, Travel, Waste, Procurement), calculating emissions using documented factors.
- **Forecasting & Uncertainty**: Analyzes historical emissions and forecasts future energy and travel emissions using statistical models (e.g., Holt-Winters) compared against naive baselines. Incorporates Monte Carlo simulations for uncertainty estimates.
- **Scenario Optimization**: Simulates decarbonization interventions (e.g., renewable energy adoption) and compares intervention portfolios to maximize emission reductions under user-defined budget constraints.
- **Interactive Dashboard**: Provides a dynamic, interactive frontend powered by Streamlit to explore historical data, forecasts, and budget optimizations.

## Project Structure
- `src/`: Core Python packages
  - `forecasting/`: Forecasting models (baselines, stats models) and evaluation metrics
  - `scenarios/`: Intervention modeling and budget optimization
  - `uncertainty/`: Monte Carlo simulations
  - `dashboard/`: Streamlit application and UI components
- `data/`: Datasets for analysis and scenarios
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `tests/`: Automated unit testing framework
- `docs/` & `reports/`: Documentation and generated analysis reports

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd campus-carbon-analytics
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard Locally

To run the Streamlit dashboard on your local machine, execute:
```bash
streamlit run src/dashboard/app.py
```
The dashboard will open automatically in your web browser, typically at `http://localhost:8501`.

## Deployment

This dashboard is fully configured for deployment on platforms like **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
3. Select this repository and set the **Main file path** to `src/dashboard/app.py`.
4. Click **Deploy!** The platform will automatically install the required packages from `requirements.txt` and launch the app.

## Academic Context
T.Y. B.Sc. Data Science Semester V (Academic Year 2026-27). Developed according to the specifications outlined in `PROJECT_SPEC.md`.

> **Note**: This is an academic prototype utilizing synthetic campus data. Simulated data is clearly distinguished from externally sourced emission factors. Do not use for production financial systems or real-world policy enforcement without verification.
