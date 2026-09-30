"""
Campus Carbon Analytics - Executive Dashboard Application
=========================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 17: Executive Dashboard (Streamlit Presentation Layer)
Orchestrates a 5-page presentation interface consuming static Phase 16 pipeline artifacts.
Strictly read-only: does not execute any model training, simulation, or optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Configure module logger
logger = logging.getLogger("campus_carbon.dashboard.app")

# Ensure repository root is on sys.path regardless of launch directory or Streamlit runner
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.adapter import (
    DEFAULT_RUN_DIR,
    DashboardDataError,
    load_dashboard_data,
)
from src.dashboard.components import (
    create_48m_timeline_chart,
    create_building_emissions_bar_chart,
    create_category_donut_chart,
    create_efficient_frontier_chart,
    create_forecast_trajectory_chart,
    create_scenario_reduction_chart,
    create_stacked_monthly_bar_chart,
    create_variance_decomposition_chart,
)

# Target column mapping for UI dropdowns
TARGET_OPTIONS: dict[str, str] = {
    "Total Campus Emissions": "total_emissions_kg",
    "Electricity Emissions": "electricity_emissions_kg",
    "Travel Emissions": "travel_emissions_kg",
}


def setup_page() -> None:
    """Configure Streamlit page layout and title."""
    st.set_page_config(
        page_title="Campus Carbon Analytics | Executive Dashboard",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Custom CSS for academic, high-contrast, clean layout
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1a365d;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4a5568;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2b5c8f;
        }
        .metric-caption {
            font-size: 0.85rem;
            color: #718096;
        }
        .insight-box {
            background-color: #f7fafc;
            border-left: 4px solid #2b5c8f;
            padding: 1rem;
            border-radius: 4px;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(data: dict) -> str:
    """Render sidebar navigation and institutional metadata."""
    summary = data.get("summary", {})
    cfg = summary.get("config", {})
    opt = summary.get("optimal_portfolio", {})

    st.sidebar.markdown("## 🌿 Campus Carbon Analytics")
    st.sidebar.caption("Academic Project BDS-36 | Semester V (2026-27)")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate Dashboard Pages:",
        [
            "🏛️ Executive Overview",
            "📊 Historical Accounting",
            "📈 Forecasting & Uncertainty",
            "💡 Decarbonization Scenarios",
            "🎯 Portfolio Optimization",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Institutional Scope")
    st.sidebar.markdown(f"**Baseline Year:** `{cfg.get('baseline_year', 2025)}`")
    st.sidebar.markdown(f"**Timeline:** `2023–2025 (36 Months)`")
    st.sidebar.markdown(f"**Facilities:** `5 Campus Complexes`")
    budget_val = float(cfg.get("optimization_budget_inr", 2500000.0))
    st.sidebar.markdown(f"**Active Budget:** `₹{budget_val/100000.0:,.1f} Lakhs`")
    st.sidebar.markdown(f"**Recommended Portfolio:** `{opt.get('portfolio_id', 'P-10110')}`")

    st.sidebar.markdown("---")
    st.sidebar.caption("Data Source: Precomputed Phase 16 Integrated Pipeline Artifacts (Read-Only)")

    return page


def render_overview_page(data: dict) -> None:
    """Page 1: Executive Overview with High-Level KPIs and Condensed Horizon."""
    st.markdown('<div class="main-header">🏛️ Executive Overview & Decarbonization Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">High-level institutional carbon performance, emissions intensity, and optimal decarbonization strategy.</div>', unsafe_allow_html=True)

    summary = data["summary"]
    acct = summary.get("carbon_accounting", {})
    opt = summary.get("optimal_portfolio", {})
    cfg = summary.get("config", {})

    # Top-Level KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tot_mt = float(acct.get("total_emissions_mt", 0.0))
        st.metric(
            label="Total Historical Emissions",
            value=f"{tot_mt:,.2f} MTCO₂e",
            help="Total measured greenhouse gas emissions across all campus facilities (2023–2025).",
        )
        st.markdown('<span class="metric-caption">36-Month Cumulative Total</span>', unsafe_allow_html=True)

    with col2:
        per_student = float(acct.get("emissions_per_student_kg", 0.0))
        st.metric(
            label="Emissions Intensity",
            value=f"{per_student:,.2f} kgCO₂e",
            help="Emissions per student-month across the institution.",
        )
        st.markdown('<span class="metric-caption">Per Student-Month Intensity</span>', unsafe_allow_html=True)

    with col3:
        pct_red = float(opt.get("percentage_reduction", 0.0))
        kg_red = float(opt.get("total_absolute_reduction_kg", 0.0))
        st.metric(
            label="Optimal Carbon Abatement",
            value=f"{pct_red:.2f}%",
            delta=f"{kg_red/1000.0:,.1f} MTCO₂e/yr Avoided",
            delta_color="normal",
            help="Projected greenhouse gas reduction under the optimal budget-constrained portfolio.",
        )
        st.markdown('<span class="metric-caption">Relative to 2025 Forward Baseline</span>', unsafe_allow_html=True)

    with col4:
        cost_inr = float(opt.get("total_cost_inr", 0.0))
        rem_inr = float(opt.get("remaining_budget_inr", 0.0))
        st.metric(
            label="Required Capital Outlay",
            value=f"₹{cost_inr/100000.0:,.2f} Lakhs",
            delta=f"₹{rem_inr/100000.0:,.2f}L Unspent Surplus",
            delta_color="off",
            help="Total capital expenditure required to implement the optimal portfolio.",
        )
        st.markdown('<span class="metric-caption">Allocated from Active Budget Cap</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Main Visuals: 48-Month Horizon & Category Donut
    v_col1, v_col2 = st.columns([7, 5])
    with v_col1:
        fig_timeline = create_48m_timeline_chart(
            historical_df=data["historical_monthly"],
            forecast_df=data["forecast_projections"],
            target="total_emissions_kg",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with v_col2:
        fig_donut = create_category_donut_chart(acct.get("category_contributions", {}))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Executive Key Takeaways Callout
    st.markdown(
        """
        <div class="insight-box">
        <b>Executive Summary & Academic Findings:</b>
        <ul>
          <li><b>Dominant Scope 2 Footprint:</b> Grid electricity accounts for <b>79.77%</b> of campus emissions, establishing energy efficiency and rooftop solar as the highest-leverage decarbonization avenues.</li>
          <li><b>Budget Optimization Finding:</b> Under the ₹25 Lakh institutional grant, the optimal portfolio is <b>P-10110</b> (LED Retrofit + AC Summer Cycling + Waste Composting). It achieves <b>15.80% carbon reduction</b> (225.8 MTCO₂e) for only <b>₹9.5 Lakhs</b>, conserving ₹15.5 Lakhs in surplus capital.</li>
          <li><b>Forecasting Robustness:</b> The primary Holt-Winters model captures annual growth (2.5%) and academic seasonality with an empirical test MAPE of <b>1.27%</b> and 100% prediction interval coverage.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_accounting_page(data: dict) -> None:
    """Page 2: Carbon Accounting & Emissions Breakdown."""
    st.markdown('<div class="main-header">📊 Campus Carbon Accounting & Historical Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous Scope 1, 2, and 3 accounting adhering to CEA, India GHG, and DEFRA emission factors.</div>', unsafe_allow_html=True)

    # Visual: Monthly Stacked Bar Chart
    fig_stacked = create_stacked_monthly_bar_chart(data["historical_monthly"])
    st.plotly_chart(fig_stacked, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns([6, 6])
    with c1:
        fig_bldg = create_building_emissions_bar_chart(data["campus_emissions"])
        st.plotly_chart(fig_bldg, use_container_width=True)

    with c2:
        st.markdown("#### 🏢 Facility-Level Cumulative Metrics")
        campus_df = data["campus_emissions"]
        bldg_grp = campus_df.groupby("building").agg(
            total_emissions_mt=("total_emissions_mt", "sum"),
            avg_students=("student_count", "mean"),
            avg_intensity_kg=("emissions_per_student_kg", "mean"),
        ).reset_index()
        bldg_grp["total_emissions_mt"] = bldg_grp["total_emissions_mt"].round(2)
        bldg_grp["avg_intensity_kg"] = bldg_grp["avg_intensity_kg"].round(2)
        bldg_grp["avg_students"] = bldg_grp["avg_students"].astype(int)

        bldg_grp.columns = ["Campus Facility", "Cumulative MTCO₂e", "Mean Student Headcount", "Intensity (kg/student)"]
        st.dataframe(
            bldg_grp.sort_values("Cumulative MTCO₂e", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### 🔍 Top 10 Monthly Facility Emissions Drivers")
    top_drivers = (
        campus_df[["date", "building", "electricity_emissions_kg", "travel_emissions_kg", "waste_emissions_kg", "total_emissions_kg", "total_emissions_mt"]]
        .sort_values("total_emissions_kg", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_drivers.columns = [
        "Month",
        "Facility",
        "Electricity (kg)",
        "Travel (kg)",
        "Waste (kg)",
        "Total (kgCO₂e)",
        "Total (MTCO₂e)",
    ]
    st.dataframe(top_drivers, use_container_width=True)


def render_forecasting_page(data: dict) -> None:
    """Page 3: Primary Forecasting Model & Uncertainty Analysis."""
    st.markdown('<div class="main-header">📈 Time Series Forecasting & Uncertainty Quantification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exponential Smoothing (Holt-Winters) forward projections with state-space simulated prediction intervals.</div>', unsafe_allow_html=True)

    # Target selection dropdown
    selected_label = st.selectbox(
        "Select Emissions Time Series to Inspect:",
        options=list(TARGET_OPTIONS.keys()),
        index=0,
    )
    target_col = TARGET_OPTIONS[selected_label]

    # Model Performance Metrics
    metrics_df = data.get("forecast_metrics")
    if metrics_df is not None and not metrics_df.empty:
        target_metrics = metrics_df[
            (metrics_df["target"] == target_col) & (metrics_df["model"].str.contains("Holt-Winters"))
        ]
        if not target_metrics.empty:
            m_row = target_metrics.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Primary Model", "Holt-Winters (Add/Add)")
            with m2:
                st.metric("Mean Absolute Error (MAE)", f"{float(m_row['mae_kg']):,.1f} kgCO₂e")
            with m3:
                st.metric("Root Mean Squared Error (RMSE)", f"{float(m_row['rmse_kg']):,.1f} kgCO₂e")
            with m4:
                st.metric("Mean Absolute % Error (MAPE)", f"{float(m_row['mape_pct']):.2f}%", delta="Beats Naive Baseline")

    # Main Forecast Trajectory with Shaded 95% Confidence Intervals
    fig_fc = create_forecast_trajectory_chart(
        historical_df=data["historical_monthly"],
        forecast_df=data["forecast_projections"],
        target=target_col,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # Uncertainty Quantification (Variance Decomposition)
    st.markdown("#### 🔬 One-at-a-Time (OAT) Monte Carlo Variance Decomposition")
    st.caption("Quantifies the isolated contribution of Forecast State-Space Error, Activity Measurement Error, and Emission Factor Conversion Error to total forecast uncertainty.")

    fig_var = create_variance_decomposition_chart(
        uncertainty_decomp_df=data["uncertainty_decomp"],
        target=target_col,
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # 12-Month Monte Carlo Simulation Trajectory Table
    st.markdown("#### 📅 2026 Forecast Schedule with 95% Prediction Bounds")
    fc_sub = data["forecast_projections"][data["forecast_projections"]["target"] == target_col].copy()
    fc_sub["uncertainty_spread_kg"] = fc_sub["upper_bound_95"] - fc_sub["lower_bound_95"]
    fc_sub = fc_sub[["date", "point_forecast", "lower_bound_95", "upper_bound_95", "uncertainty_spread_kg"]].reset_index(drop=True)
    fc_sub.columns = [
        "Forecast Date",
        "Point Forecast (kgCO₂e)",
        "Lower Bound 95% (kgCO₂e)",
        "Upper Bound 95% (kgCO₂e)",
        "Uncertainty Spread (kg)",
    ]
    st.dataframe(fc_sub, use_container_width=True, hide_index=True)


def render_scenarios_page(data: dict) -> None:
    """Page 4: Standalone Decarbonization Scenarios."""
    st.markdown('<div class="main-header">💡 Standalone Decarbonization Scenarios</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Isolated One-At-a-Time (OAT) marginal emissions reduction and cost efficiency of candidate interventions.</div>', unsafe_allow_html=True)

    scen_df = data["scenario_evaluations"]

    # Visual: Bar Chart of Standalone Reductions
    fig_scen = create_scenario_reduction_chart(scen_df)
    st.plotly_chart(fig_scen, use_container_width=True)

    st.markdown("---")

    # Detailed Table of All 5 Interventions
    st.markdown("#### 📋 Comprehensive Intervention Financial & Abatement Table")
    display_df = scen_df.copy()
    display_df["implementation_cost_lakhs"] = (display_df["implementation_cost_inr"] / 100000.0).round(2)
    display_df["absolute_reduction_mt"] = (display_df["absolute_reduction_kg"] / 1000.0).round(2)
    display_df["baseline_emissions_mt"] = (display_df["baseline_emissions_kg"] / 1000.0).round(2)
    display_df["scenario_emissions_mt"] = (display_df["scenario_emissions_kg"] / 1000.0).round(2)

    cols = [
        "intervention_id",
        "intervention_name",
        "affected_category",
        "implementation_cost_lakhs",
        "baseline_emissions_mt",
        "scenario_emissions_mt",
        "absolute_reduction_mt",
        "percentage_reduction",
        "cost_per_kg_reduced",
    ]
    renamed = display_df[cols].copy()
    renamed.columns = [
        "ID",
        "Intervention Initiative",
        "Target Category",
        "Cost (₹ Lakhs)",
        "Baseline (MT)",
        "Scenario (MT)",
        "Avoided (MT)",
        "Reduction (%)",
        "MAC (₹/kgCO₂e)",
    ]
    st.dataframe(renamed.sort_values("Avoided (MT)", ascending=False), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight-box">
        <b>Marginal Abatement Cost (MAC) Key Takeaways:</b>
        <ul>
          <li><b>Most Capital Efficient:</b> <code>INT-001</code> (LED Lighting Retrofit) achieves the lowest MAC at <b>₹3.65 / kgCO₂e</b>, followed closely by <code>INT-003</code> (AC Optimization) at <b>₹4.16 / kgCO₂e</b>.</li>
          <li><b>Highest Total Abatement:</b> LED Retrofit provides the largest physical reduction (136.9 MTCO₂e / 9.58% of campus total).</li>
          <li><b>Capital-Intensive Infrastructure:</b> Rooftop Solar (₹58.19/kg) and EV Transit (₹44.30/kg) require significantly greater capital expenditure per kilogram of carbon abated.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_optimization_page(data: dict) -> None:
    """Page 5: Constrained Portfolio Budget Optimization & Efficient Frontier."""
    st.markdown('<div class="main-header">🎯 Constrained Portfolio Budget Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exhaustive powerset evaluation (32 combinations) resolving combinatorial interactions without double-counting.</div>', unsafe_allow_html=True)

    summary = data["summary"]
    cfg = summary.get("config", {})
    opt = summary.get("optimal_portfolio", {})
    budget_inr = float(cfg.get("optimization_budget_inr", 2500000.0))
    p32_df = data["portfolio_all_32"]

    # Optimal Portfolio Callout Banner
    opt_id = opt.get("portfolio_id", "P-10110")
    cost_val = float(opt.get("total_cost_inr", 0.0))
    red_val = float(opt.get("total_absolute_reduction_kg", 0.0))
    pct_val = float(opt.get("percentage_reduction", 0.0))
    rem_val = float(opt.get("remaining_budget_inr", 0.0))
    mac_val = float(opt.get("cost_per_kg_reduced", 0.0))
    sel_invs = ", ".join(opt.get("selected_interventions", []))

    st.markdown(
        f"""
        <div class="insight-box">
        <h4 style="margin-top:0; color:#1a365d;">★ Selected Optimal Portfolio: <code>{opt_id}</code></h4>
        <p><b>Included Initiatives:</b> {sel_invs}</p>
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
          <div><b>Capital Expenditure:</b> ₹{cost_val/100000.0:,.2f} Lakhs</div>
          <div><b>Unspent Surplus:</b> ₹{rem_val/100000.0:,.2f} Lakhs</div>
          <div><b>Annual Avoided Carbon:</b> {red_val/1000.0:,.2f} MTCO₂e ({pct_val:.2f}%)</div>
          <div><b>Marginal Abatement Cost:</b> ₹{mac_val:.2f} / kgCO₂e</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual: Efficient Frontier Scatter Plot
    fig_frontier = create_efficient_frontier_chart(
        portfolio_df=p32_df,
        optimal_portfolio_id=opt_id,
        budget_inr=budget_inr,
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    st.markdown("---")

    # Interactive Portfolio Table Explorer
    st.markdown("#### 🔍 Combinatorial Powerset Solution Space (All 32 Combinations)")
    filter_mode = st.radio(
        "Display Portfolios:",
        ["Feasible Only (Within Budget)", "All 32 Portfolios (Including Over-Budget)"],
        horizontal=True,
    )

    if filter_mode.startswith("Feasible"):
        view_df = p32_df[p32_df["is_feasible"]].copy()
    else:
        view_df = p32_df.copy()

    view_df["cost_lakhs"] = (view_df["total_cost_inr"] / 100000.0).round(2)
    view_df["reduction_mt"] = (view_df["total_absolute_reduction_kg"] / 1000.0).round(2)
    view_df["cost_per_kg_reduced"] = view_df["cost_per_kg_reduced"].round(2)

    table_cols = [
        "portfolio_id",
        "selected_interventions",
        "num_interventions",
        "cost_lakhs",
        "reduction_mt",
        "percentage_reduction",
        "cost_per_kg_reduced",
        "is_feasible",
    ]
    renamed_p = view_df[table_cols].copy()
    renamed_p.columns = [
        "Portfolio ID",
        "Selected Initiatives",
        "Project Count",
        "Cost (₹ Lakhs)",
        "Avoided (MT)",
        "Reduction (%)",
        "MAC (₹/kg)",
        "Feasible?",
    ]
    st.dataframe(renamed_p.sort_values("Avoided (MT)", ascending=False), use_container_width=True, hide_index=True)


def main() -> None:
    """Main dashboard entry point."""
    setup_page()

    # Load Phase 16 precomputed pipeline artifacts
    try:
        data = load_dashboard_data(DEFAULT_RUN_DIR)
    except (DashboardDataError, FileNotFoundError) as exc:
        st.error(
            "⚠️ **Pipeline Artifacts Not Found or Incomplete**\n\n"
            "The executive dashboard is a read-only presentation interface that consumes precomputed "
            "Phase 16 pipeline outputs. Please execute the integrated pipeline before launching the dashboard:\n\n"
            "```bash\npython -m src.pipeline\n```"
        )
        with st.expander("🛠️ Technical Diagnostic Details"):
            st.code(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"⚠️ **Unexpected Error Ingesting Dashboard Data:** {exc}")
        st.stop()

    # Render Sidebar and Route Pages
    page = render_sidebar(data)

    if page == "🏛️ Executive Overview":
        render_overview_page(data)
    elif page == "📊 Historical Accounting":
        render_accounting_page(data)
    elif page == "📈 Forecasting & Uncertainty":
        render_forecasting_page(data)
    elif page == "💡 Decarbonization Scenarios":
        render_scenarios_page(data)
    elif page == "🎯 Portfolio Optimization":
        render_optimization_page(data)


if __name__ == "__main__":
    main()
