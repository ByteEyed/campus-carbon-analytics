"""
Campus Carbon Analytics - Dashboard Visual Components
=====================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 17: Interactive Plotly Visualizations & Presentation Components
Provides reusable, theme-consistent Plotly figures and metric blocks
strictly adhering to the academic color palette and unit conventions.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Academic Color Palette Specification
COLOR_HISTORICAL = "#2b5c8f"      # Academic Navy Blue
COLOR_FORECAST = "#d95f02"        # Rust / Orange
COLOR_INTERVAL = "rgba(217, 95, 2, 0.22)"  # Translucent Rust
COLOR_FEASIBLE = "#2b5c8f"        # Blue Dots
COLOR_INFEASIBLE = "#b0bec5"      # Muted Grey Dots
COLOR_OPTIMAL = "#f1a340"         # Gold Highlight

CATEGORY_COLORS: dict[str, str] = {
    "Electricity": "#2b5c8f",
    "Travel": "#e66101",
    "Waste": "#5e3c99",
    "Procurement": "#998ec3",
}

CATEGORY_FIELD_MAP: dict[str, str] = {
    "Electricity": "electricity_emissions_kg",
    "Travel": "travel_emissions_kg",
    "Waste": "waste_emissions_kg",
    "Procurement": "procurement_emissions_kg",
}


def create_category_donut_chart(contributions_dict: dict[str, float]) -> go.Figure:
    """
    Render executive donut chart showing percentage contribution by category.

    Parameters
    ----------
    contributions_dict : dict[str, float]
        Mapping from Category Name -> percentage share (0.0 to 100.0).

    Returns
    -------
    go.Figure
        Plotly Donut Chart figure.
    """
    labels = list(contributions_dict.keys())
    values = [float(v) for v in contributions_dict.values()]
    colors = [CATEGORY_COLORS.get(lbl, "#7f7f7f") for lbl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13, family="sans-serif"),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="<b>Emissions Share by Activity Category</b>", font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        template="plotly_white",
    )
    return fig


def create_48m_timeline_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str = "total_emissions_kg",
) -> go.Figure:
    """
    Render condensed 48-month transition timeline (36m historical + 12m forward projections).

    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical monthly aggregated emissions (36 rows).
    forecast_df : pd.DataFrame
        Forecast projections DataFrame (filtered to requested target).
    target : str, default "total_emissions_kg"
        Target emission variable.
    """
    fig = go.Figure()

    # 1. Historical monthly emissions trace
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(historical_df["date"]),
            y=historical_df[target] / 1000.0,
            mode="lines+markers",
            name="Historical (2023–2025)",
            line=dict(color=COLOR_HISTORICAL, width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Historical: %{y:,.2f} MTCO₂e<extra></extra>",
        )
    )

    # 2. Forward forecast trace
    target_fc = forecast_df[forecast_df["target"] == target].sort_values("date")
    if not target_fc.empty:
        # Connect historical last point to forecast first point
        last_hist_date = pd.to_datetime(historical_df["date"].iloc[-1])
        last_hist_val = float(historical_df[target].iloc[-1]) / 1000.0

        fc_dates = [last_hist_date] + list(pd.to_datetime(target_fc["date"]))
        fc_pts = [last_hist_val] + list(target_fc["point_forecast"] / 1000.0)

        fig.add_trace(
            go.Scatter(
                x=fc_dates,
                y=fc_pts,
                mode="lines+markers",
                name="Forecast Baseline (2026)",
                line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Projected: %{y:,.2f} MTCO₂e<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="<b>48-Month Institutional Emissions Trajectory (2023–2026)</b>", font=dict(size=16)),
        xaxis=dict(title="Calendar Timeline", showgrid=True),
        yaxis=dict(title="Emissions (MTCO₂e / Metric Tonnes)", showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def create_stacked_monthly_bar_chart(historical_df: pd.DataFrame) -> go.Figure:
    """
    Render stacked bar chart of monthly emissions broken down by category.
    """
    fig = go.Figure()

    dates = pd.to_datetime(historical_df["date"])

    for cat_name, col_name in CATEGORY_FIELD_MAP.items():
        if col_name in historical_df.columns:
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=historical_df[col_name] / 1000.0,
                    name=cat_name,
                    marker=dict(color=CATEGORY_COLORS.get(cat_name, "#333333")),
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{cat_name}: %{{y:,.2f}} MTCO₂e<extra></extra>",
                )
            )

    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Historical Monthly Emissions by Category (MTCO₂e)</b>", font=dict(size=16)),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Total Emissions (MTCO₂e)", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def create_forecast_trajectory_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str = "total_emissions_kg",
) -> go.Figure:
    """
    Render interactive Plotly line chart with shaded 95% prediction interval polygon.
    """
    fig = go.Figure()

    # 1. Historical series
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(historical_df["date"]),
            y=historical_df[target],
            mode="lines+markers",
            name="Observed Emissions (kgCO₂e)",
            line=dict(color=COLOR_HISTORICAL, width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Observed: %{y:,.1f} kgCO₂e<extra></extra>",
        )
    )

    # 2. Forecast and 95% Prediction Interval Polygon
    target_fc = forecast_df[forecast_df["target"] == target].sort_values("date")
    if not target_fc.empty:
        fc_dates = pd.to_datetime(target_fc["date"])
        pt = target_fc["point_forecast"]
        lb = target_fc["lower_bound_95"]
        ub = target_fc["upper_bound_95"]

        # Upper bound line (invisible)
        fig.add_trace(
            go.Scatter(
                x=fc_dates,
                y=ub,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound line with fill to upper bound
        fig.add_trace(
            go.Scatter(
                x=fc_dates,
                y=lb,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=COLOR_INTERVAL,
                name="95% Prediction Interval",
                hovertemplate="<b>%{x|%b %Y}</b><br>Lower Bound (95%): %{y:,.1f} kgCO₂e<extra></extra>",
            )
        )

        # Point forecast trace
        fig.add_trace(
            go.Scatter(
                x=fc_dates,
                y=pt,
                mode="lines+markers",
                name="Holt-Winters Point Forecast",
                line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.1f} kgCO₂e<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="<b>Primary Holt-Winters Forecast & 95% Prediction Intervals</b>", font=dict(size=16)),
        xaxis=dict(title="Calendar Date", showgrid=True),
        yaxis=dict(title="Emissions (kgCO₂e)", showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def create_variance_decomposition_chart(
    uncertainty_decomp_df: pd.DataFrame,
    target: str = "total_emissions_kg",
) -> go.Figure:
    """
    Render horizontal stacked bar chart showing One-at-a-Time variance decomposition.
    """
    target_row = uncertainty_decomp_df[uncertainty_decomp_df["target"] == target]
    if target_row.empty:
        target_row = uncertainty_decomp_df.iloc[0:1]

    row = target_row.iloc[0]
    pct_fc = float(row.get("forecast_variance_pct", 0.0))
    pct_act = float(row.get("activity_variance_pct", 0.0))
    pct_ef = float(row.get("ef_variance_pct", 0.0))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Variance Contribution"],
            x=[pct_fc],
            name="Forecast State-Space Error (V_forecast)",
            orientation="h",
            marker=dict(color="#2b5c8f"),
            text=f"{pct_fc:.1f}%",
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Variance Contribution"],
            x=[pct_act],
            name="Activity Data Uncertainty (V_act)",
            orientation="h",
            marker=dict(color="#f1a340"),
            text=f"{pct_act:.1f}%",
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Variance Contribution"],
            x=[pct_ef],
            name="Emission Factor Uncertainty (V_ef)",
            orientation="h",
            marker=dict(color="#e66101"),
            text=f"{pct_ef:.1f}%",
            textposition="inside",
        )
    )

    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Uncertainty Variance Decomposition (OAT Sensitivity)</b>", font=dict(size=15)),
        xaxis=dict(title="Share of Total Variance (%)", range=[0, 100], showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        template="plotly_white",
        height=220,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def create_scenario_reduction_chart(scenarios_df: pd.DataFrame) -> go.Figure:
    """
    Render bar chart comparing standalone carbon reduction and MAC across the 5 interventions.
    """
    df_sorted = scenarios_df.sort_values("absolute_reduction_kg", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=df_sorted["intervention_name"],
            x=df_sorted["absolute_reduction_kg"] / 1000.0,
            orientation="h",
            marker=dict(color="#2b5c8f"),
            name="Avoided Carbon (MTCO₂e)",
            text=df_sorted.apply(
                lambda r: f"{r['absolute_reduction_kg']/1000.0:,.1f} MT ({r['percentage_reduction']:.1f}%) | MAC: ₹{r['cost_per_kg_reduced']:.1f}/kg",
                axis=1,
            ),
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Carbon Avoided: %{x:,.2f} MTCO₂e<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Standalone Intervention Carbon Abatement (MTCO₂e / Year)</b>", font=dict(size=16)),
        xaxis=dict(title="Annual Emissions Avoided (MTCO₂e)", showgrid=True),
        yaxis=dict(title="Intervention Initiative"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_efficient_frontier_chart(
    portfolio_df: pd.DataFrame,
    optimal_portfolio_id: str,
    budget_inr: float,
) -> go.Figure:
    """
    Render Plotly scatter plot of the 32-portfolio Efficient Frontier.

    X-axis: Implementation Capital Cost (INR ₹)
    Y-axis: Total Absolute Emissions Reduction (kgCO2e)
    Highlights optimal portfolio with a distinct Gold Star.
    """
    fig = go.Figure()

    # 1. Infeasible portfolios (cost > budget)
    infeasible_df = portfolio_df[~portfolio_df["is_feasible"]].copy()
    if not infeasible_df.empty:
        fig.add_trace(
            go.Scatter(
                x=infeasible_df["total_cost_inr"] / 100000.0,
                y=infeasible_df["total_absolute_reduction_kg"] / 1000.0,
                mode="markers",
                name="Infeasible (Over Budget)",
                marker=dict(color=COLOR_INFEASIBLE, size=9, opacity=0.7),
                text=infeasible_df["selected_interventions"],
                customdata=infeasible_df[["portfolio_id", "percentage_reduction", "cost_per_kg_reduced"]],
                hovertemplate=(
                    "<b>%{customdata[0]} (Infeasible)</b><br>"
                    "Cost: ₹%{x:,.1f} Lakhs<br>"
                    "Reduction: %{y:,.1f} MTCO₂e (%{customdata[1]:.1f}%)<br>"
                    "MAC: ₹%{customdata[2]:,.1f}/kg<br>"
                    "Projects: %{text}<extra></extra>"
                ),
            )
        )

    # 2. Feasible portfolios (cost <= budget) excluding the optimal one
    feasible_df = portfolio_df[portfolio_df["is_feasible"] & (portfolio_df["portfolio_id"] != optimal_portfolio_id)].copy()
    if not feasible_df.empty:
        fig.add_trace(
            go.Scatter(
                x=feasible_df["total_cost_inr"] / 100000.0,
                y=feasible_df["total_absolute_reduction_kg"] / 1000.0,
                mode="markers",
                name="Feasible Portfolios",
                marker=dict(color=COLOR_FEASIBLE, size=10, opacity=0.85),
                text=feasible_df["selected_interventions"],
                customdata=feasible_df[["portfolio_id", "percentage_reduction", "cost_per_kg_reduced"]],
                hovertemplate=(
                    "<b>%{customdata[0]} (Feasible)</b><br>"
                    "Cost: ₹%{x:,.1f} Lakhs<br>"
                    "Reduction: %{y:,.1f} MTCO₂e (%{customdata[1]:.1f}%)<br>"
                    "MAC: ₹%{customdata[2]:,.1f}/kg<br>"
                    "Projects: %{text}<extra></extra>"
                ),
            )
        )

    # 3. Optimal portfolio highlighted as Gold Star
    opt_row = portfolio_df[portfolio_df["portfolio_id"] == optimal_portfolio_id]
    if not opt_row.empty:
        r = opt_row.iloc[0]
        cost_lakhs = float(r["total_cost_inr"]) / 100000.0
        red_mt = float(r["total_absolute_reduction_kg"]) / 1000.0
        fig.add_trace(
            go.Scatter(
                x=[cost_lakhs],
                y=[red_mt],
                mode="markers+text",
                name="★ Selected Optimal Portfolio",
                marker=dict(color=COLOR_OPTIMAL, size=20, symbol="star", line=dict(color="#000000", width=1.5)),
                text=[f"  <b>Optimal: {r['portfolio_id']}</b>"],
                textposition="top right",
                textfont=dict(size=13, color="#000000"),
                customdata=[[r["portfolio_id"], r["percentage_reduction"], r["cost_per_kg_reduced"]]],
                hovertemplate=(
                    "<b>★ OPTIMAL: %{customdata[0]}</b><br>"
                    "Cost: ₹%{x:,.1f} Lakhs<br>"
                    "Reduction: %{y:,.1f} MTCO₂e (%{customdata[1]:.1f}%)<br>"
                    "MAC: ₹%{customdata[2]:,.1f}/kg<br>"
                    "Projects: " + str(r["selected_interventions"]) + "<extra></extra>"
                ),
            )
        )

    # 4. Budget vertical line
    budget_lakhs = budget_inr / 100000.0
    fig.add_vline(
        x=budget_lakhs,
        line_dash="dash",
        line_color="#e66101",
        annotation_text=f"Budget Cap: ₹{budget_lakhs:,.1f}L",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text="<b>Constrained Portfolio Frontier (Cost vs. Carbon Abatement)</b>", font=dict(size=16)),
        xaxis=dict(title="Capital Expenditure (INR Lakhs ₹)", showgrid=True),
        yaxis=dict(title="Avoided Emissions (MTCO₂e / Year)", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_building_emissions_bar_chart(campus_emissions_df: pd.DataFrame) -> go.Figure:
    """
    Render horizontal bar chart of cumulative emissions by facility.
    """
    bldg_summary = (
        campus_emissions_df.groupby("building", as_index=False)["total_emissions_kg"]
        .sum()
        .sort_values("total_emissions_kg", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=bldg_summary["building"],
            x=bldg_summary["total_emissions_kg"] / 1000.0,
            orientation="h",
            marker=dict(color="#2b5c8f"),
            text=bldg_summary["total_emissions_kg"].apply(lambda v: f"{v/1000.0:,.1f} MT"),
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Total: %{x:,.2f} MTCO₂e<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Total Historical Emissions by Campus Facility (MTCO₂e)</b>", font=dict(size=16)),
        xaxis=dict(title="Cumulative Emissions (MTCO₂e)", showgrid=True),
        yaxis=dict(title="Facility"),
        template="plotly_white",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig
