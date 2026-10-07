"""
Campus Carbon Analytics — Executive Dashboard Application
=========================================================

Executive Dashboard (Streamlit Presentation Layer)
--------------------------------------------------
Orchestrates a 5-page presentation interface consuming static pipeline
artifacts. Strictly read-only: does not execute any model training,
simulation, or optimization.

Presentation layer restructured around a lightweight design system:
  • Themed design tokens (light / dark) with auto-adapting neutrals
  • Card-based surfaces for charts, tables, and callouts
  • Custom KPI metric cards with accent bars, delta chips, and captions
  • Styled sidebar navigation, institutional scope panel, and notices
  • Native st.column_config table polish (progress bars, dates, currency)

Best experienced on Streamlit >= 1.29 (bordered containers); degrades
gracefully on older versions.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
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

# ═══════════════════════════════════════════════════════════════════════
# Constants & Navigation Registry
# ═══════════════════════════════════════════════════════════════════════

TARGET_OPTIONS: dict[str, str] = {
    "Total Campus Emissions": "total_emissions_kg",
    "Electricity Emissions": "electricity_emissions_kg",
    "Travel Emissions": "travel_emissions_kg",
}

NAV_TITLES: list[str] = [
    "Executive Overview",
    "Historical Accounting",
    "Forecasting & Uncertainty",
    "Decarbonization Scenarios",
    "Portfolio Optimization",
]

# ═══════════════════════════════════════════════════════════════════════
# Design System — Custom CSS
# ═══════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ─────────── Design Tokens ─────────── */
:root {
    --cc-primary: #0d9488;                       /* teal-600: legible on light & dark */
    --cc-primary-deep: #0f766e;                  /* teal-700 */
    --cc-primary-soft: rgba(13, 148, 136, 0.10);
    --cc-accent: #10b981;
    --cc-green: #059669;
    --cc-green-soft: rgba(5, 150, 105, 0.12);

    /* Neutrals auto-adapt to any Streamlit theme via currentColor */
    --cc-muted: color-mix(in srgb, currentColor 60%, transparent);
    --cc-border: color-mix(in srgb, currentColor 13%, transparent);
    --cc-surface: color-mix(in srgb, currentColor 4%, transparent);
    --cc-shadow: 0 1px 2px rgba(0, 0, 0, 0.05), 0 12px 32px rgba(0, 0, 0, 0.08);
    --cc-radius: 16px;
}
@media (prefers-color-scheme: dark) {
    :root {
        --cc-primary: #2dd4bf;
        --cc-primary-deep: #5eead4;
        --cc-accent: #34d399;
        --cc-green: #34d399;
        --cc-primary-soft: rgba(45, 212, 191, 0.12);
        --cc-green-soft: rgba(52, 211, 153, 0.14);
    }
}

/* ─────────── App Chrome ─────────── */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1360px;
    margin-inline: auto;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] { border-right: 1px solid var(--cc-border); }

/* ─────────── Page Header ─────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 1.05rem;
    flex-wrap: wrap;
    padding: 1.05rem 1.35rem;
    border: 1px solid var(--cc-border);
    border-radius: var(--cc-radius);
    background: var(--cc-surface);
    box-shadow: var(--cc-shadow);
    margin-bottom: 1.35rem;
}
.ph-badge {
    width: 52px; height: 52px; flex: 0 0 52px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; font-weight: 800; color: #ffffff;
    letter-spacing: 0.03em;
    border-radius: 14px;
    background: linear-gradient(135deg, #0f766e, #10b981);
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35);
}
.ph-body { flex: 1 1 340px; min-width: 260px; }
.ph-body h1 { font-size: 1.42rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; }
.ph-body p { margin: 0.18rem 0 0; font-size: 0.92rem; color: var(--cc-muted); line-height: 1.45; }
.ph-chips { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: center; }
.chip {
    font-size: 0.72rem; font-weight: 700;
    padding: 0.30rem 0.7rem; border-radius: 999px;
    border: 1px solid var(--cc-border);
    color: var(--cc-muted);
    background: var(--cc-surface);
    white-space: nowrap;
}

/* ─────────── KPI Cards ─────────── */
.kpi-card {
    border: 1px solid var(--cc-border);
    border-top: 3px solid var(--cc-primary);
    border-radius: var(--cc-radius);
    background: var(--cc-surface);
    box-shadow: var(--cc-shadow);
    padding: 1.05rem 1.15rem 0.95rem;
    min-height: 152px;
    box-sizing: border-box;
    animation: cc-fade-up 0.45s ease both;
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label {
    display: block;
    font-size: 0.73rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--cc-muted); line-height: 1.35;
}
.kpi-value {
    font-size: 1.72rem; font-weight: 800; letter-spacing: -0.02em;
    margin: 0.7rem 0 0.35rem; line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.kpi-delta {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.74rem; font-weight: 700;
    padding: 0.2rem 0.6rem; border-radius: 999px;
}
.kpi-delta.up { color: var(--cc-green); background: var(--cc-green-soft); }
.kpi-delta.flat { color: var(--cc-muted); background: var(--cc-surface); }
.kpi-caption { font-size: 0.75rem; color: var(--cc-muted); margin-top: 0.45rem; line-height: 1.4; }

@keyframes cc-fade-up {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: none; }
}

/* ─────────── Card Surfaces ─────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--cc-border) !important;
    border-radius: var(--cc-radius) !important;
    background: var(--cc-surface);
    box-shadow: var(--cc-shadow);
    padding: 1.15rem 1.3rem 0.75rem !important;
}
.card-header { margin-bottom: 0.55rem; }
.card-title { font-size: 1.02rem; font-weight: 800; margin: 0; letter-spacing: -0.01em; }
.card-subtitle { font-size: 0.8rem; color: var(--cc-muted); margin: 0.15rem 0 0; line-height: 1.4; }
.filter-hint { font-size: 0.8rem; color: var(--cc-muted); line-height: 1.5; margin-top: 2rem; }

/* ─────────── Insight Panel ─────────── */
.insight-panel {
    border: 1px solid var(--cc-border);
    border-left: 4px solid var(--cc-primary);
    border-radius: 14px;
    background: var(--cc-primary-soft);
    padding: 1.15rem 1.45rem;
    margin-top: 1.35rem;
    box-shadow: var(--cc-shadow);
}
.insight-panel h4 {
    margin: 0 0 0.65rem;
    font-size: 0.82rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--cc-primary-deep);
}
.insight-panel ul { margin: 0; padding-left: 1.15rem; }
.insight-panel li { margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.55; }
.insight-panel code {
    background: var(--cc-surface);
    padding: 0.1rem 0.38rem; border-radius: 6px; font-size: 0.84em;
}

/* ─────────── Optimal Portfolio Hero ─────────── */
.opt-hero {
    border-radius: var(--cc-radius);
    padding: 1.45rem 1.6rem 1.3rem;
    background:
        radial-gradient(900px 320px at 92% -60%, rgba(16, 185, 129, 0.35), transparent 60%),
        linear-gradient(118deg, #06342c 0%, #0b5d51 45%, #0e7f6f 100%);
    color: #ecfdf5;
    box-shadow: 0 14px 36px rgba(6, 52, 44, 0.45);
    margin-bottom: 1.35rem;
    animation: cc-fade-up 0.45s ease both;
}
.opt-kicker {
    font-size: 0.7rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.16em; opacity: 0.82;
}
.opt-hero h3 {
    margin: 0.28rem 0 0.15rem; font-size: 1.45rem;
    font-weight: 800; letter-spacing: -0.02em; color: #ffffff;
}
.opt-chips { margin: 0.35rem 0 0.15rem; }
.opt-chip {
    display: inline-block;
    font-size: 0.74rem; font-weight: 700;
    padding: 0.28rem 0.75rem; border-radius: 999px;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.22);
    margin: 0 0.35rem 0.35rem 0;
}
.opt-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.85rem; margin-top: 0.9rem;
}
.opt-stat {
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 12px; padding: 0.72rem 0.9rem;
}
.opt-stat .s-label {
    font-size: 0.65rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.09em; opacity: 0.78;
}
.opt-stat .s-value {
    font-size: 1.1rem; font-weight: 800; margin-top: 0.2rem;
    font-variant-numeric: tabular-nums;
}

/* ─────────── Sidebar ─────────── */
.sb-brand { display: flex; align-items: center; gap: 0.8rem; padding: 0.25rem 0.15rem 0.85rem; }
.sb-logo {
    width: 46px; height: 46px; flex: 0 0 46px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.92rem; font-weight: 800; color: #ffffff;
    letter-spacing: 0.03em; border-radius: 13px;
    background: linear-gradient(135deg, #0f766e, #10b981);
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35);
}
.sb-name { font-size: 1.02rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.2; }
.sb-tag {
    font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; color: var(--cc-muted);
}
.sb-status {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.72rem; font-weight: 700;
    color: var(--cc-green); background: var(--cc-green-soft);
    border-radius: 999px; padding: 0.28rem 0.7rem; margin-bottom: 0.4rem;
}
.sb-status .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cc-green);
    animation: cc-pulse 2s infinite;
}
@keyframes cc-pulse {
    0%, 100% { box-shadow: 0 0 0 0 var(--cc-green-soft); }
    50%      { box-shadow: 0 0 0 5px transparent; }
}
.sb-section {
    font-size: 0.67rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.13em;
    color: var(--cc-muted); margin: 1.05rem 0 0.45rem;
}
.meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.meta-item {
    border: 1px solid var(--cc-border); border-radius: 11px;
    padding: 0.5rem 0.65rem; background: var(--cc-surface);
}
.meta-item span {
    display: block; font-size: 0.62rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.07em; color: var(--cc-muted);
}
.meta-item b { font-size: 0.88rem; font-weight: 700; }
.sb-note {
    border: 1px solid var(--cc-border); border-radius: 12px;
    background: var(--cc-surface);
    padding: 0.85rem 0.95rem; font-size: 0.78rem; line-height: 1.5;
}
.sb-note p { margin: 0 0 0.5rem; }
.sb-note p:last-child { margin-bottom: 0; }

/* Sidebar navigation pills */
[data-testid="stSidebar"] div[role="radiogroup"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.5rem 0.7rem;
    border-radius: 10px;
    border: 1px solid transparent;
    font-weight: 600; font-size: 0.9rem;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: var(--cc-primary-soft);
}
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
    background: var(--cc-primary-soft);
    border-color: var(--cc-primary);
}

/* ─────────── Tables & Footer ─────────── */
[data-testid="stDataFrame"] { border-radius: 12px; }
.app-footer {
    margin-top: 2.25rem; padding-top: 1.2rem;
    border-top: 1px solid var(--cc-border);
    text-align: center; font-size: 0.76rem;
    color: var(--cc-muted); line-height: 1.55;
}
.app-footer .foot-chips {
    display: flex; justify-content: center; gap: 0.45rem;
    flex-wrap: wrap; margin-bottom: 0.6rem;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════
# Formatting Helpers
# ═══════════════════════════════════════════════════════════════════════


def fmt_lakhs(value_inr: float, decimals: int = 2) -> str:
    """Format an INR amount in Lakhs."""
    return f"₹{value_inr / 100000.0:,.{decimals}f} Lakhs"


def kg_to_mt(value_kg: float, decimals: int = 1) -> str:
    """Format a kilogram CO2e quantity in metric tonnes."""
    return f"{value_kg / 1000.0:,.{decimals}f} MTCO₂e"


# ═══════════════════════════════════════════════════════════════════════
# UI Primitives
# ═══════════════════════════════════════════════════════════════════════


def setup_page() -> None:
    """Configure page metadata and inject the dashboard design system."""
    st.set_page_config(
        page_title="Campus Carbon Analytics | Executive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(badge: str, title: str, subtitle: str, chips: list[str] | None = None) -> None:
    """Render the hero band at the top of each page (badge = page number)."""
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f"""
        <div class="page-header">
          <div class="ph-badge">{badge}</div>
          <div class="ph-body">
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div class="ph-chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card():
    """Bordered card surface (graceful fallback for Streamlit < 1.29)."""
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()
    with box:
        yield


def card_header(title: str, subtitle: str | None = None) -> None:
    """Consistent title/subtitle block atop a card."""
    subtitle_html = f'<p class="card-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="card-header"><p class="card-title">{title}</p>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    caption: str = "",
    delta: str | None = None,
    delta_kind: str = "up",
    help_text: str | None = None,
) -> None:
    """Custom KPI metric card with accent bar, value, delta chip, and caption."""
    delta_html = f'<div class="kpi-delta {delta_kind}">{delta}</div>' if delta else ""
    title_attr = f' title="{help_text}"' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card"{title_attr}>
          <span class="kpi-label">{label}</span>
          <div class="kpi-value">{value}</div>
          {delta_html}
          <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_panel(heading: str, body_html: str) -> None:
    """Highlighted executive callout panel."""
    st.markdown(
        f'<div class="insight-panel"><h4>{heading}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )


def render_chart(fig) -> None:
    """Render a pipeline chart with dashboard-wide polish (blended background)."""
    try:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    except Exception:  # pragma: no cover — defensive, never block rendering
        pass
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "displaylogo": False},
    )


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════


def render_sidebar(data: dict) -> str:
    """Render sidebar brand, navigation, institutional scope, and notices."""
    summary = data.get("summary", {})
    cfg = summary.get("config", {})
    opt = summary.get("optimal_portfolio", {})
    budget_lakhs = float(cfg.get("optimization_budget_inr", 2500000.0)) / 100000.0

    with st.sidebar:
        # ── Brand ──
        st.markdown(
            """
            <div class="sb-brand">
              <div class="sb-logo">CCA</div>
              <div>
                <div class="sb-name">Campus Carbon Analytics</div>
                <div class="sb-tag">Decarbonization DSS</div>
              </div>
            </div>
            <div class="sb-status"><span class="dot"></span> Phase 16 Artifacts Loaded</div>
            """,
            unsafe_allow_html=True,
        )

        # ── Navigation ──
        st.markdown('<p class="sb-section">Navigation</p>', unsafe_allow_html=True)
        page = st.radio(
            "Dashboard pages",
            options=NAV_TITLES,
            label_visibility="collapsed",
        )

        # ── Institutional Scope ──
        st.markdown('<p class="sb-section">Institutional Scope</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="meta-grid">
              <div class="meta-item"><span>Baseline Year</span><b>{cfg.get('baseline_year', 2025)}</b></div>
              <div class="meta-item"><span>Timeline</span><b>2023–2025 · 36 mo</b></div>
              <div class="meta-item"><span>Facilities</span><b>5 Campus Complexes</b></div>
              <div class="meta-item"><span>Active Budget</span><b>₹{budget_lakhs:,.1f} Lakhs</b></div>
              <div class="meta-item" style="grid-column: span 2;">
                <span>Recommended Portfolio</span><b>{opt.get('portfolio_id', 'P-10110')}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Decision Support Notice ──
        st.markdown('<p class="sb-section">Decision Support Notice</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sb-note">
              <p><b>Synthetic Activity</b> — Activity data is synthetically generated for scenario modeling and simulation.</p>
              <p><b>Documented Factors</b> — Emission factors are sourced from statutory standards (CEA CO₂ Database and India GHG Program).</p>
              <p><b>Decision Support</b> — Forecasts and optimization models provide exploratory decision-support, not prescriptive mandates.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Data Source: Precomputed Phase 16 Integrated Pipeline Artifacts (Read-Only)")

    return page


# ═══════════════════════════════════════════════════════════════════════
# Page 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════════


def render_overview_page(data: dict) -> None:
    """Executive Overview: headline KPIs and the condensed 48-month horizon."""
    page_header(
        badge="01",
        title="Executive Overview & Decarbonization Roadmap",
        subtitle="High-level institutional carbon performance, emissions intensity, and optimal decarbonization strategy.",
        chips=["36-Month Actuals", "48-Month Horizon", "Read-Only"],
    )

    summary = data["summary"]
    acct = summary.get("carbon_accounting", {})
    opt = summary.get("optimal_portfolio", {})

    tot_mt = float(acct.get("total_emissions_mt", 0.0))
    per_student = float(acct.get("emissions_per_student_kg", 0.0))
    pct_red = float(opt.get("percentage_reduction", 0.0))
    kg_red = float(opt.get("total_absolute_reduction_kg", 0.0))
    cost_inr = float(opt.get("total_cost_inr", 0.0))
    rem_inr = float(opt.get("remaining_budget_inr", 0.0))

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            label="Total Historical Emissions",
            value=f"{tot_mt:,.2f} MTCO₂e",
            caption="36-month cumulative total across all facilities",
            help_text="Total measured greenhouse gas emissions across all campus facilities (2023–2025).",
        )
    with c2:
        kpi_card(
            label="Emissions Intensity",
            value=f"{per_student:,.2f} kgCO₂e",
            caption="Per student-month intensity",
            help_text="Emissions per student-month across the institution.",
        )
    with c3:
        kpi_card(
            label="Optimal Carbon Abatement",
            value=f"{pct_red:.2f}%",
            caption="Relative to 2025 forward baseline",
            delta=f"{kg_to_mt(kg_red)}/yr avoided",
            delta_kind="up",
            help_text="Projected greenhouse gas reduction under the optimal budget-constrained portfolio.",
        )
    with c4:
        kpi_card(
            label="Required Capital Outlay",
            value=fmt_lakhs(cost_inr),
            caption="Allocated from active budget cap",
            delta=f"₹{rem_inr / 100000.0:,.2f}L unspent surplus",
            delta_kind="flat",
            help_text="Total capital expenditure required to implement the optimal portfolio.",
        )

    # ── Main Visuals ──
    v1, v2 = st.columns([7, 5])
    with v1:
        with card():
            card_header(
                "48-Month Emissions Horizon",
                "Historical actuals (2023–2025) and optimized forward trajectory (2026–2027)",
            )
            render_chart(
                create_48m_timeline_chart(
                    historical_df=data["historical_monthly"],
                    forecast_df=data["forecast_projections"],
                    target="total_emissions_kg",
                )
            )
    with v2:
        with card():
            card_header("Emissions Mix by Category", "Cumulative footprint share by scope category")
            render_chart(create_category_donut_chart(acct.get("category_contributions", {})))

    # ── Executive Takeaways ──
    insight_panel(
        "Executive Summary & Key Findings",
        """
        <ul>
          <li><b>Dominant Scope 2 Footprint:</b> Grid electricity accounts for <b>79.77%</b> of campus emissions, establishing energy efficiency and rooftop solar as the highest-leverage decarbonization avenues.</li>
          <li><b>Budget Optimization Finding:</b> Under the ₹25 Lakh institutional grant, the optimal portfolio is <b>P-10110</b> (LED Retrofit + AC Summer Cycling + Waste Composting). It achieves <b>15.80% carbon reduction</b> (225.8 MTCO₂e) for only <b>₹9.5 Lakhs</b>, conserving ₹15.5 Lakhs in surplus capital.</li>
          <li><b>Forecasting Robustness:</b> The primary Holt-Winters model captures annual growth (2.5%) and operational seasonality with an empirical test MAPE of <b>1.27%</b> and 100% prediction interval coverage.</li>
        </ul>
        """,
    )


# ═══════════════════════════════════════════════════════════════════════
# Page 2 — Historical Accounting
# ═══════════════════════════════════════════════════════════════════════


def render_accounting_page(data: dict) -> None:
    """Carbon Accounting & Historical Emissions Breakdown."""
    page_header(
        badge="02",
        title="Campus Carbon Accounting & Historical Breakdown",
        subtitle="Rigorous Scope 1, 2, and 3 accounting adhering to CEA, India GHG, and DEFRA emission factors.",
        chips=["Scope 1 · 2 · 3", "CEA · India GHG · DEFRA", "2023–2025"],
    )

    campus_df = data["campus_emissions"]

    # ── Monthly Composition ──
    with card():
        card_header(
            "Monthly Emissions Composition",
            "Stacked scope-category time series across the 36-month historical window (kg CO₂e)",
        )
        render_chart(create_stacked_monthly_bar_chart(data["historical_monthly"]))

    # ── Facility Breakdown ──
    c1, c2 = st.columns(2)
    with c1:
        with card():
            card_header("Emissions by Campus Facility", "Cumulative footprint ranked by building complex")
            render_chart(create_building_emissions_bar_chart(campus_df))
    with c2:
        with card():
            card_header("Facility-Level Cumulative Metrics", "Headcount, footprint, and per-student intensity by facility")
            bldg_grp = campus_df.groupby("building").agg(
                total_emissions_mt=("total_emissions_mt", "sum"),
                avg_students=("student_count", "mean"),
                avg_intensity_kg=("emissions_per_student_kg", "mean"),
            ).reset_index()
            bldg_grp["total_emissions_mt"] = bldg_grp["total_emissions_mt"].round(2)
            bldg_grp["avg_intensity_kg"] = bldg_grp["avg_intensity_kg"].round(2)
            bldg_grp["avg_students"] = bldg_grp["avg_students"].astype(int)
            bldg_grp.columns = [
                "Campus Facility",
                "Cumulative MTCO₂e",
                "Mean Student Headcount",
                "Intensity (kg/student)",
            ]
            st.dataframe(
                bldg_grp.sort_values("Cumulative MTCO₂e", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Cumulative MTCO₂e": st.column_config.NumberColumn("Cumulative MTCO₂e", format="%.2f"),
                    "Mean Student Headcount": st.column_config.NumberColumn("Mean Student Headcount", format="%d"),
                    "Intensity (kg/student)": st.column_config.NumberColumn("Intensity (kg/student)", format="%.2f"),
                },
            )

    # ── Top Drivers ──
    with card():
        card_header("Top 10 Monthly Facility Emissions Drivers", "Highest-emitting facility-month records across the accounting period")
        top_drivers = (
            campus_df[
                [
                    "date", "building", "electricity_emissions_kg",
                    "travel_emissions_kg", "waste_emissions_kg",
                    "total_emissions_kg", "total_emissions_mt",
                ]
            ]
            .sort_values("total_emissions_kg", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_drivers["date"] = pd.to_datetime(top_drivers["date"])
        top_drivers.columns = [
            "Month", "Facility", "Electricity (kg)", "Travel (kg)",
            "Waste (kg)", "Total (kgCO₂e)", "Total (MTCO₂e)",
        ]
        st.dataframe(
            top_drivers,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": st.column_config.DateColumn("Month", format="MMM YYYY"),
                "Electricity (kg)": st.column_config.NumberColumn("Electricity (kg)", format="%.1f"),
                "Travel (kg)": st.column_config.NumberColumn("Travel (kg)", format="%.1f"),
                "Waste (kg)": st.column_config.NumberColumn("Waste (kg)", format="%.1f"),
                "Total (kgCO₂e)": st.column_config.NumberColumn("Total (kgCO₂e)", format="%.1f"),
                "Total (MTCO₂e)": st.column_config.NumberColumn("Total (MTCO₂e)", format="%.2f"),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Page 3 — Forecasting & Uncertainty
# ═══════════════════════════════════════════════════════════════════════


def render_forecasting_page(data: dict) -> None:
    """Primary Forecasting Model & Uncertainty Analysis."""
    page_header(
        badge="03",
        title="Time Series Forecasting & Uncertainty Quantification",
        subtitle="Exponential smoothing (Holt-Winters) forward projections with state-space simulated prediction intervals.",
        chips=["Holt-Winters Add/Add", "95% Prediction Intervals", "Monte Carlo UQ"],
    )

    # ── Target Selection Control Bar ──
    with card():
        sel_col, hint_col = st.columns([5, 7])
        with sel_col:
            selected_label = st.selectbox(
                "Select emissions time series to inspect",
                options=list(TARGET_OPTIONS.keys()),
                index=0,
            )
        with hint_col:
            st.markdown(
                '<div class="filter-hint">Switch between institution-wide and category-level '
                "projections — all performance metrics, charts, and tables update accordingly.</div>",
                unsafe_allow_html=True,
            )
    target_col = TARGET_OPTIONS[selected_label]

    # ── Model Performance Metrics ──
    metrics_df = data.get("forecast_metrics")
    if metrics_df is not None and not metrics_df.empty:
        target_metrics = metrics_df[
            (metrics_df["target"] == target_col) & (metrics_df["model"].str.contains("Holt-Winters"))
        ]
        if not target_metrics.empty:
            m_row = target_metrics.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                kpi_card("Primary Model", "Holt-Winters", caption="Additive trend + additive seasonality")
            with m2:
                kpi_card("MAE", f"{float(m_row['mae_kg']):,.1f} kgCO₂e", caption="Mean absolute error — test hold-out")
            with m3:
                kpi_card("RMSE", f"{float(m_row['rmse_kg']):,.1f} kgCO₂e", caption="Root mean squared error — test hold-out")
            with m4:
                kpi_card(
                    "MAPE", f"{float(m_row['mape_pct']):.2f}%",
                    caption="Mean absolute % error — test hold-out",
                    delta="Beats naive baseline", delta_kind="up",
                )

    # ── Forecast Trajectory ──
    with card():
        card_header(
            "Forecast Trajectory with 95% Prediction Intervals",
            f"Point forecasts for {selected_label.lower()} with state-space simulated bounds",
        )
        render_chart(
            create_forecast_trajectory_chart(
                historical_df=data["historical_monthly"],
                forecast_df=data["forecast_projections"],
                target=target_col,
            )
        )

    # ── Variance Decomposition ──
    with card():
        card_header(
            "One-at-a-Time (OAT) Monte Carlo Variance Decomposition",
            "Isolated contribution of forecast state-space error, activity measurement error, "
            "and emission factor conversion error to total forecast uncertainty",
        )
        render_chart(
            create_variance_decomposition_chart(
                uncertainty_decomp_df=data["uncertainty_decomp"],
                target=target_col,
            )
        )

    # ── Forecast Schedule Table ──
    with card():
        card_header("2026 Forecast Schedule with 95% Prediction Bounds", "Monthly point forecasts and simulated prediction interval spread")
        fc_sub = data["forecast_projections"][data["forecast_projections"]["target"] == target_col].copy()
        fc_sub["uncertainty_spread_kg"] = fc_sub["upper_bound_95"] - fc_sub["lower_bound_95"]
        fc_sub = fc_sub[
            ["date", "point_forecast", "lower_bound_95", "upper_bound_95", "uncertainty_spread_kg"]
        ].reset_index(drop=True)
        fc_sub["date"] = pd.to_datetime(fc_sub["date"])
        fc_sub.columns = [
            "Forecast Date", "Point Forecast (kgCO₂e)",
            "Lower Bound 95% (kgCO₂e)", "Upper Bound 95% (kgCO₂e)",
            "Uncertainty Spread (kg)",
        ]
        st.dataframe(
            fc_sub,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Forecast Date": st.column_config.DateColumn("Forecast Date", format="MMM YYYY"),
                "Point Forecast (kgCO₂e)": st.column_config.NumberColumn("Point Forecast (kgCO₂e)", format="%.1f"),
                "Lower Bound 95% (kgCO₂e)": st.column_config.NumberColumn("Lower Bound 95% (kgCO₂e)", format="%.1f"),
                "Upper Bound 95% (kgCO₂e)": st.column_config.NumberColumn("Upper Bound 95% (kgCO₂e)", format="%.1f"),
                "Uncertainty Spread (kg)": st.column_config.NumberColumn("Uncertainty Spread (kg)", format="%.1f"),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Page 4 — Decarbonization Scenarios
# ═══════════════════════════════════════════════════════════════════════


def render_scenarios_page(data: dict) -> None:
    """Standalone Decarbonization Scenarios."""
    page_header(
        badge="04",
        title="Standalone Decarbonization Scenarios",
        subtitle="Isolated one-at-a-time (OAT) marginal emissions reduction and cost efficiency of candidate interventions.",
        chips=["5 Interventions", "OAT Isolation", "MAC Ranked"],
    )

    scen_df = data["scenario_evaluations"]

    # ── Standalone Reduction Chart ──
    with card():
        card_header(
            "Standalone Abatement by Intervention",
            "Marginal annual emissions reduction achieved by each intervention in isolation",
        )
        render_chart(create_scenario_reduction_chart(scen_df))

    # ── Intervention Matrix Table ──
    with card():
        card_header(
            "Intervention Financial & Abatement Matrix",
            "Implementation cost, avoided emissions, and marginal abatement cost (MAC) per initiative",
        )
        display_df = scen_df.copy()
        display_df["implementation_cost_lakhs"] = (display_df["implementation_cost_inr"] / 100000.0).round(2)
        display_df["absolute_reduction_mt"] = (display_df["absolute_reduction_kg"] / 1000.0).round(2)
        display_df["baseline_emissions_mt"] = (display_df["baseline_emissions_kg"] / 1000.0).round(2)
        display_df["scenario_emissions_mt"] = (display_df["scenario_emissions_kg"] / 1000.0).round(2)

        cols = [
            "intervention_id", "intervention_name", "affected_category",
            "implementation_cost_lakhs", "baseline_emissions_mt",
            "scenario_emissions_mt", "absolute_reduction_mt",
            "percentage_reduction", "cost_per_kg_reduced",
        ]
        renamed = display_df[cols].copy()
        renamed.columns = [
            "ID", "Intervention Initiative", "Target Category",
            "Cost (₹ Lakhs)", "Baseline (MT)", "Scenario (MT)",
            "Avoided (MT)", "Reduction (%)", "MAC (₹/kgCO₂e)",
        ]
        st.dataframe(
            renamed.sort_values("Avoided (MT)", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cost (₹ Lakhs)": st.column_config.NumberColumn("Cost (₹ Lakhs)", format="₹%.2f"),
                "Baseline (MT)": st.column_config.NumberColumn("Baseline (MT)", format="%.2f"),
                "Scenario (MT)": st.column_config.NumberColumn("Scenario (MT)", format="%.2f"),
                "Avoided (MT)": st.column_config.NumberColumn("Avoided (MT)", format="%.2f"),
                "Reduction (%)": st.column_config.ProgressColumn(
                    "Reduction (%)", min_value=0.0, max_value=10.0, format="%.2f%%"
                ),
                "MAC (₹/kgCO₂e)": st.column_config.NumberColumn("MAC (₹/kgCO₂e)", format="₹%.2f"),
            },
        )

    # ── MAC Takeaways ──
    insight_panel(
        "Marginal Abatement Cost (MAC) Key Takeaways",
        """
        <ul>
          <li><b>Most Capital Efficient:</b> <code>INT-001</code> (LED Lighting Retrofit) achieves the lowest MAC at <b>₹3.65 / kgCO₂e</b>, followed closely by <code>INT-003</code> (AC Optimization) at <b>₹4.16 / kgCO₂e</b>.</li>
          <li><b>Highest Total Abatement:</b> LED Retrofit provides the largest physical reduction (136.9 MTCO₂e / 9.58% of campus total).</li>
          <li><b>Capital-Intensive Infrastructure:</b> Rooftop Solar (₹58.19/kg) and EV Transit (₹44.30/kg) require significantly greater capital expenditure per kilogram of carbon abated.</li>
        </ul>
        """,
    )


# ═══════════════════════════════════════════════════════════════════════
# Page 5 — Portfolio Optimization
# ═══════════════════════════════════════════════════════════════════════


def render_optimization_page(data: dict) -> None:
    """Constrained Portfolio Budget Optimization & Efficient Frontier."""
    page_header(
        badge="05",
        title="Constrained Portfolio Budget Optimization",
        subtitle="Exhaustive powerset evaluation (32 combinations) resolving combinatorial interactions without double-counting.",
        chips=["2⁵ = 32 Portfolios", "Exact Search", "Budget-Constrained"],
    )

    summary = data["summary"]
    cfg = summary.get("config", {})
    opt = summary.get("optimal_portfolio", {})
    budget_inr = float(cfg.get("optimization_budget_inr", 2500000.0))
    p32_df = data["portfolio_all_32"]

    # ── Optimal Portfolio Hero Banner ──
    opt_id = opt.get("portfolio_id", "P-10110")
    cost_val = float(opt.get("total_cost_inr", 0.0))
    red_val = float(opt.get("total_absolute_reduction_kg", 0.0))
    pct_val = float(opt.get("percentage_reduction", 0.0))
    rem_val = float(opt.get("remaining_budget_inr", 0.0))
    mac_val = float(opt.get("cost_per_kg_reduced", 0.0))
    sel_invs = ", ".join(opt.get("selected_interventions", []))

    inv_chips = "".join(
        f'<span class="opt-chip">{name.strip()}</span>' for name in sel_invs.split(",")
    ) or '<span class="opt-chip">—</span>'

    st.markdown(
        f"""
        <div class="opt-hero">
          <div class="opt-kicker">Selected Optimal Portfolio</div>
          <h3>{opt_id}</h3>
          <div class="opt-chips">{inv_chips}</div>
          <div class="opt-stats">
            <div class="opt-stat">
              <div class="s-label">Capital Expenditure</div>
              <div class="s-value">{fmt_lakhs(cost_val)}</div>
            </div>
            <div class="opt-stat">
              <div class="s-label">Unspent Surplus</div>
              <div class="s-value">{fmt_lakhs(rem_val)}</div>
            </div>
            <div class="opt-stat">
              <div class="s-label">Annual Avoided Carbon</div>
              <div class="s-value">{kg_to_mt(red_val, 2)} · {pct_val:.2f}%</div>
            </div>
            <div class="opt-stat">
              <div class="s-label">Marginal Abatement Cost</div>
              <div class="s-value">₹{mac_val:.2f} / kgCO₂e</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Efficient Frontier ──
    with card():
        card_header(
            "Efficient Frontier — Cost vs. Abatement",
            "Feasible and over-budget portfolios across the budget-constrained solution space",
        )
        render_chart(
            create_efficient_frontier_chart(
                portfolio_df=p32_df,
                optimal_portfolio_id=opt_id,
                budget_inr=budget_inr,
            )
        )

    # ── Portfolio Explorer Table ──
    with card():
        card_header("Combinatorial Powerset Solution Space", "All 32 candidate portfolios ranked by annual avoided emissions")
        filter_mode = st.radio(
            "Display portfolios",
            options=["Feasible Only (Within Budget)", "All 32 Portfolios (Including Over-Budget)"],
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
            "portfolio_id", "selected_interventions", "num_interventions",
            "cost_lakhs", "reduction_mt", "percentage_reduction",
            "cost_per_kg_reduced", "is_feasible",
        ]
        renamed_p = view_df[table_cols].copy()
        renamed_p.columns = [
            "Portfolio ID", "Selected Initiatives", "Project Count",
            "Cost (₹ Lakhs)", "Avoided (MT)", "Reduction (%)",
            "MAC (₹/kg)", "Feasible?",
        ]
        st.dataframe(
            renamed_p.sort_values("Avoided (MT)", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=480,
            column_config={
                "Project Count": st.column_config.NumberColumn("Project Count", format="%d"),
                "Cost (₹ Lakhs)": st.column_config.NumberColumn("Cost (₹ Lakhs)", format="₹%.2f"),
                "Avoided (MT)": st.column_config.NumberColumn("Avoided (MT)", format="%.2f"),
                "Reduction (%)": st.column_config.ProgressColumn(
                    "Reduction (%)", min_value=0.0, max_value=20.0, format="%.2f%%"
                ),
                "MAC (₹/kg)": st.column_config.NumberColumn("MAC (₹/kg)", format="₹%.2f"),
                "Feasible?": st.column_config.CheckboxColumn("Feasible", default=False),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════


def render_footer() -> None:
    """Global dashboard footer and disclaimer."""
    st.markdown(
        """
        <div class="app-footer">
          <div class="foot-chips">
            <span class="chip">Phase 16 Pipeline</span>
            <span class="chip">Read-Only Artifacts</span>
            <span class="chip">CEA · India GHG · DEFRA Factors</span>
          </div>
          <div><b>Campus Carbon Analytics System</b></div>
          <div><em>Disclaimer:</em> Activity data and intervention scenarios are simulated decision-support models.
          All analytical outputs are intended solely for institutional planning and decision-support exploration.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main dashboard entry point."""
    setup_page()

    # Load Phase 16 precomputed pipeline artifacts
    try:
        data = load_dashboard_data(DEFAULT_RUN_DIR)
    except (DashboardDataError, FileNotFoundError) as exc:
        st.error(
            "**Pipeline Artifacts Not Found or Incomplete**\n\n"
            "The executive dashboard is a read-only presentation interface that consumes precomputed "
            "Phase 16 pipeline outputs. Please execute the integrated pipeline before launching the dashboard:\n\n"
            "```bash\npython -m src.pipeline\n```"
        )
        with st.expander("Technical Diagnostic Details"):
            st.code(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"**Unexpected Error Ingesting Dashboard Data:** {exc}")
        st.stop()

    # Sidebar navigation
    page = render_sidebar(data)

    # Page router
    routes = {
        "Executive Overview": render_overview_page,
        "Historical Accounting": render_accounting_page,
        "Forecasting & Uncertainty": render_forecasting_page,
        "Decarbonization Scenarios": render_scenarios_page,
        "Portfolio Optimization": render_optimization_page,
    }
    routes[page](data)

    render_footer()


if __name__ == "__main__":
    main()