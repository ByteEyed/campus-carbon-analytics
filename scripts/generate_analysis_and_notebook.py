"""
Exploratory Carbon Analysis and Notebook Builder
================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

This script conducts the comprehensive exploratory analysis of the campus activity
and carbon emissions dataset, generates publication-quality visualization figures,
and creates the standalone exploratory analysis Jupyter Notebook:
`notebooks/exploratory_analysis.ipynb`.

METHODOLOGICAL & EPISTEMIC PRINCIPLES:
---------------------------------------
- Strictly non-causal: Empirical associations and patterns are described without
  claiming unobserved causal mechanisms.
- Clear separation: Observed data values are explicitly distinguished from
  underlying parameterized simulation assumptions and external emission factors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Set clean publication plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

CATEGORY_COLORS = {
    "Electricity": "#1f77b4",   # Primary blue
    "Travel": "#ff7f0e",        # Energetic orange
    "Waste": "#2ca02c",         # Environmental green
    "Procurement": "#9467bd",   # Purple
}

BUILDING_COLORS = [
    "#2b5c8f",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#1b9e77",
]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned activity data, computed emissions, and emission factors."""
    activity_path = Path("data/processed/cleaned_activity.csv")
    emissions_path = Path("data/processed/campus_emissions.csv")
    factors_path = Path("data/emission_factors.csv")

    act_df = pd.read_csv(activity_path)
    em_df = pd.read_csv(emissions_path)
    ef_df = pd.read_csv(factors_path)

    em_df["date"] = pd.to_datetime(em_df["date"])
    em_df["year"] = em_df["date"].dt.year
    em_df["month"] = em_df["date"].dt.month
    em_df["year_month"] = em_df["date"].dt.strftime("%Y-%m")

    return act_df, em_df, ef_df


def generate_figures(em_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Generate and save publication-quality simple figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    # -------------------------------------------------------------------------
    # Figure 1: Monthly Campus Emissions Trajectory & Category Stack
    # -------------------------------------------------------------------------
    monthly_cat = em_df.groupby("year_month")[
        ["electricity_emissions_kg", "travel_emissions_kg", "waste_emissions_kg", "procurement_emissions_kg"]
    ].sum() / 1000.0  # Convert to MTCO2e

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(monthly_cat))
    dates_str = monthly_cat.index.tolist()

    # Stacked bar chart
    bottom = np.zeros(len(monthly_cat))
    cat_keys = [
        ("Electricity", "electricity_emissions_kg"),
        ("Travel", "travel_emissions_kg"),
        ("Waste", "waste_emissions_kg"),
        ("Procurement", "procurement_emissions_kg"),
    ]
    for label, col in cat_keys:
        vals = monthly_cat[col].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=label,
            color=CATEGORY_COLORS[label],
            alpha=0.88,
            width=0.75,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    # Add total line overlay
    ax.plot(x, bottom, color="#111111", linewidth=1.5, marker="o", markersize=3, label="Total Emissions")

    # Annotate academic calendar breaks
    for i, ym in enumerate(dates_str):
        if ym.endswith("-06"):  # Summer break
            ax.axvspan(i - 0.4, i + 0.4, color="#e0e0e0", alpha=0.5, zorder=0)
            ax.text(i, bottom[i] + 3.5, "Summer\nBreak", ha="center", va="bottom", fontsize=7.5, color="#555555")
        elif ym.endswith("-12"):  # Winter break
            ax.axvspan(i - 0.4, i + 0.4, color="#e0e0e0", alpha=0.5, zorder=0)
            ax.text(i, bottom[i] + 3.5, "Winter\nBreak", ha="center", va="bottom", fontsize=7.5, color="#555555")

    # Format ticks: show quarterly labels
    quarterly_indices = [i for i, d in enumerate(dates_str) if d.endswith(("-01", "-04", "-07", "-10"))]
    ax.set_xticks(quarterly_indices)
    ax.set_xticklabels([dates_str[i] for i in quarterly_indices], rotation=45, ha="right")
    ax.set_ylabel("Emissions ($tCO_2e$ / Metric Tonnes)")
    ax.set_title("Figure 1: Monthly Campus Carbon Emissions Trajectory (2023 - 2025)", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, ncol=3)
    ax.set_ylim(0, max(bottom) * 1.22)

    fig.tight_layout()
    p1 = output_dir / "01_monthly_emissions_trend.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_paths["monthly_trend"] = p1

    # -------------------------------------------------------------------------
    # Figure 2: Yearly Emissions Comparison by Category
    # -------------------------------------------------------------------------
    yearly_cat = em_df.groupby("year")[
        ["electricity_emissions_kg", "travel_emissions_kg", "waste_emissions_kg", "procurement_emissions_kg"]
    ].sum() / 1000.0

    fig, ax = plt.subplots(figsize=(8, 4.8))
    years = yearly_cat.index.tolist()
    y_pos = np.arange(len(years))
    bar_width = 0.55

    bottom = np.zeros(len(years))
    for label, col in cat_keys:
        vals = yearly_cat[col].values
        bars = ax.bar(
            y_pos,
            vals,
            bottom=bottom,
            label=label,
            color=CATEGORY_COLORS[label],
            alpha=0.88,
            width=bar_width,
            edgecolor="white",
        )
        bottom += vals

    # Add total label on top of each bar
    for idx, tot in enumerate(bottom):
        ax.text(idx, tot + 15, f"{tot:,.1f} t\n({tot * 1000:,.0f} kg)", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.set_xticks(y_pos)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Annual Emissions ($tCO_2e$ / Metric Tonnes)")
    ax.set_title("Figure 2: Annual Campus Carbon Emissions by Category (2023 - 2025)", pad=12)
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    p2 = output_dir / "02_yearly_emissions_comparison.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_paths["yearly_comparison"] = p2

    # -------------------------------------------------------------------------
    # Figure 3: Category Contribution Share
    # -------------------------------------------------------------------------
    tot_elec = em_df["electricity_emissions_kg"].sum() / 1000.0
    tot_travel = em_df["travel_emissions_kg"].sum() / 1000.0
    tot_waste = em_df["waste_emissions_kg"].sum() / 1000.0
    tot_proc = em_df["procurement_emissions_kg"].sum() / 1000.0
    grand_tot = tot_elec + tot_travel + tot_waste + tot_proc

    cat_df = pd.DataFrame([
        {"category": "Electricity", "emissions_t": tot_elec, "pct": tot_elec / grand_tot * 100},
        {"category": "Travel", "emissions_t": tot_travel, "pct": tot_travel / grand_tot * 100},
        {"category": "Waste", "emissions_t": tot_waste, "pct": tot_waste / grand_tot * 100},
        {"category": "Procurement", "emissions_t": tot_proc, "pct": tot_proc / grand_tot * 100},
    ]).sort_values(by="emissions_t", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    y_idx = np.arange(len(cat_df))
    colors = [CATEGORY_COLORS[c] for c in cat_df["category"]]
    bars = ax.barh(y_idx, cat_df["pct"], color=colors, alpha=0.88, height=0.55, edgecolor="none")

    for idx, (_, row) in enumerate(cat_df.iterrows()):
        ax.text(
            row["pct"] + 1.0,
            idx,
            f"{row['pct']:.1f}%  ({row['emissions_t']:,.1f} tCO2e)",
            va="center",
            fontweight="semibold",
            fontsize=9,
        )

    ax.set_yticks(y_idx)
    ax.set_yticklabels(cat_df["category"], fontweight="semibold")
    ax.set_xlabel("Contribution to Total Campus Emissions (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Figure 3: Category Contribution to Total Campus Emissions (3-Year Aggregation)", pad=12)

    fig.tight_layout()
    p3 = output_dir / "03_category_contribution.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_paths["category_contribution"] = p3

    # -------------------------------------------------------------------------
    # Figure 4: Building Facility Emissions Contribution
    # -------------------------------------------------------------------------
    bldg_summary = em_df.groupby("building")[
        ["electricity_emissions_kg", "travel_emissions_kg", "waste_emissions_kg", "procurement_emissions_kg", "total_emissions_kg"]
    ].sum() / 1000.0
    bldg_summary = bldg_summary.sort_values(by="total_emissions_kg", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    b_y = np.arange(len(bldg_summary))
    left = np.zeros(len(bldg_summary))

    for label, col in cat_keys:
        vals = bldg_summary[col].values
        ax.barh(b_y, vals, left=left, label=label, color=CATEGORY_COLORS[label], alpha=0.88, height=0.55)
        left += vals

    for idx, tot in enumerate(left):
        pct = (tot / (grand_tot) * 100)
        ax.text(tot + 15, idx, f"{tot:,.1f} t ({pct:.1f}%)", va="center", fontweight="semibold", fontsize=8.5)

    ax.set_yticks(b_y)
    ax.set_yticklabels(bldg_summary.index, fontweight="semibold")
    ax.set_xlabel("Total Cumulative Emissions ($tCO_2e$ / Metric Tonnes)")
    ax.set_title("Figure 4: Cumulative Emissions Breakdown by Campus Facility (2023 - 2025)", pad=12)
    ax.set_xlim(0, max(left) * 1.18)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    p4 = output_dir / "04_building_contribution.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_paths["building_contribution"] = p4

    # -------------------------------------------------------------------------
    # Figure 5: Activity Time Series & Seasonality Multi-panel
    # -------------------------------------------------------------------------
    monthly_activity = em_df.groupby("year_month")[
        ["electricity_kwh", "travel_km", "waste_kg"]
    ].sum()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    months_x = np.arange(len(monthly_activity))
    ticks_dates = monthly_activity.index.tolist()

    # Electricity
    axes[0].plot(months_x, monthly_activity["electricity_kwh"] / 1000.0, color="#1f77b4", marker="s", markersize=3, linewidth=1.5)
    axes[0].set_ylabel("Electricity\n(MWh / mo)")
    axes[0].set_title("Monthly Campus Activity Metric Trends (2023 - 2025)", pad=10)

    # Travel
    axes[1].plot(months_x, monthly_activity["travel_km"] / 1000.0, color="#ff7f0e", marker="^", markersize=3, linewidth=1.5)
    axes[1].set_ylabel("Commuting\n(1,000 km / mo)")

    # Waste
    axes[2].plot(months_x, monthly_activity["waste_kg"] / 1000.0, color="#2ca02c", marker="o", markersize=3, linewidth=1.5)
    axes[2].set_ylabel("Solid Waste\n(Tonnes / mo)")

    # Shading for summer break (June) and winter break (Dec) across all subplots
    for ax in axes:
        for i, ym in enumerate(ticks_dates):
            if ym.endswith("-06"):
                ax.axvspan(i - 0.4, i + 0.4, color="#d9d9d9", alpha=0.35)
            elif ym.endswith("-12"):
                ax.axvspan(i - 0.4, i + 0.4, color="#d9d9d9", alpha=0.35)

    q_idx = [i for i, d in enumerate(ticks_dates) if d.endswith(("-01", "-04", "-07", "-10"))]
    axes[2].set_xticks(q_idx)
    axes[2].set_xticklabels([ticks_dates[i] for i in q_idx], rotation=45, ha="right")
    axes[2].set_xlabel("Billing Period (YYYY-MM)")

    fig.tight_layout()
    p5 = output_dir / "05_activity_trends_seasonality.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved_paths["activity_trends"] = p5

    # -------------------------------------------------------------------------
    # Figure 6: Per-Student Carbon Intensity
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Boxplot of building-level emissions per student
    buildings = sorted(em_df["building"].unique())
    per_student_data = [em_df[em_df["building"] == b]["emissions_per_student_kg"].values for b in buildings]

    bp = ax.boxplot(
        per_student_data,
        patch_artist=True,
        tick_labels=[b.replace(" - ", "\n").replace(" & ", "\n& ") for b in buildings],
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.5},
    )
    for patch, col in zip(bp["boxes"], BUILDING_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    # Campus overall average line
    overall_mean = em_df["total_emissions_kg"].sum() / em_df["student_count"].sum()
    ax.axhline(overall_mean, color="#d62728", linestyle="--", linewidth=1.5, label=f"Campus Weighted Average ({overall_mean:.2f} kg/student-mo)")

    ax.set_ylabel("Emissions Intensity ($kgCO_2e$ / student-month)")
    ax.set_title("Figure 6: Per-Student Carbon Intensity Distribution by Facility", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    p6 = output_dir / "06_per_student_emissions.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    saved_paths["per_student_distribution"] = p6

    return saved_paths


def build_exploratory_notebook(output_notebook_path: Path) -> Path:
    """Build a complete, reproducible Jupyter Notebook for exploratory analysis."""
    cells: list[dict[str, Any]] = []

    def md_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")],
        }

    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")],
        }

    # Notebook Header
    cells.append(md_cell(
        """# Exploratory Data Analysis: Campus Carbon Emissions & Activity Dynamics
**Academic Project:** BDS-36 | T.Y. B.Sc. Data Science, Semester V | Academic Year 2026-27  
**Dataset:** Validated Campus Monthly Activity (`data/processed/cleaned_activity.csv`) & Carbon Accounting Output (`data/processed/campus_emissions.csv`)  
**Methodological Standard:** GHG Protocol Corporate Standard (Operational Control Approach)

---

### Epistemic & Methodological Notice:
1. **Non-Causal Framework:** All analyses herein report observed mathematical associations, seasonal patterns, and accounting distributions. **No unverified causal mechanisms are claimed**.
2. **Clear Distinction:**
   - **Observed Characteristics:** Empirical values, totals, and distributions resulting from the 36-month dataset.
   - **Underlying Parameterized Assumptions:** Academic semester calendars (vacations in June/Dec), climate cooling indices, and static documented emission factors from official sources (CEA India 2024, DEFRA 2023.1)."""
    ))

    # Cell 2: Imports & Environment
    cells.append(code_cell(
        """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = 'Arial'

print("Analysis environment ready.")"""
    ))

    # Cell 3: Data Loading
    cells.append(code_cell(
        """# Load validated activity data, computed carbon emissions, and factor registry
emissions_df = pd.read_csv('../data/processed/campus_emissions.csv')
factors_df = pd.read_csv('../data/emission_factors.csv')

emissions_df['date'] = pd.to_datetime(emissions_df['date'])
emissions_df['year'] = emissions_df['date'].dt.year
emissions_df['month'] = emissions_df['date'].dt.month
emissions_df['year_month'] = emissions_df['date'].dt.strftime('%Y-%m')

print(f"Emissions Dataset Shape: {emissions_df.shape[0]} rows x {emissions_df.shape[1]} columns")
print(f"Date Range: {emissions_df['date'].min().strftime('%Y-%m-%d')} to {emissions_df['date'].max().strftime('%Y-%m-%d')}")
print(f"Facilities Tracked ({len(emissions_df['building'].unique())}): {list(emissions_df['building'].unique())}")"""
    ))

    # Cell 4: Section 1 Markdown
    cells.append(md_cell(
        """---
## 1. Monthly Emissions Trajectory & Seasonal Cycles

### Analysis Questions:
- How do total campus emissions evolve over the 36-month timeline?
- Are recurring cyclical troughs and peaks observable across annual billing cycles?

![Figure 1: Monthly Campus Carbon Emissions Trajectory](../reports/figures/01_monthly_emissions_trend.png)"""
    ))

    # Cell 5: Section 1 Code
    cells.append(code_cell(
        """# Monthly aggregation across campus
monthly_totals = emissions_df.groupby('year_month').agg({
    'electricity_emissions_kg': 'sum',
    'travel_emissions_kg': 'sum',
    'waste_emissions_kg': 'sum',
    'procurement_emissions_kg': 'sum',
    'total_emissions_kg': 'sum'
}) / 1000.0  # Convert to tCO2e

print("Top 3 Peak Emission Months:")
print(monthly_totals['total_emissions_kg'].sort_values(ascending=False).head(3))

print("\\nTop 3 Lowest Emission Months (Troughs):")
print(monthly_totals['total_emissions_kg'].sort_values(ascending=True).head(3))"""
    ))

    # Cell 6: Section 1 Observations vs Assumptions
    cells.append(md_cell(
        """> [!NOTE]
> **Observations vs. Assumptions (Monthly Trajectory):**
> - **Empirical Observations:**
>   - Campus emissions reach recurring annual peaks in **April** (~131.8 tCO2e) and **August** (~136.3 tCO2e).
>   - Pronounced drops occur consistently in **June** (~74.7 tCO2e, a ~45% reduction from peak) and **December** (~69.0 tCO2e).
> - **Underlying Parameterized Assumptions:**
>   - The observed troughs reflect the academic calendar assumptions modeled in `src/data_generator.py` (summer break in May–June with 85% student headcount drop; winter break in December).
>   - The April and August peaks coincide with higher climate cooling factors (1.30–1.45) and active semester attendance."""
    ))

    # Cell 7: Section 2 Markdown
    cells.append(md_cell(
        """---
## 2. Yearly Emissions & Inter-Annual Growth

### Analysis Questions:
- What are the cumulative annual emissions for 2023, 2024, and 2025?
- What is the year-over-year rate of emissions change?

![Figure 2: Annual Campus Carbon Emissions by Category](../reports/figures/02_yearly_emissions_comparison.png)"""
    ))

    # Cell 8: Section 2 Code
    cells.append(code_cell(
        """yearly_summary = emissions_df.groupby('year').agg({
    'total_emissions_kg': 'sum',
    'electricity_emissions_kg': 'sum',
    'travel_emissions_kg': 'sum',
    'waste_emissions_kg': 'sum',
    'procurement_emissions_kg': 'sum'
})
yearly_summary['total_emissions_mt'] = yearly_summary['total_emissions_kg'] / 1000.0
yearly_summary['yoy_growth_pct'] = yearly_summary['total_emissions_kg'].pct_change() * 100.0

yearly_summary[['total_emissions_kg', 'total_emissions_mt', 'yoy_growth_pct']].round(2)"""
    ))

    # Cell 9: Section 2 Observations vs Assumptions
    cells.append(md_cell(
        """> [!NOTE]
> **Observations vs. Assumptions (Annual Trends):**
> - **Empirical Observations:**
>   - 2023 total: **1,361.81 tCO2e** ($1,361,814\text{ kg}$)
>   - 2024 total: **1,405.15 tCO2e** ($1,405,149\text{ kg}$, +3.18% YoY)
>   - 2025 total: **1,429.23 tCO2e** ($1,429,226\text{ kg}$, +1.71% YoY)
>   - 3-Year Total: **4,196.19 tCO2e** ($4,196,188.51\text{ kg}$)
> - **Underlying Parameterized Assumptions:**
>   - Secular growth rates of 2.0%–2.5% per annum were parameterized in the synthetic generator to reflect organic campus expansion and digital equipment adoption."""
    ))

    # Cell 10: Section 3 Markdown
    cells.append(md_cell(
        """---
## 3. Emission Category Contribution & Pareto Analysis

### Analysis Questions:
- Which activity category constitutes the primary driver of the campus carbon footprint?
- What are the proportional shares of Electricity, Commuter Travel, Solid Waste, and Procurement?

![Figure 3: Category Contribution to Total Campus Emissions](../reports/figures/03_category_contribution.png)"""
    ))

    # Cell 11: Section 3 Code
    cells.append(code_cell(
        """tot_elec = emissions_df['electricity_emissions_kg'].sum()
tot_travel = emissions_df['travel_emissions_kg'].sum()
tot_waste = emissions_df['waste_emissions_kg'].sum()
tot_proc = emissions_df['procurement_emissions_kg'].sum()
grand_total = emissions_df['total_emissions_kg'].sum()

category_breakdown = pd.DataFrame([
    {'Category': 'Electricity', 'Emissions (kg)': tot_elec, 'Emissions (t)': tot_elec / 1000.0, 'Share (%)': tot_elec / grand_total * 100},
    {'Category': 'Travel', 'Emissions (kg)': tot_travel, 'Emissions (t)': tot_travel / 1000.0, 'Share (%)': tot_travel / grand_total * 100},
    {'Category': 'Waste', 'Emissions (kg)': tot_waste, 'Emissions (t)': tot_waste / 1000.0, 'Share (%)': tot_waste / grand_total * 100},
    {'Category': 'Procurement', 'Emissions (kg)': tot_proc, 'Emissions (t)': tot_proc / 1000.0, 'Share (%)': tot_proc / grand_total * 100},
]).sort_values(by='Share (%)', ascending=False)

category_breakdown.round(2)"""
    ))

    # Cell 12: Section 3 Observations vs Assumptions
    cells.append(md_cell(
        """> [!NOTE]
> **Observations vs. Assumptions (Category Breakdown):**
> - **Empirical Observations:**
>   - **Electricity is the dominant driver**, accounting for **79.77%** ($3,347.35\text{ tCO}_2\text{e}$) of all emissions.
>   - **Travel** is the second largest driver at **14.16%** ($594.09\text{ tCO}_2\text{e}$).
>   - **Waste** contributes **5.20%** ($218.03\text{ tCO}_2\text{e}$).
>   - **Procurement** represents **0.87%** ($36.71\text{ tCO}_2\text{e}$).
> - **Underlying Parameterized Assumptions:**
>   - Electricity emission factor of **0.716 kgCO2e/kWh** (CEA India CO2 Baseline Database v19, 2024 generation-end average).
>   - Commuting factor of **0.140 kgCO2e/km** (DEFRA 2023.1 passenger mix).
>   - Waste factor of **0.446 kgCO2e/kg** (DEFRA 2023.1 commercial/industrial waste).
>   - Procurement spend factor of **0.00042 kgCO2e/INR** (CEDA/ADB EEIO 2023)."""
    ))

    # Cell 13: Section 4 Markdown
    cells.append(md_cell(
        """---
## 4. Facility / Building Carbon Intensity Breakdown

### Analysis Questions:
- How are emissions distributed across different functional building archetypes?
- Which building represents the highest priority target for decarbonization?

![Figure 4: Cumulative Emissions Breakdown by Campus Facility](../reports/figures/04_building_contribution.png)"""
    ))

    # Cell 14: Section 4 Code
    cells.append(code_cell(
        """building_summary = emissions_df.groupby('building').agg({
    'total_emissions_kg': 'sum',
    'electricity_emissions_kg': 'sum',
    'travel_emissions_kg': 'sum',
    'waste_emissions_kg': 'sum',
    'procurement_emissions_kg': 'sum',
    'student_count': 'mean',
    'staff_count': 'mean'
})
building_summary['total_emissions_t'] = building_summary['total_emissions_kg'] / 1000.0
building_summary['share_pct'] = building_summary['total_emissions_kg'] / grand_total * 100.0

building_summary[['total_emissions_t', 'share_pct', 'student_count', 'staff_count']].sort_values(by='total_emissions_t', ascending=False).round(2)"""
    ))

    # Cell 15: Section 4 Observations vs Assumptions
    cells.append(md_cell(
        """> [!NOTE]
> **Observations vs. Assumptions (Facility Distributions):**
> - **Empirical Observations:**
>   - **Science & Engineering Complex** produces the highest emissions: **1,317.28 tCO2e (31.39%)**.
>   - **Hostel & Dining Complex** ranks second: **920.98 tCO2e (21.95%)**.
>   - **Central Library & Student Hub** ranks third: **837.11 tCO2e (19.95%)**.
>   - **Academic Block - Arts & Humanities**: **649.56 tCO2e (15.48%)**.
>   - **Administrative Headquarters**: **471.25 tCO2e (11.23%)**.
> - **Underlying Parameterized Assumptions:**
>   - S&E complex profile reflects high base electrical loads (42,000 kWh/mo) due to research equipment and continuous lab ventilation.
>   - Hostel profile has lower commuter travel (residents live on campus) but highest per-capita solid waste generation (6.5 kg/student-month) from cafeteria operations."""
    ))

    # Cell 16: Section 5 Markdown
    cells.append(md_cell(
        """---
## 5. Primary Activity Trends: Electricity, Commute, and Solid Waste

### Analysis Questions:
- What are the underlying seasonal rhythms in raw consumption units (kWh, km, kg)?
- Do activity rhythms align with institutional academic and fiscal calendars?

![Figure 5: Activity Time Series & Seasonality Multi-panel](../reports/figures/05_activity_trends_seasonality.png)"""
    ))

    # Cell 17: Section 5 Code
    cells.append(code_cell(
        """monthly_activity_stats = emissions_df.groupby('month').agg({
    'electricity_kwh': ['mean', 'min', 'max'],
    'travel_km': ['mean', 'min', 'max'],
    'waste_kg': ['mean', 'min', 'max'],
    'procurement_inr': ['mean', 'min', 'max']
}).round(1)

monthly_activity_stats"""
    ))

    # Cell 18: Section 5 Observations vs Assumptions
    cells.append(md_cell(
        """> [!NOTE]
> **Observations vs. Assumptions (Activity Metric Trends):**
> - **Empirical Observations:**
>   - **Electricity (kWh):** Reaches maximum monthly averages in April (30,347 kWh/bldg) and August (30,004 kWh/bldg). Minimum occurs in December (16,437 kWh/bldg).
>   - **Commuting (km):** Peak commuter travel occurs in October (30,260 km/bldg) and August (29,974 km/bldg); sharp drop in June (6,664 km/bldg, a 78% drop).
>   - **Waste (kg):** Bimodal surges in February (3,502 kg/bldg) and October (3,494 kg/bldg).
>   - **Procurement (INR):** Notable spending spikes in March (₹740,776/bldg average) and July (₹683,522/bldg average).
> - **Underlying Parameterized Assumptions:**
>   - March procurement surge reflects Indian fiscal year closeout budget utilization.
>   - July procurement surge reflects new academic year commencement (laboratory consumables, library books, orientation materials).
>   - Waste spikes in February and October reflect modeled campus festival events."""
    ))

    # Cell 19: Section 6 Markdown
    cells.append(md_cell(
        """---
## 6. Per-Student Carbon Intensity Analysis

### Analysis Questions:
- What is the overall carbon intensity per student across the campus?
- How does carbon intensity vary between specialized laboratory facilities and general lecture blocks?

![Figure 6: Per-Student Carbon Intensity Distribution](../reports/figures/06_per_student_emissions.png)"""
    ))

    # Cell 20: Section 6 Code
    cells.append(code_cell(
        """# Overall campus emissions per student-month
overall_student_months = emissions_df['student_count'].sum()
campus_intensity = emissions_df['total_emissions_kg'].sum() / overall_student_months

print(f"Overall Campus Emissions Intensity: {campus_intensity:.2f} kgCO2e per student-month")

# Building level intensity
bldg_intensity = emissions_df.groupby('building').apply(
    lambda g: g['total_emissions_kg'].sum() / g['student_count'].sum(),
    include_groups=False
).round(2)

print("\\nCarbon Intensity by Facility (kgCO2e / student-month):")
print(bldg_intensity.sort_values(ascending=False))"""
    ))

    # Cell 21: Section 6 Observations vs Assumptions
    cells.append(md_cell(
        """> [!NOTE]
> **Observations vs. Assumptions (Per-Student Intensity):**
> - **Empirical Observations:**
>   - The overall campus-wide intensity is **25.48 kgCO2e per student-month**.
>   - **Administrative Headquarters** exhibits an apparent high ratio (~54.0 kgCO2e/student-month) because student presence in administrative offices is low (staff-dominated operations).
>   - **Science & Engineering Complex** has an intensity of **32.83 kgCO2e/student-month**.
>   - **Academic Block - Arts & Humanities** has the lowest intensity: **15.29 kgCO2e/student-month**.
> - **Underlying Parameterized Assumptions:**
>   - Arts & Humanities has high classroom density (base 1,400 students) and low baseline equipment, whereas Science & Engineering supports intensive laboratory equipment over a smaller student cohort."""
    ))

    # Cell 22: Synthesis & Summary Table
    cells.append(md_cell(
        """---
## 7. Synthesis & Decarbonization Implications

| Dimension | Empirical Observation | Modeling Assumption | Strategic Decarbonization Implication |
| :--- | :--- | :--- | :--- |
| **Primary Category** | Electricity accounts for 79.8% of emissions | Grid emission factor = 0.716 kgCO2e/kWh (CEA 2024) | On-site solar PV and energy efficiency offer the highest carbon reduction leverage per dollar. |
| **Secondary Category** | Travel accounts for 14.2% of emissions | Commute factor = 0.140 kgCO2e/km (DEFRA 2023) | EV campus transit shuttles and hybrid/remote academic scheduling offer moderate reduction. |
| **Facility Priority** | Science Complex + Hostels = 53.3% of total | High equipment load (Science) + 24/7 dining/living (Hostels) | Interventions should prioritize lab HVAC optimization and kitchen waste recovery. |
| **Temporal Seasonality** | Recurring April/August peaks, June/Dec troughs | Summer cooling index (1.45) + semester attendance cycles | Demand management during peak cooling months will yield substantial emissions savings. |

---
*Report generated for BDS-36 Academic Analytics prototype. All activity data synthetic; all emission factors externally documented.*"""
    ))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14.2",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=1)

    return output_notebook_path


def main() -> None:
    print("=================================================================")
    print("EXPLORATORY CARBON ANALYSIS & NOTEBOOK GENERATOR")
    print("=================================================================")
    print("Loading validated datasets...")
    act_df, em_df, ef_df = load_datasets()

    figures_dir = Path("reports/figures")
    print(f"Generating publication-quality charts in {figures_dir.resolve()}...")
    figures = generate_figures(em_df, figures_dir)
    for name, path in figures.items():
        print(f"  - Saved figure [{name}]: {path}")

    notebook_path = Path("notebooks/exploratory_analysis.ipynb")
    print(f"\nBuilding exploratory notebook at: {notebook_path.resolve()}...")
    build_exploratory_notebook(notebook_path)
    print(f"Notebook successfully created at: {notebook_path}")

    print("\n=== SUMMARY OF KEY FINDINGS ===")
    tot_mt = em_df["total_emissions_kg"].sum() / 1000.0
    print(f"Total Cumulative Emissions (3-Year): {tot_mt:,.2f} MTCO2e ({em_df['total_emissions_kg'].sum():,.2f} kg)")
    print(f"Overall Intensity: {em_df['total_emissions_kg'].sum() / em_df['student_count'].sum():.2f} kgCO2e/student-month")
    print("Category Breakdown:")
    for cat in ["electricity", "travel", "waste", "procurement"]:
        val = em_df[f"{cat}_emissions_kg"].sum()
        pct = val / em_df["total_emissions_kg"].sum() * 100
        print(f"  - {cat.capitalize():<12}: {val / 1000.0:>8,.2f} tCO2e ({pct:>5.2f}%)")

    print("\nBuilding Breakdown:")
    for bldg, grp in em_df.groupby("building"):
        b_val = grp["total_emissions_kg"].sum()
        b_pct = b_val / em_df["total_emissions_kg"].sum() * 100
        print(f"  - {bldg:<36}: {b_val / 1000.0:>8,.2f} tCO2e ({b_pct:>5.2f}%)")
    print("=================================================================")


if __name__ == "__main__":
    main()
