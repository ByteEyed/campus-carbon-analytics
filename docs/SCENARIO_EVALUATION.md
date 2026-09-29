# Phase 14: Decarbonization Scenario Evaluation & Marginal Abatement Cost (MAC)

**Academic Project:** BDS-36 (T.Y. B.Sc. Data Science, Semester V)  
**Academic Year:** 2026-27  
**Module:** [`src.scenarios.evaluation`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/scenarios/evaluation.py)  
**Test Suite:** [`tests/test_scenario_evaluation.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/tests/test_scenario_evaluation.py)  
**Artifact Generated:** [`data/processed/scenario_evaluation_summary.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/scenario_evaluation_summary.csv)  

---

## 1. Objective & Scenario Evaluation Methodology

Phase 14 establishes the **Isolated Scenario Evaluation Engine** for evaluating campus decarbonization interventions.

### 1.1 Isolated One-At-a-Time (OAT) Protocol
To determine the true marginal effectiveness of each candidate intervention without conflating interactive effects or cannibalization, each intervention defined in Phase 13 is evaluated **strictly in isolation**:

1. **Immutable Baseline Horizon:** A standardized 12-month Business-as-Usual (BAU) Activity DataFrame ($5\text{ facilities} \times 12\text{ months} = 60\text{ records}$) is loaded and defensively cloned (`df.copy()`).
2. **Baseline Carbon Accounting:** The baseline DataFrame is passed into the Phase 8 Carbon Accounting Engine (`calculate_campus_emissions`) to compute baseline emissions ($E_{\text{baseline}}$).
3. **Black-Box Transformation:** The cloned activity data is passed into the intervention's `.apply()` method, returning a transformed Activity DataFrame ($X_{\text{scenario}}$).
4. **Scenario Carbon Accounting:** The transformed activity table is passed into the Carbon Accounting Engine to compute scenario emissions ($E_{\text{scenario}}$).
5. **Abatement & Cost Efficiency Metrics:** The absolute avoided emissions ($R_{\text{abs}}$), percentage reduction ($R_{\text{pct}}$), and Marginal Abatement Cost ($MAC$) are computed.
6. **Sensitivity Propagation:** Phase 12 activity uncertainty bounds are pushed through the pipeline to determine the lower and upper bounds of carbon reduction.

```
       [12-Month Baseline Activity Data]
             (Immutable df.copy())
             │                  │
             ▼                  ▼
    [Phase 8 Accounting]   [Phase 13 Intervention]
             │                  │
             │                  ▼
             │         [Modified Activity Data]
             │                  │
             │                  ▼
             │         [Phase 8 Accounting]
             │                  │
             ▼                  ▼
     (E_baseline)        (E_scenario)
             │                  │
             └─────────┬────────┘
                       ▼
            [Impact Delta Engine]
          - Absolute Reduction (kgCO2e)
          - Percentage Reduction (%)
          - Marginal Abatement Cost (INR/kg)
          - Phase 12 Sensitivity Bounds
```

---

## 2. Mathematical Formulations

### 2.1 Absolute Emissions Reduction ($R_{\text{abs}}$)
The total physical greenhouse gas emissions avoided over the 12-month evaluation horizon:
$$R_{\text{abs}} = E_{\text{baseline}} - E_{\text{scenario}} \quad (\text{kgCO}_2\text{e})$$

### 2.2 Percentage Emissions Reduction ($R_{\text{pct}}$)
The relative emissions reduction relative to the campus baseline:
$$R_{\text{pct}} = \begin{cases} \left( \frac{R_{\text{abs}}}{E_{\text{baseline}}} \right) \times 100\% & \text{if } E_{\text{baseline}} > 0 \\ 0.0\% & \text{if } E_{\text{baseline}} = 0 \end{cases}$$

### 2.3 Marginal Abatement Cost ($MAC$)
The financial capital investment required per kilogram of $\text{CO}_2\text{e}$ abated over the evaluation horizon:
$$MAC = \begin{cases} \frac{C_{\text{implementation}}}{R_{\text{abs}}} & \text{if } R_{\text{abs}} > 0 \\ +\infty & \text{if } R_{\text{abs}} = 0 \text{ and } C_{\text{implementation}} \ge 0 \\ \frac{C_{\text{implementation}}}{R_{\text{abs}}} \, (< 0) & \text{if } R_{\text{abs}} < 0 \text{ (Maladaptive increase)} \end{cases}$$

#### Key Properties of the MAC Formulation:
- **Lower MAC is Better:** A lower MAC indicates greater capital efficiency (fewer rupees spent per kg $\text{CO}_2\text{e}$ eliminated).
- **Division-by-Zero Safety:** If an intervention produces zero physical reduction ($R_{\text{abs}} = 0$), $MAC$ is defined as `float('inf')` (`+∞`), accurately reflecting an infinite cost per unit of carbon reduced.
- **Maladaptive Transparency:** If an intervention increases emissions ($R_{\text{abs}} < 0$, e.g. severe embodied carbon rebound), $R_{\text{abs}}$ and $MAC$ remain negative and are **never** silently clipped to zero via `max(0, ...)`.

---

## 3. Intervention-Specific Calculation Behaviors

Phase 14 treats Phase 13 interventions as black boxes, honoring their pure functional contracts:

1. **LED Lighting Retrofit (`INT-001`):** Reduces `electricity_kwh` by 12% across all 12 months. Affects only electricity emissions and total emissions; other categories remain identical to baseline.
2. **Rooftop Solar Installation (`INT-002`):** Subtracts 5,000 kWh/month of behind-the-meter generation across campus facilities. Total annual electricity avoided is $12 \times 5,000 = 60,000\text{ kWh}$. Multiplying by the CEA grid emission factor ($0.716\text{ kgCO}_2\text{e/kWh}$) yields exactly $42,960.00\text{ kgCO}_2\text{e}$ avoided.
3. **AC / HVAC Optimization (`INT-003`):** Curtains electricity consumption by 15% strictly during summer months (April–August, months 4 through 8). Captures cooling seasonality naturally.
4. **Waste Segregation & Composting (`INT-004`):** Diverts 35% of solid waste (`waste_kg`) away from municipal landfills. Phase 8 applies the DESNZ landfill emission factor ($0.446\text{ kgCO}_2\text{e/kg}$) to the diverted tonnage.
5. **Low-Carbon Transport (`INT-005`):** Substitutes 20% of commuter vehicle travel distance (`travel_km`) with zero-emission electric shuttles, eliminating passenger combustion emissions ($0.140\text{ kgCO}_2\text{e/km}$).

---

## 4. Phase 12 Uncertainty Bounds Integration

Rather than recalculating uncertainty from scratch, Phase 14 directly propagates the Phase 12 Truncated Normal activity relative standard deviations ($\sigma_{\text{act}}$) through the baseline:

$$\text{Activity}_{\text{lower}} = \max\left(0.0, \, \text{Activity} \times (1 - 1.96 \times \sigma_{\text{act}})\right)$$
$$\text{Activity}_{\text{upper}} = \text{Activity} \times (1 + 1.96 \times \sigma_{\text{act}})$$

The lower and upper activity DataFrames are passed through the intervention transformation and carbon accounting engine, generating:
- `absolute_reduction_lower_kg`: Abatement achieved if activity runs at the lower 95% sensitivity bound.
- `absolute_reduction_upper_kg`: Abatement achieved if activity runs at the upper 95% sensitivity bound.

> [!NOTE]
> For additive rooftop solar generation, the lower and upper reduction bounds are identical ($42,959.99\text{ kgCO}_2\text{e}$) because monthly building demand significantly exceeds solar generation ($5,000\text{ kWh}$) across all sensitivity states, preventing curtailment. For multiplicative measures (LED, HVAC, Waste, Travel), reduction scales proportionally with activity variance.

---

## 5. Verified Empirical Results

Execution of `python -m src.scenarios.evaluation` on the canonical 12-month forward baseline (2025-01-01 to 2025-12-01) produced the following verified results:

- **Baseline Total Campus Emissions:** $1,429,225.82\text{ kgCO}_2\text{e}$ ($1,429.23\text{ MTCO}_2\text{e}$)
- **Baseline Electricity Emissions:** $1,140,467.43\text{ kgCO}_2\text{e}$ ($79.8\%$ of total)
- **Baseline Travel Emissions:** $203,173.10\text{ kgCO}_2\text{e}$ ($14.2\%$ of total)
- **Baseline Waste Emissions:** $72,815.16\text{ kgCO}_2\text{e}$ ($5.1\%$ of total)
- **Baseline Procurement Emissions:** $12,770.14\text{ kgCO}_2\text{e}$ ($0.9\%$ of total)

### Table 1: Standalone Scenario Evaluation & Marginal Abatement Cost (MAC) Ranking
Artifact: [`data/processed/scenario_evaluation_summary.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/processed/scenario_evaluation_summary.csv)

| MAC Rank | ID | Intervention Name | Affected Category | Capital Cost (INR) | Absolute Reduction (kgCO2e) | % Campus Reduction | Cost per kg Reduced (INR/kg) | 95% Sensitivity Bounds (kgCO2e) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `INT-001` | **LED Lighting Retrofit** | `Electricity` | ₹5,00,000 | **136,856.07** | **9.58%** | **₹3.65** | [131,491.34, 142,220.88] |
| **2** | `INT-003` | **AC / HVAC Optimization** | `Electricity` | ₹3,00,000 | **72,060.97** | **5.04%** | **₹4.16** | [69,236.17, 74,885.75] |
| **3** | `INT-004` | **Waste Segregation & Composting** | `Waste` | ₹1,50,000 | **25,485.30** | **1.78%** | **₹5.89** | [20,490.17, 30,480.44] |
| **4** | `INT-005` | **Low-Carbon / Sustainable Transport** | `Travel` | ₹18,00,000 | **40,634.62** | **2.84%** | **₹44.30** | [28,688.04, 52,581.20] |
| **5** | `INT-002` | **Rooftop Solar Installation** | `Electricity` | ₹25,00,000 | **42,959.98** | **3.01%** | **₹58.19** | [42,959.99, 42,959.99] |

---

## 6. Institutional Decision-Making Insights

### 6.1 The Efficiency vs. Generation Hierarchy
The Marginal Abatement Cost analysis reveals a crucial insight for campus sustainability directors:
1. **Energy Efficiency is $16\times$ More Cost-Effective Than Solar PV:**
   - **LED Lighting Retrofit** achieves the highest abatement ($136.86\text{ MTCO}_2\text{e}$) at the lowest cost (**₹3.65 / kg**).
   - **AC / HVAC Optimization** achieves $72.06\text{ MTCO}_2\text{e}$ at only **₹4.16 / kg**.
   - **Rooftop Solar Installation**, while physically valuable for long-term zero-carbon electricity, has a capital-intensive MAC of **₹58.19 / kg**.
   - **Recommendation:** Campuses with constrained upfront capital should exhaust all energy efficiency retrofits (LED, HVAC) before deploying large-scale capital into rooftop solar.

2. **Waste Diversion Offers High Marginal Value at Low Absolute Budget:**
   - For an investment of only ₹1,50,000, waste segregation diverts $25.49\text{ MTCO}_2\text{e}$ at **₹5.89 / kg**, making it an ideal quick-win project for immediate ESG impact.

---

## 7. Assumptions & Known Limitations

1. **Standalone OAT Isolation Limits:**  
   In Phase 14, every intervention is evaluated strictly in isolation. Summing their individual reductions ($136.86 + 72.06 + 25.49 + 40.63 + 42.96 = 318.00\text{ MTCO}_2\text{e}$) would overstate combined potential due to **interactive cannibalization** (e.g. LED and HVAC reduce gross electricity consumption, leaving less grid power for solar to offset). Multi-intervention stacking and budget portfolio optimization are deferred to Phase 15.
2. **Static Emission Factors:**  
   Conversion factors are assumed constant over the 12-month horizon. In multi-decade evaluations, grid greening naturally reduces the marginal abatement of rooftop solar.
3. **Simulated Financial Provenance:**  
   Capital costs are prototype assumptions (`is_simulated_assumption = True`) intended to model realistic institutional scales (TRL 4).

---

## 8. Explicit Out of Scope Declarations

In strict adherence to Phase 14 scope boundaries:
- **NO portfolio stacking or multi-intervention aggregation** (Reserved for Phase 15).
- **NO cannibalization resolution between overlapping interventions** (Reserved for Phase 15).
- **NO Knapsack or Integer Programming budget optimizers** (Reserved for Phase 15).
- **NO interactive Streamlit dashboard toggles or UI widgets** (Reserved for Phase 15).
