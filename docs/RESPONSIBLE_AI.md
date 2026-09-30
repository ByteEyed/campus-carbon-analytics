# Responsible AI, Model Governance & Provenance Specification
**Campus Carbon Analytics** | Academic Project BDS-36 (T.Y. B.Sc. Data Science, Semester V, 2026-27)

---

## 1. Executive Summary & Purpose

The **Campus Carbon Analytics System** is an educational, research-grade decision-support platform designed to quantify, forecast, and optimize greenhouse gas (GHG) decarbonization pathways for higher education institutions.

In accordance with responsible AI standards, academic integrity guidelines, and environmental data ethics, this document defines the provenance of data inputs, mathematical model boundaries, error propagation characteristics, and ethical guidelines governing decision-support recommendations.

---

## 2. Data Provenance & Integrity Classification

The analytics pipeline maintains strict separation between synthetic empirical activity proxies and externally validated regulatory emission factors:

```
+-----------------------------------------------------------------------------------+
|                                  DATA STREAMS                                     |
+----------------------------------------+------------------------------------------+
|          SYNTHETIC ACTIVITY            |       DOCUMENTED EMISSION FACTORS        |
|  - Electricity consumption (kWh)       |  - Central Electricity Authority (CEA)   |
|  - Commuter travel distance (km)       |  - India GHG Program Transport Tool      |
|  - Solid waste generation (kg)         |  - UK DESNZ / DEFRA Solid Waste Mix      |
|  - Operational procurement spend (INR) |  - ADB / CEDA Multipliers                |
|  [Simulated Academic Data Proxy]       |  [Externally Documented Standards]       |
+----------------------------------------+------------------------------------------+
```

### 2.1 Synthetic Campus Activity Data
* **Provenance:** Programmatically simulated via `src/data_generator.py` calibrated against published academic campus energy consumption patterns (5 campus facilities, 36-month timeline: Jan 2023 – Dec 2025).
* **Intended Use:** Serves as an academic demonstration dataset reflecting realistic seasonal, academic-term, and building-level heterogeneity without violating institutional privacy or exposing non-public metering records.
* **Limitations:** Does not represent actual utility bills or audited operational logs of any named university.

### 2.2 Emission Factors Provenance
Emission factors are immutable, external constants documented in `data/emission_factors.csv`:
1. **Grid Electricity:** $0.716\text{ kgCO}_2\text{e/kWh}$ (CEA CO2 Baseline Database for Indian Power Sector, v19.0/20.0).
2. **Passenger Travel:** $0.140\text{ kgCO}_2\text{e/km}$ (India GHG Program / UK DESNZ transit passenger mix).
3. **Solid Waste:** $0.446\text{ kgCO}_2\text{e/kg}$ (UK DESNZ / DEFRA Commercial & Municipal Waste).
4. **Institutional Procurement:** $0.00042\text{ kgCO}_2\text{e/INR}$ (ADB/CEDA Input-Output GHG Multiplier for Indian Higher Education).

---

## 3. Model Cards & Algorithmic Specifications

### 3.1 Primary Forecasting Engine: Additive Holt-Winters Exponential Smoothing
* **Mathematical Specification:**
  $$\hat{y}_{t+h} = \ell_t + h b_t + s_{t + h - m(k+1)}$$
  Captures additive trend ($\beta$) and 12-month seasonality ($\gamma$) for campus operational cycles.
* **Prediction Intervals:** Residual-bootstrap Monte Carlo simulations (2,000 paths) generating empirical 95% intervals bounded at zero ($y \ge 0$).
* **Intended Application:** 12-month baseline trend extrapolation for campus facility planning.
* **Out-of-Scope Use:** Multi-year long-range climate projections (> 24 months) or black-swan disruption modeling (pandemics, campus closures).

### 3.2 Uncertainty Quantification: Monte Carlo Variance Decomposition
* **Methodology:** One-At-a-Time (OAT) parameter perturbation across:
  1. Forecast Model Variance ($\sigma^2_{fc}$)
  2. Activity Data Measurement Variance ($\sigma^2_{act}$, $\text{RSD} = 5\%$)
  3. Emission Factor Carbon Intensity Variance ($\sigma^2_{ef}$, $\text{RSD} = 7\%$)
* **Total Variance Decomposition:**
  $$\text{Var}_{\text{total}} = \sigma^2_{fc} + \sigma^2_{act} + \sigma^2_{ef}$$
  Exposes the relative contribution of data quality vs. carbon intensity vs. temporal volatility, guiding where institutional measurement investment yields maximum precision.

### 3.3 Decarbonization Intervention Scenario Library
* **Mechanism:** Pure functional transformations on Activity DataFrames:
  - Multiplicative load reductions: $X_{new} = X_{old} \times (1 - r)$
  - Additive clean generation subtraction: $X_{new} = \max(0, X_{old} - G_{solar})$
* **Engineering Assumptions:** Standardised capital expenditures, lifespans (5–25 years), and technical efficiencies based on Indian market benchmarks (2024–2026).
* **Boundary Safeguards:** Clamped strictly to physical non-negativity ($X \ge 0, E \ge 0$).

### 3.4 Budget Optimization: Powerset Frontier Search
* **Methodology:** Deterministic exhaustive enumeration of $2^5 = 32$ intervention portfolios.
* **Objective:** Maximize total emissions reduction subject to capital expenditure constraint $\sum C_i \le B$.
* **Transparency:** No black-box heuristic search; 100% auditable solution space exposing dominated and Pareto-optimal choices.

---

## 4. Ethical Considerations & Responsible Decision-Support

### 4.1 Decision Support vs. Automated Governance
> [!IMPORTANT]
> The recommendations produced by this pipeline (such as optimal portfolio `P-10110`) represent **advisory decision-support analytics**. They must never be applied autonomously. Institutional administrators, facilities directors, and sustainability committees must review technical feasibility, architectural constraints, and localized campus needs.

### 4.2 Environmental Justice & Equity
- Energy efficiency and HVAC adjustments must not compromise student learning environments, laboratory safety, or thermal comfort in residential hostels.
- Transit interventions (electric shuttle routing) must prioritize equitable campus access for commuting students and shift workers.

### 4.3 Transparency & Reproducibility
- All analytical components are deterministic under `random_seed=42`.
- Zero proprietary dependencies; full codebase is inspectable and test-covered (266 unit/integration tests).
- Precomputed artifacts are serialized in human-readable CSV and JSON formats.

---

## 5. Summary of Governance Guardrails

| Guardrail Domain | Implementation | Security / Responsible Mechanism |
| :--- | :--- | :--- |
| **Emission Factors** | `EmissionFactorRegistry` | Strictly positive check ($EF > 0$), rejection of unvetted zeros |
| **Intervention Economics** | `BaseIntervention` | Capital cost non-negativity ($C \ge 0$), bounded reduction fractions ($0 \le r \le 1$) |
| **Physical Reality** | Scenario & Portfolio Engines | Hard clamping at zero ($E_{scenario} \ge 0$); impossible negative emissions blocked |
| **Data Provenance** | Streamlit Dashboard & Logs | Prominent disclaimer banners and institutional metadata display |
| **File System Security** | `PipelineConfig` | Strict rejection of path traversal (`..`) and protected system directories |
