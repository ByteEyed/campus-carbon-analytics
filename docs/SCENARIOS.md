# Phase 13: Decarbonization Intervention Scenario Library

**Academic Project:** BDS-36 (T.Y. B.Sc. Data Science, Semester V)  
**Academic Year:** 2026-27  
**Module:** [`src.scenarios.interventions`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/scenarios/interventions.py)  
**Test Suite:** [`tests/test_interventions.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/tests/test_interventions.py)  

---

## 1. Objective & Architecture Overview

Phase 13 establishes the foundational **Decarbonization Intervention Scenario Library** for campus carbon abatement. It defines the standardized physical intervention contracts and activity transformation mechanics for five canonical campus sustainability initiatives:

1. **INT-001: LED Lighting Retrofit**
2. **INT-002: Rooftop Solar Installation**
3. **INT-003: AC / HVAC Optimization**
4. **INT-004: Waste Segregation & Composting**
5. **INT-005: Low-Carbon / Sustainable Transport**

### The Activity Transformation Paradigm

A core architectural principle of this system is the **separation of physical activity modification from emissions calculation**.

Interventions **do not calculate carbon emissions directly**. Rather, interventions act as pure functional transformations on the *Activity DataFrame*:

```
[Raw / Cleaned Activity DataFrame]
               │
               ▼
   [Intervention Engine (.apply)]
      (Transforms physical activity:
       kWh, km, kg waste, etc.)
               │
               ▼
[Transformed Activity DataFrame]
               │
               ▼
[Phase 8 Carbon Accounting Engine]
    (calculate_campus_emissions)
               │
               ▼
   [Reported Campus Emissions]
```

#### Why This Paradigm Matters:
1. **Preservation of Provenance:** Emission factors from [`data/emission_factors.csv`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/emission_factors.csv) (e.g., Central Electricity Authority grid factor $0.716\text{ kgCO}_2\text{e/kWh}$) remain the single source of truth for carbon conversion. Interventions only model physical reductions.
2. **Zero Code Duplication:** Interventions do not duplicate the unit checks, per-student calculations, or reporting logic of [`src/carbon_accounting.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py).
3. **Composability:** Transformed activity tables can be passed seamlessly into subsequent downstream stages, including Phase 11 forecasting models and Phase 12 uncertainty simulations.

---

## 2. Simulated Assumptions & Provenance Transparency

> [!CAUTION]
> ### CRITICAL DISCLAIMER: SIMULATED ACADEMIC ASSUMPTIONS
> All intervention capital costs ($\text{INR } ₹$), expected operational lifespans ($\text{years}$), reduction fractions ($\%$, and solar generation outputs ($\text{kWh}$) implemented in this library are **simulated prototype assumptions** chosen for this undergraduate academic demonstration (TRL 4).
>
> They are based on illustrative clean-energy benchmark ranges (e.g., Bureau of Energy Efficiency India, MNRE rooftop solar guidelines, and academic campus retrofit case studies), but **DO NOT** represent audited engineering contractor bids or vendor quotations. Every intervention class enforces `is_simulated_assumption = True`.
>
> Users and facility planners can override any parameter in the class constructors when real-world engineering audit data is available.

---

## 3. Intervention Catalog & Specification

### Table 1: Intervention Library Catalog
| ID | Name | Category | Target Column | Mechanism | Priority | Default Capital Cost (INR) | Lifespan | Default Parameter |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **INT-001** | LED Lighting Retrofit | `Electricity` | `electricity_kwh` | Multiplicative | 1 | ₹5,00,000 | 10 yrs | $r_{\text{led}} = 12\%$ |
| **INT-002** | Rooftop Solar Installation | `Electricity` | `electricity_kwh` | Additive | 2 | ₹25,00,000 | 25 yrs | $G_{\text{solar}} = 5,000\text{ kWh/mo}$ |
| **INT-003** | AC / HVAC Optimization | `Electricity` | `electricity_kwh` | Multiplicative (Seasonal) | 1 | ₹3,00,000 | 7 yrs | $r_{\text{hvac}} = 15\%$, Months: 4–8 |
| **INT-004** | Waste Segregation & Composting | `Waste` | `waste_kg` | Multiplicative | 1 | ₹1,50,000 | 5 yrs | $r_{\text{diverted}} = 35\%$ |
| **INT-005** | Low-Carbon / Sustainable Transport | `Travel` | `travel_km` | Multiplicative | 1 | ₹18,00,000 | 8 yrs | $r_{\text{ev}} = 20\%$ |

---

## 4. Mathematical Formulations & Physical Mechanics

### 4.1 INT-001: LED Lighting Retrofit (`LEDLightingRetrofit`)
- **Target Category:** `Electricity`
- **Target Column:** `electricity_kwh`
- **Mechanism:** Multiplicative Activity Reduction
- **Mathematical Formula:**
  $$X_{\text{new}} = X_{\text{old}} \times (1.0 - r_{\text{led}})$$
- **Parameters:**
  - `reduction_fraction` ($r_{\text{led}}$): Fractional reduction in electricity consumption ($0.0 \le r_{\text{led}} \le 1.0$). Default: `0.12` (12%).
  - `capital_cost`: ₹5,00,000 INR (simulated fixtures, installation, and labor).
  - `lifespan_years`: 10 years (standard L70 rated LED operational life).
- **Physical Rationale:** High-efficiency LED luminaires reduce base lighting loads by ~40-60%. Because lighting represents approximately 20-25% of commercial campus building loads, the whole-building net electricity curtailment is modeled at 12%.

---

### 4.2 INT-002: Rooftop Solar Installation (`RooftopSolarInstallation`)
- **Target Category:** `Electricity`
- **Target Column:** `electricity_kwh`
- **Mechanism:** Additive Behind-the-Meter Generation Subtraction
- **Mathematical Formula:**
  $$X_{\text{new}} = \max\left(0.0, \, X_{\text{old}} - G_{\text{solar}}\right)$$
- **Parameters:**
  - `monthly_generation_kwh` ($G_{\text{solar}}$): Monthly solar PV generation ($G_{\text{solar}} \ge 0.0$). Default: `5000.0` kWh/month (~40 kWp rooftop array operating at standard Indian solar insolation).
  - `target_building`: Optional specific facility identifier. If `None` and multiple campus buildings exist for the billing period, solar generation is allocated proportionally to building electricity demand.
  - `capital_cost`: ₹25,00,000 INR (~₹62,500/kWp turnkey installation).
  - `lifespan_years`: 25 years (photovoltaic module linear warranty standard).
- **Physical Rationale:** Rooftop solar PV generates power behind the institutional utility meter, directly reducing grid electricity imports. The $\max(0.0, \dots)$ clamping prevents negative grid imports in the absence of net-metering export contracts.

---

### 4.3 INT-003: AC / HVAC Optimization (`ACOptimization`)
- **Target Category:** `Electricity`
- **Target Column:** `electricity_kwh`
- **Mechanism:** Multiplicative Activity Reduction with Seasonal Masking
- **Mathematical Formula:**
  $$X_{\text{new}} = X_{\text{old}} \times \left(1.0 - \left(r_{\text{hvac}} \times I_{\text{active}}(t)\right)\right)$$
  $$\text{where } I_{\text{active}}(t) = \begin{cases} 1 & \text{if } \text{month}(t) \in \text{active\_months} \\ 0 & \text{otherwise} \end{cases}$$
- **Parameters:**
  - `reduction_fraction` ($r_{\text{hvac}}$): Fractional curtailment during active cooling season ($0.0 \le r_{\text{hvac}} \le 1.0$). Default: `0.15` (15%).
  - `active_months`: Calendar months subject to cooling optimization. Default: `(4, 5, 6, 7, 8)` (April through August — peak summer cooling months in the Indian subcontinent).
  - `capital_cost`: ₹3,00,000 INR (smart thermostats, variable refrigerant flow tuning, duct insulation).
  - `lifespan_years`: 7 years (sensor recalibration and control hardware life).
- **Physical Rationale:** Space cooling constitutes over 40% of building electricity during hot summer months but is idle during winter. A seasonal indicator $I_{\text{active}}(t)$ prevents artificial energy reductions during off-season months.

---

### 4.4 INT-004: Waste Segregation & Composting (`WasteSegregation`)
- **Target Category:** `Waste`
- **Target Column:** `waste_kg`
- **Mechanism:** Multiplicative Solid Waste Diversion
- **Mathematical Formula:**
  $$X_{\text{new}} = X_{\text{old}} \times (1.0 - r_{\text{diverted}})$$
- **Parameters:**
  - `diversion_fraction` ($r_{\text{diverted}}$): Fraction of municipal solid waste diverted to organic composting ($0.0 \le r_{\text{diverted}} \le 1.0$). Default: `0.35` (35%).
  - `capital_cost`: ₹1,50,000 INR (compost tumblers, color-coded dual-bin infrastructure, shredders).
  - `lifespan_years`: 5 years.
- **Physical Rationale:** Biodegradable canteen, garden, and dining waste represents 35-50% of campus solid waste. Diverting it to aerobic composting eliminates anaerobic landfill decomposition and its associated methane ($\text{CH}_4$) emissions.

---

### 4.5 INT-005: Low-Carbon / Sustainable Transport (`LowCarbonTransport`)
- **Target Category:** `Travel`
- **Target Column:** `travel_km`
- **Mechanism:** Multiplicative Travel Substitution
- **Mathematical Formula:**
  $$X_{\text{new}} = X_{\text{old}} \times (1.0 - r_{\text{ev}})$$
- **Parameters:**
  - `ev_adoption_fraction` ($r_{\text{ev}}$): Fraction of commuter distance converted to zero-emission electric shuttles ($0.0 \le r_{\text{ev}} \le 1.0$). Default: `0.20` (20%).
  - `capital_cost`: ₹18,00,000 INR (electric mini-shuttle purchase and Level 2 AC chargers).
  - `lifespan_years`: 8 years (lithium iron phosphate traction battery life).
- **Physical Rationale:** Transitioning commuter travel distance from internal combustion engine (ICE) cars and motorcycles to zero-emission electric shuttles directly removes passenger travel distance from the Scope 3 reporting boundary.

---

## 5. Interaction Considerations (Sequential Stacking)

When evaluating combinations of interventions that target the same activity category (e.g., LED Lighting, HVAC Optimization, and Rooftop Solar all targeting `electricity_kwh`), the order of application dictates physical correctness.

### 5.1 The Multiplicative-Then-Additive Stacking Rule
> [!IMPORTANT]
> **Physical Stacking Rule:**  
> Multiplicative demand-reduction interventions (priority 1: LED, HVAC) **MUST** be applied before Additive generation interventions (priority 2: Solar).

#### The Mathematical Proof:
Suppose a facility consumes $10,000\text{ kWh/month}$. An LED retrofit reduces demand by $20\%$ ($r_{\text{led}} = 0.20$), and a Solar PV array generates $5,000\text{ kWh/month}$ ($G_{\text{solar}} = 5000$).

- **Physical Reality (Multiplicative First):**
  1. LEDs reduce the gross building load:
     $$X_1 = 10,000 \times (1 - 0.20) = 8,000\text{ kWh}$$
  2. Solar generates 5,000 kWh behind the meter, subtracting from the remaining load:
     $$X_2 = \max(0, 8,000 - 5,000) = 3,000\text{ kWh}$$
  - **Net Grid Import:** $3,000\text{ kWh}$ (Total energy saved: $7,000\text{ kWh}$).

- **Incorrect Order (Additive First):**
  1. Solar subtracts 5,000 kWh:
     $$X_1 = 10,000 - 5,000 = 5,000\text{ kWh}$$
  2. LEDs then reduce the remaining load by 20%:
     $$X_2 = 5,000 \times (1 - 0.20) = 4,000\text{ kWh}$$
  - **Flaw:** LEDs only saved $1,000\text{ kWh}$ instead of their true physical capability of $2,000\text{ kWh}$, artificially diluting the impact of energy efficiency!

### 5.2 Implementation of Interaction Metadata
To enforce this physical principle automatically, every intervention subclass defines a `priority` attribute:
- `priority = 1`: Multiplicative interventions (`LEDLightingRetrofit`, `ACOptimization`, `WasteSegregation`, `LowCarbonTransport`).
- `priority = 2`: Additive interventions (`RooftopSolarInstallation`).

The utility function `sort_interventions(interventions)` sorts any list of interventions by priority in ascending order, guaranteeing that efficiency measures are calculated on gross demand before clean generation is subtracted.

---

## 6. Input Validation & Physical Boundary Contracts

Every intervention implements strict programmatic guards in `validate()` and `.apply()`:

1. **Non-Negative Capital Cost:** `capital_cost >= 0.0`.
2. **Positive Lifespan:** `lifespan_years >= 1` (integer).
3. **Bounded Reduction Fractions:** `0.0 <= reduction_fraction <= 1.0` (finite, no `NaN` or `Inf`).
4. **Non-Negative Additive Generation:** `monthly_generation_kwh >= 0.0` (finite).
5. **Calendar Month Integrity:** `active_months` must contain valid calendar integers ($1 \le m \le 12$).
6. **Physical Non-Negativity:** Output activity values are strictly bounded by $\max(0.0, \dots)$, guaranteeing zero negative activity numbers.
7. **Collateral Column Invariance:** Applying an intervention to `electricity_kwh` guarantees that `travel_km`, `waste_kg`, `procurement_inr`, and headcounts remain strictly unmodified.

---

## 7. Known Limitations & Domain Modeling Boundaries

1. **Linearity Assumption:** Multiplicative reductions assume a linear percentage drop across all hours of operation. In reality, building baseloads (server rooms, emergency lighting) create non-linear floors below which energy consumption cannot drop.
2. **Stranded Solar Generation:** If aggressive energy efficiency reduces electricity consumption below monthly solar output, the excess is clipped to zero ($\max(0.0, \dots)$). In real campuses with net-metering contracts, excess generation would be exported to the grid with financial credits.
3. **Static Fleet Replacement:** Travel reductions assume that electric shuttles operate entirely on on-site renewable energy. If EV shuttles are charged from the standard fossil-dominated grid, electricity consumption would experience a small secondary rebound load.

---

## 8. Explicit Out of Scope Declarations

In strict adherence to Phase 13 boundary constraints:
- **NO scenario evaluation over Holt-Winters forecasts** (Reserved for Phase 14).
- **NO financial payback, Net Present Value (NPV), or Marginal Abatement Cost (MAC) curves** (Reserved for Phase 14).
- **NO Knapsack or Integer Programming budget optimizers** (Reserved for Phase 15).
- **NO Streamlit dashboard UI widgets or sliders** (Reserved for Phase 14/15).
