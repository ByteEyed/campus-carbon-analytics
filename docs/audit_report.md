# Carbon Accounting Implementation — Full Audit Report

> [!NOTE]
> Audit scope: [`src/carbon_accounting.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py), [`src/data_validation.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/data_validation.py), [`src/data_generator.py`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/data_generator.py), all tests, and data files. All 44 existing tests pass.

---

## 1. Calculation Correctness

### 🔴 CRITICAL — `total_emissions_mt` uses ÷1000 (metric tonnes) but summary label says "MTCO2e"

| Location | Detail |
|---|---|
| [carbon_accounting.py L345-347](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L345-L347) | `total_emissions_mt = total_emissions_kg / 1000.0` |
| [carbon_accounting.py L516](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L516) | Summary prints `"MTCO2e"` |

**Problem:** "MT" in GHG reporting universally means **metric tonnes** (1 MT = 1000 kg). The division by 1000 is arithmetically correct for that conversion. However, the column name `total_emissions_mt` and the label `"MTCO2e"` are ambiguous — in North American carbon accounting, "MT" can also stand for "million tonnes." The code is *correct* for metric-tonnes but the naming invites misinterpretation. This is not a calculation bug per se, but the doc-string (L305) says `"total_emissions_mt (metric tonnes)"` which resolves the ambiguity properly. **Downgrade to Minor.**

**Reclassified: 🟡 MINOR** — Label ambiguity only; arithmetic is correct.

---

### 🔴 CRITICAL — `emissions_per_student_kg` semantics are incoherent at the row level vs. aggregate level

| Location | Detail |
|---|---|
| [carbon_accounting.py L349-356](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L349-L356) | Row-level: `total_emissions_kg[row] / student_count[row]` |
| [carbon_accounting.py L457-478](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L457-L478) | Aggregate: `sum(total_emissions_kg) / sum(student_count)` |
| [carbon_accounting.py L383-393](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L383-L393) | Date aggregation: `sum(student_count)` then divides |

**Problem:** At the row level (one building, one month), `emissions_per_student_kg` means "kgCO2e per student *for this building this month*." At the aggregate level in [`calculate_emissions_per_student`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L457-L478), it divides total cumulative emissions by total cumulative student-months — giving "kgCO2e per student-month" (as the docstring correctly says on L462). But in [`aggregate_emissions_by_date`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L383-L393), `student_count` is **summed** across buildings for each date, then used as the denominator. This sum of headcounts across buildings counts students who use multiple buildings multiple times — a **double-counted denominator** that deflates the per-student metric.

**Severity: 🔴 CRITICAL** — The per-student metric in the date-aggregation is systematically under-reported because the denominator is inflated by counting the same students across buildings.

---

### 🟠 MAJOR — `calculate_emissions()` pure function is NOT used by `calculate_campus_emissions()`

| Location | Detail |
|---|---|
| [carbon_accounting.py L225-277](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L225-L277) | Pure function with validation |
| [carbon_accounting.py L325-336](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L325-L336) | DataFrame path uses raw `* ef_elec` |

**Problem:** The carefully validated `calculate_emissions()` function is defined but the actual DataFrame pipeline bypasses it entirely, using direct multiplication. This means the NaN/Inf/None guards in the pure function are inactive for real data. If a NaN slips through validation, it will silently propagate as `NaN * 0.716 = NaN` without raising.

**Severity: 🟠 MAJOR** — The code claims in the docstring (L317: "Vectorized computation using calculate_emissions constraint verification") that it uses the pure function's constraints, but it does not.

---

## 2. Unit Consistency

### 🟡 MINOR — No runtime unit-matching between activity columns and emission factor `unit` field

| Location | Detail |
|---|---|
| [carbon_accounting.py L310-313](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L310-L313) | Lookups by activity_type string only |
| [EmissionFactor.unit](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L66) | `unit` field stored but never checked |

**Problem:** The `EmissionFactor` dataclass stores a `unit` field (e.g. `"kgCO2e/kWh"`) but the pipeline never validates that the unit of the factor matches the unit of the activity column. If someone registers a factor with `unit="kgCO2e/MWh"` for `activity_type="electricity_kwh"`, the multiplication would silently produce results that are off by 1000×.

**Severity: 🟡 MINOR** — The current defaults are consistent, but there is no guard against misconfiguration.

---

### 🟢 OK — kgCO2e is used consistently throughout

All intermediate and output columns use kgCO2e. The metric-tonne conversion is a simple ÷1000. No mixed-unit bugs found in the core calculation chain.

---

## 3. Handling of Missing Factors

### 🟠 MAJOR — `run_carbon_accounting()` silently falls back to defaults when factors file is absent

| Location | Detail |
|---|---|
| [carbon_accounting.py L560-564](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L560-L564) | `if f_path.exists(): ... else: logger.warning(...)` |

**Problem:** If the user-specified emission factors file doesn't exist, the code logs a warning and falls back to hardcoded defaults. For a carbon accounting system, silently using different factors than the user intended is dangerous — results will appear correct but use wrong factors. This should at minimum be configurable (strict mode that raises, vs. lenient mode that warns).

**Severity: 🟠 MAJOR** — Silent factor substitution can produce materially incorrect reports without the user realizing.

---

### 🟢 OK — Missing factor *lookup* raises `KeyError`

The [`EmissionFactorRegistry.get_factor()`](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L155-L164) correctly raises a `KeyError` with a helpful message when a key is not found.

---

## 4. Validation

### 🟠 MAJOR — `calculate_campus_emissions()` does not validate NaN/Inf in activity columns

| Location | Detail |
|---|---|
| [carbon_accounting.py L319-323](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L319-L323) | Only checks `< 0`, not NaN/Inf |

**Problem:** The function checks for negative values but does NOT check for NaN or Inf in the activity columns. If a NaN value reaches this function (e.g., from a CSV loaded without going through `data_validation.py`), it will silently propagate through all emission calculations and aggregations. The pure function `calculate_emissions()` *does* check for NaN/Inf, but as noted above, it's never called.

**Severity: 🟠 MAJOR** — NaN propagation would corrupt the entire emissions dataset silently.

---

### 🟡 MINOR — `EmissionFactor` allows `factor=0.0`

| Location | Detail |
|---|---|
| [carbon_accounting.py L91-92](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L91-L92) | Only rejects `< 0` |

**Problem:** A zero emission factor is physically meaningless for most categories (it implies zero emissions regardless of activity). While there are edge cases where it could be valid (e.g., 100% renewable electricity), accepting `0.0` without any warning makes accidental zero-factor registration undetectable.

**Severity: 🟡 MINOR** — Edge case; zero-factors are uncommon but not impossible.

---

### 🟡 MINOR — `calculate_campus_emissions()` does not check for required columns before use

| Location | Detail |
|---|---|
| [carbon_accounting.py L325-336](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L325-L336) | Direct `df["electricity_kwh"]` access |

**Problem:** If a DataFrame is passed missing any of the expected activity columns (e.g., `electricity_kwh`), it will crash with an unhelpful `KeyError` rather than a clear validation message.

**Severity: 🟡 MINOR** — Defensive check missing but data_validation should catch this upstream.

---

## 5. Aggregation Logic

### 🔴 CRITICAL — `aggregate_emissions_by_date()` sums `student_count` across buildings, then divides

Already described in Section 1. The `student_count` column gets `"sum"` aggregation across buildings for each date. If the campus has 5 buildings, each with ~1000 students, the sum shows ~5000 "students" for that date — but many of the same real students use multiple buildings. The per-student metric is therefore **systematically diluted**.

**Example:** If the real campus has 3000 unique students but the building-level counts sum to 5200 student-rows, emissions_per_student is reported as `total / 5200` instead of `total / 3000`.

**Severity: 🔴 CRITICAL** — Produces misleading KPIs that understate per-student carbon intensity.

---

### 🟡 MINOR — `aggregate_emissions_by_category` recomputes `grand_total` independently

| Location | Detail |
|---|---|
| [carbon_accounting.py L421](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L421) | `grand_total = elec_tot + travel_tot + waste_tot + proc_tot` |

**Problem:** This sums the four category totals instead of using the pre-computed `total_emissions_kg` column. Due to floating-point rounding at the row level (`.round(4)`) the two values can diverge slightly. This could cause the contribution percentages to not sum to exactly 100% (the test already allows `abs=0.1` tolerance).

**Severity: 🟡 MINOR** — Rounding inconsistency, not a logic error.

---

### 🟡 MINOR — Rounding cascade: row-level `.round(4)` feeds into aggregates that `.round(2)`

Rounding at each step introduces cumulative truncation errors. The row-level emissions are rounded to 4 decimal places, then summed, then the sums are rounded to 2. This is standard practice for financial reporting but could produce small discrepancies in audit reconciliation.

**Severity: 🟡 MINOR**

---

## 6. Possible Double Counting

### 🟢 OK — No cross-category double counting

Each emission category (electricity, travel, waste, procurement) maps to a distinct activity column. There is no overlap in the data model. The Scope 1/2/3 boundary is not explicitly modeled, but within the project's scope this is fine.

### 🟡 MINOR — Procurement EEIO factors may partially overlap with electricity/waste

| Location | Detail |
|---|---|
| [emission_factors.csv L5](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/emission_factors.csv#L5) | Procurement: `0.00042 kgCO2e/INR` via EEIO multiplier |

**Problem:** EEIO (Environmentally Extended Input-Output) multipliers for procurement spending include upstream embodied emissions, which may partially include electricity generation and waste processing emissions already counted in the Electricity and Waste categories. This is a well-known methodological challenge in Scope 3 accounting. The project should acknowledge this in documentation.

**Severity: 🟡 MINOR** — Methodological limitation, not a code bug. Should be documented.

---

## 7. Data Leakage / Accidental Assumptions

### 🟠 MAJOR — Emission factors are hardcoded in Python AND duplicated in CSV

| Location | Detail |
|---|---|
| [carbon_accounting.py L100-133](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/src/carbon_accounting.py#L100-L133) | `DEFAULT_EMISSION_FACTORS` list |
| [emission_factors.csv](file:///d:/user%20(sarthak)/projects/campus-carbon-analytics/data/emission_factors.csv) | Same values |

**Problem:** The emission factors exist in two authoritative locations. If a user updates the CSV but forgets the Python defaults (or vice versa), the system can silently use stale factors. The `run_carbon_accounting()` function tries the CSV first and falls back to Python defaults, creating a confusing precedence chain.

**Severity: 🟠 MAJOR** — Dual source of truth for critical accounting constants.

---

### 🟡 MINOR — `cleaned_activity.csv` and `campus_activity.csv` are identical

The raw and cleaned CSVs have identical content (181 data rows each), meaning the validation pipeline passed 100% of records. This is expected for generated data but means the validation pipeline has never been exercised on *actually dirty* production data.

**Severity: 🟡 MINOR** — Not a code bug, but a testing observation.

---

### 🟢 OK — No train/test data leakage

The data generator, validation, and accounting pipelines are cleanly separated. The generator does not embed emission factors. The accounting module does not embed activity assumptions.

---

## 8. Test Coverage

### Overall: 44 tests, all passing ✅

```
tests/test_carbon_accounting.py  — 15 tests
tests/test_data_validation.py   — 13 tests  
tests/test_data_generator.py    — 16 tests
```

### 🟠 MAJOR — No test verifies numeric precision of emission calculations against hand-computed expected values

| Gap | Detail |
|---|---|
| `test_calculate_campus_emissions` | Checks column existence and sum identity, but never checks that `15900.64 * 0.716 = 11384.8582` for a known row |

**Problem:** No test performs a **known-answer verification** against independently hand-calculated expected emissions for a specific data row. The tests verify structural properties (columns exist, sums match, percentages add to 100%) but never the actual numeric output against a hand-computed oracle.

**Severity: 🟠 MAJOR** — A factor could be applied to the wrong column and all structural tests would still pass.

---

### 🟠 MAJOR — No test for NaN/Inf propagation through `calculate_campus_emissions()`

No test verifies what happens when NaN or Inf values reach `calculate_campus_emissions()`. Given the missing validation (Section 4), this is a significant gap.

---

### 🟡 MINOR — Missing test coverage items

| Missing Test | Severity |
|---|---|
| Registry with duplicate category/activity_type keys (collision behavior) | Minor |
| `from_dataframe()` with extra columns, wrong dtypes, empty DataFrame | Minor |
| `aggregate_emissions_by_date()` with no `student_count` column | Minor |
| `CarbonAccountingSummary.summary()` string formatting | Minor |
| `calculate_campus_emissions()` with missing required columns (expected error message) | Minor |
| `run_carbon_accounting()` when factors file is missing (fallback path) | Minor |
| Edge: single-row DataFrame through entire pipeline | Minor |

---

## Summary Table

| # | Finding | Severity | Category |
|---|---|---|---|
| 1 | `emissions_per_student` in date aggregation double-counts students across buildings | 🔴 Critical | Aggregation |
| 2 | `calculate_emissions()` pure function is bypassed — NaN/Inf guards inactive | 🟠 Major | Calculation |
| 3 | `calculate_campus_emissions()` does not validate NaN/Inf in activity columns | 🟠 Major | Validation |
| 4 | Silent fallback to default factors when factors file is missing | 🟠 Major | Missing factors |
| 5 | Dual source of truth: factors hardcoded in Python AND in CSV | 🟠 Major | Data leakage |
| 6 | No known-answer test for emission calculation precision | 🟠 Major | Test coverage |
| 7 | No test for NaN/Inf propagation through DataFrame pipeline | 🟠 Major | Test coverage |
| 8 | No runtime unit-matching between factor `unit` and activity column | 🟡 Minor | Unit consistency |
| 9 | `EmissionFactor` allows `factor=0.0` without warning | 🟡 Minor | Validation |
| 10 | No column-existence check before accessing activity columns | 🟡 Minor | Validation |
| 11 | Procurement EEIO factor may overlap Scope 1/2 categories (undocumented) | 🟡 Minor | Double counting |
| 12 | `aggregate_emissions_by_category` recomputes grand_total independently | 🟡 Minor | Aggregation |
| 13 | Rounding cascade across pipeline stages | 🟡 Minor | Calculation |
| 14 | `MTCO2e` label ambiguity | 🟡 Minor | Unit consistency |
| 15 | Several missing edge-case tests | 🟡 Minor | Test coverage |

---

## Proposed Fixes

### Fix 1 — 🔴 Critical: Repair per-student aggregation in `aggregate_emissions_by_date()`

**Problem:** `student_count` is summed across buildings, inflating the denominator.

**Fix:** Do NOT aggregate `student_count` by sum. Instead, either:
- **(a)** Remove `emissions_per_student_kg` from the date aggregation entirely (compute it only at the summary level), or
- **(b)** Use the **campus-wide unique student count** as a separate input parameter, not derived from building-level sums.

Option (a) is simpler and more correct for this project:

```python
# In aggregate_emissions_by_date(): remove student_count from agg_dict
# Remove the per-student calculation from this function entirely
```

### Fix 2 — 🟠 Major: Add NaN/Inf validation to `calculate_campus_emissions()`

```python
# Before emission calculations, add:
for col in CATEGORY_TO_ACTIVITY_COL.values():
    if col in emissions_df.columns:
        if emissions_df[col].isna().any():
            raise ValueError(f"Activity column '{col}' contains NaN values")
        if np.isinf(emissions_df[col].astype(float)).any():
            raise ValueError(f"Activity column '{col}' contains Inf values")
```

### Fix 3 — 🟠 Major: Either use `calculate_emissions()` in the pipeline or unify validation

Two options:
- **(a)** Apply `calculate_emissions` via `.apply()` or vectorized wrapper — slower but honours the pure function contract
- **(b)** Inline the same validation checks into `calculate_campus_emissions()` — faster, keeps vectorized ops

Recommend **(b)** since performance matters for DataFrames, but explicitly document that the vectorized path includes equivalent guards.

### Fix 4 — 🟠 Major: Make factor-file fallback explicit

```python
def run_carbon_accounting(..., strict_factors: bool = False):
    if not f_path.exists():
        if strict_factors:
            raise FileNotFoundError(f"Emission factors file required but not found: {f_path}")
        logger.warning("Factors file not found at %s. Using default documented factors.", f_path)
        registry = EmissionFactorRegistry.default()
```

### Fix 5 — 🟠 Major: Single source of truth for emission factors

Remove `DEFAULT_EMISSION_FACTORS` from Python. Make `emission_factors.csv` the canonical source. The `EmissionFactorRegistry.default()` method should load from the CSV, with a fallback error if the file is missing. Alternatively, generate the CSV *from* the Python defaults at build time, making Python the single source.

### Fix 6 — 🟠 Major: Add known-answer precision tests

```python
def test_known_answer_emission_calculation():
    """Verify exact emission values against hand-computed results."""
    # Row: electricity_kwh=15900.64, factor=0.716
    # Expected: 15900.64 * 0.716 = 11384.85824 → round(4) = 11384.8582
    df = pd.DataFrame({
        "date": ["2023-01-01"],
        "building": ["Test"],
        "electricity_kwh": [15900.64],
        "travel_km": [38600.36],
        "waste_kg": [2664.63],
        "procurement_inr": [187000.13],
        "student_count": [1319],
        "staff_count": [54],
    })
    result = calculate_campus_emissions(df)
    assert result.loc[0, "electricity_emissions_kg"] == 11384.8582
    assert result.loc[0, "travel_emissions_kg"] == 5404.0504
    assert result.loc[0, "waste_emissions_kg"] == 1188.425  # 2664.63 * 0.446
    assert result.loc[0, "procurement_emissions_kg"] == 78.5401  # 187000.13 * 0.00042
```

### Fix 7 — 🟡 Minor: Add column-existence guard

```python
def calculate_campus_emissions(df, registry=None):
    required = list(CATEGORY_TO_ACTIVITY_COL.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Activity DataFrame missing required columns: {missing}")
```

### Fix 8 — 🟡 Minor: Document EEIO overlap risk

Add a note to `DATA_DICTIONARY.md` under the Procurement factor:
> ⚠️ EEIO-based procurement factors include upstream embodied emissions that may partially overlap with direct electricity and waste emissions already counted in Scope 1/2 categories. This is a known limitation of spend-based Scope 3 accounting.
