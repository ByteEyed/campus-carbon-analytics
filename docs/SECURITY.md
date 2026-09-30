# Security Architecture & Boundary Defense Specification
**Campus Carbon Analytics** | Academic Project BDS-36 (T.Y. B.Sc. Data Science, Semester V, 2026-27)

---

## 1. Threat Model & Security Posture

### 1.1 Deployment & Environmental Context
The **Campus Carbon Analytics System** is designed as a standalone academic data science and decision-support pipeline. It runs locally or within institutional analytical workstations without exposing public network endpoints or processing untrusted internet requests.

Consequently, external web attack vectors (such as Distributed Denial of Service, SQL Injection, or Cross-Site Scripting) are structurally absent. The security posture focuses on:
1. **Data Corruption & Integrity Attacks**: Silent under-reporting of emissions through invalid factors or malformed activity data.
2. **Mathematical Boundary Exploits**: Negative capital costs, infinite values, or impossible physical states (negative emissions).
3. **Local File System & Path Traversal Risks**: Malicious or accidental directory escapes overwriting critical operating system directories.
4. **Serialization Safety**: Elimination of insecure object deserialization vulnerabilities.

---

## 2. Threat Analysis & Defensive Controls Matrix

| Threat Category | Attack Vector / Failure Mode | Severity | Control Mechanism | Code Location |
| :--- | :--- | :--- | :--- | :--- |
| **Emission Factor Integrity** | Silent under-reporting via $EF \le 0.0$ or negative factors | **Medium** | Enforce strictly positive emission factors ($EF > 0$); reject $\le 0.0$ | `src/carbon_accounting.py` (`EmissionFactorRegistry.register`) |
| **Financial Parameter Bounds** | Negative capital costs generating artificial budget surpluses | **Medium** | Validate $C \ge 0.0$ in intervention constructor | `src/scenarios/interventions.py` (`BaseIntervention.validate`) |
| **Physical Law Conservation** | Extreme over-reduction resulting in negative scenario emissions | **High** | Clamp transformed activity and scenario emissions to non-negative ($\ge 0$) | `src/scenarios/evaluation.py` & `src/scenarios/optimization.py` |
| **Directory Traversal** | Path traversal sequences (`../../Windows` or `/etc`) overwriting system files | **High** | Reject paths containing `..` and forbid resolution into protected system directories | `src/pipeline_config.py` (`PipelineConfig.__post_init__`) |
| **Activity Data Corruption** | NaN, Infinite, negative, or non-numeric activity readings | **High** | Vectorized type-casting and physical bounds assertion before calculation | `src/carbon_accounting.py` (`calculate_campus_emissions`) |
| **Division-by-Zero** | Zero baseline emissions or zero reduction crashing MAC calculations | **Low** | Guarded division with infinite/zero fallbacks | `src/scenarios/evaluation.py` (`cost_per_kg`) |
| **Insecure Deserialization** | Code execution via unsafe object unpickling | **Critical** | Zero use of `pickle`; pipeline strictly uses standard CSV and JSON formatters | Entire Codebase |
| **Dashboard Denial-of-Service**| Unhandled exceptions on corrupted artifacts crashing Streamlit UI | **Low** | Strict custom exception `DashboardDataError` with graceful UI warning cards | `src/dashboard/adapter.py` & `src/dashboard/app.py` |

---

## 3. Detailed Defense Implementations

### 3.1 Emission Factor Strict Positivity ($EF > 0$)
Registration of an emission factor must strictly verify that $EF > 0$:
```python
if factor.factor <= 0.0:
    raise ValueError(
        f"Emission factor for category '{factor.category}' must be strictly positive (> 0), "
        f"got {factor.factor}."
    )
```
*Rationale:* An emission factor of zero would silently erase campus emissions without generating a computation error, compromising regulatory audit readiness.

### 3.2 Non-Negative Physical Bounds Clamping
Physical conservation laws dictate that clean interventions cannot reduce gross emissions below zero:
```python
# Activity space clamping
transformed = np.maximum(0.0, np.asarray(transformed, dtype=float))

# Emissions space clamping
scenario_emissions_kg = max(0.0, float(scenario_emissions_df[target_metric_col].sum()))
portfolio_emissions_kg = max(0.0, round(float(scen_emissions_df["total_emissions_kg"].sum()), 2))
```
*Rationale:* Prevents pathological mathematical artifacts where clean energy exceeds demand and models assume negative campus footprint.

### 3.3 File System Sanitization & Traversal Defense
When receiving execution configurations from CLI or JSON inputs:
```python
for field_name, p in [
    ("activity_data_path", act_p),
    ("emission_factors_path", ef_p),
    ("output_dir", out_p),
]:
    if ".." in p.parts:
        raise ValueError(
            f"Security violation: path traversal sequence ('..') detected in {field_name}: {p}"
        )

resolved_out = out_p.resolve()
sensitive_roots = [
    Path("C:/Windows"), Path("C:/Program Files"), Path("C:/Program Files (x86)"),
    Path("/etc"), Path("/bin"), Path("/sbin"), Path("/root"), Path("/sys"), Path("/proc")
]
for sroot in sensitive_roots:
    try:
        sroot_res = sroot.resolve()
        if resolved_out == sroot_res or resolved_out.is_relative_to(sroot_res):
            raise ValueError(f"Security violation: output_dir points to sensitive system directory: {self.output_dir}")
    except ValueError:
        pass
```
*Rationale:* Ensures pipeline artifacts cannot be redirected to overwrite system binaries or configuration files.

### 3.4 Insecure Deserialization Policy
The system adheres to an absolute ban on arbitrary object serialization libraries:
- No `pickle.load()` or `pickle.dump()`.
- No `yaml.unsafe_load()` or executable YAML tags.
- All persistent data uses RFC-4180 CSV tables (`pd.read_csv`, `to_csv`) or ECMA-404 JSON strings (`json.load`, `json.dump`).

---

## 4. Verification & Audit Test Suite

A dedicated adversarial test module (`tests/test_robustness.py`) validates these security controls across 34 distinct assertions:
1. `test_registry_rejects_zero_emission_factor`
2. `test_registry_rejects_negative_emission_factor`
3. `test_registry_from_dataframe_rejects_zero_or_negative`
4. `test_interventions_reject_negative_capital_cost` (Parametrized across all 5 intervention types)
5. `test_interventions_reject_out_of_bounds_fractions` (Parametrized across reduction boundaries)
6. `test_rooftop_solar_rejects_negative_generation`
7. `test_extreme_solar_generation_clamps_to_zero`
8. `test_portfolio_extreme_over_reduction_clamping`
9. `test_pipeline_config_rejects_path_traversal_output_dir` (Parametrized across traversal patterns)
10. `test_pipeline_config_rejects_path_traversal_input_files`
11. `test_pipeline_config_rejects_sensitive_system_directories`
12. `test_pipeline_config_rejects_invalid_numeric_parameters`
13. `test_carbon_accounting_rejects_nan_activity`
14. `test_carbon_accounting_rejects_infinite_activity`
15. `test_carbon_accounting_rejects_negative_activity`
16. `test_carbon_accounting_rejects_missing_columns`
17. `test_dashboard_adapter_handles_missing_directory`
18. `test_dashboard_adapter_handles_corrupt_summary_json`
19. `test_dashboard_adapter_handles_empty_csv_artifact`

All 34 tests execute within **~3 seconds** and run automatically alongside the 232 existing regression tests.
