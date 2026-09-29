# Raw Campus Activity Data

## Simulation Notice & Methodology

All datasets in this folder (`data/raw/`) are **strictly synthetic and simulated** for academic project BDS-36 (*Campus Carbon Forecasting and Decarbonization Scenario Analytics*, T.Y. B.Sc. Data Science 2026-27).

> [!IMPORTANT]
> **No Actual Meter or Financial Data:**
> The numbers recorded in `campus_activity.csv` are generated via parameterized stochastic mathematical simulation (`src/data_generator.py`). They do not represent real-world utility meter readings, actual student rosters, or physical expense accounts from any institution.
> 
> **Separation from Emission Factors:**
> In compliance with project guidelines, **no external emission factors are embedded or invented here**. Emission factors must be ingested separately from recognized environmental agencies (e.g. CEA, EPA, DEFRA).

## Dataset Specification: `campus_activity.csv`

- **Temporal Coverage:** 2023-01-01 to 2025-12-01 (36 continuous monthly billing cycles, ~3 years)
- **Granularity:** Monthly by building facility
- **Facilities Simulated:** 5 realistic campus archetypes:
  1. `Science & Engineering Complex` (Research labs, computing clusters, high baseline electricity, active during vacations)
  2. `Academic Block - Arts & Humanities` (Classrooms, lecture halls, strong vacation sensitivity)
  3. `Central Library & Student Hub` (Continuous study halls, HVAC, long study hours)
  4. `Administrative Headquarters` (Office staff, administrative operations, steady year-round activity, March fiscal spike)
  5. `Hostel & Dining Complex` (Residential, student dorms, high dining/food waste)

## Schema

| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | String (YYYY-MM-DD) | Date | Month start date (`YYYY-MM-01`) |
| `building` | String | Categorical | Building archetype name |
| `electricity_kwh` | Float | kWh | Monthly building electricity consumption |
| `travel_km` | Float | km | Monthly commuting distance attributed to occupants |
| `waste_kg` | Float | kg | Monthly solid waste generation |
| `procurement_inr` | Float | INR (₹) | Monthly operational and capital procurement spending |
| `student_count` | Integer | Count | Active student headcount for that month |
| `staff_count` | Integer | Count | Active staff headcount for that month |

## Reproducibility

This dataset was generated using:
```bash
python src/data_generator.py --start-date 2023-01-01 --end-date 2025-12-01 --num-buildings 5 --seed 42 --output data/raw/campus_activity.csv
```
The random seed `42` guarantees exact reproducibility.
