# Data Dictionary & Schemas

## 6. Dataset Schema (Activity Data)

The synthetic campus data will be structured in a tabular format (CSV) tracking individual activity events or aggregated monthly consumption.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `date` | Date (YYYY-MM-DD) | Date of the activity or start of the billing period. | `2024-01-01` |
| `category` | String | High-level category (Electricity, Travel, Waste, Procurement). | `Electricity` |
| `subcategory` | String | Specific activity type. | `Grid`, `Commute_Car`, `Landfill` |
| `quantity` | Float | The numerical amount of the activity. | `15000.5` |
| `unit` | String | Unit of measurement for the quantity. | `kWh`, `km`, `kg` |
| `is_simulated`| Boolean | Explicit flag denoting that the activity data is synthetic. | `True` |

## 7. Emission-Factor & Intervention Schemas

### Emission Factors
Emission factors must be clearly sourced from external documentation (e.g., EPA, DEFRA).

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `category` | String | Corresponding activity category. | `Electricity` |
| `subcategory` | String | Corresponding activity subcategory. | `Grid` |
| `emission_factor`| Float | Conversion factor to CO2e. | `0.453` |
| `ef_unit` | String | Unit of the emission factor. | `kgCO2e/kWh` |
| `source` | String | Documented source of the factor. | `EPA eGRID 2023` |
| `year_published` | Integer | Year the factor was published. | `2023` |

### Decarbonization Interventions
Used by the scenario optimization engine to select portfolios.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `intervention_id` | String | Unique identifier for the project. | `INT-001` |
| `description` | String | Human-readable description. | `Rooftop Solar Phase 1` |
| `category` | String | Affected emission category. | `Electricity` |
| `capital_cost` | Float | Initial cost to implement the project ($). | `250000` |
| `annual_co2_reduction`| Float | Estimated reduction in kgCO2e per year. | `75000` |
| `lifespan_years` | Integer | Expected lifespan of the intervention. | `20` |
