# Project Pitch: Campus Carbon Forecasting and Decarbonization Scenario Analytics

## The Pitch

**"Good Morning/Afternoon, Examiners.**

I am thrilled to present our project: **Campus Carbon Forecasting and Decarbonization Scenario Analytics**.

In recent years, educational institutions have increasingly committed to achieving Net Zero emissions, yet many struggle with the transition from setting targets to actionable, data-driven planning. Our project bridges this gap by delivering a comprehensive academic prototype designed specifically for campus environments.

This platform is more than just a carbon calculator. It is a full-stack, end-to-end data science solution that takes raw, simulated campus activity data across critical domains—such as Electricity, Travel, Waste, and Procurement—and transforms it into strategic foresight.

Here is how our pipeline works:
1. **Data Ingestion & Validation:** We begin by cleaning and rigorously validating synthetic, realistic campus data to ensure high-fidelity inputs.
2. **Carbon Accounting:** Using documented, standard emission factors, we accurately convert this activity data into carbon footprints.
3. **Advanced Forecasting:** We don’t just look at the past. We apply statistical time-series models, such as the Holt-Winters method, alongside Monte Carlo simulations, to forecast future emissions and quantify uncertainty. We rigorously evaluate these models against naive baselines using robust metrics like MAE and RMSE.
4. **Scenario Optimization:** Finally, the true power of our tool lies in scenario analytics. We simulate various decarbonization interventions—like adopting renewable energy or upgrading facilities. Our optimization engine then selects the best portfolio of interventions to maximize emission reductions under strict, user-defined budget constraints.

All of this complex analytics is made accessible through an interactive, responsive Streamlit dashboard.

In building this, we focused on robust software engineering practices: modular architecture, comprehensive unit testing, and clear separation of concerns. This project demonstrates our ability to not only build predictive models, but to operationalize them into a decision-support tool that addresses a real-world, global challenge.

Thank you, and I look forward to your questions."

---

## VIVA Q&A (Expected Questions & Suggested Answers)

### 1. Data & Preprocessing
**Q: How did you source and validate your data?**
**A:** Because live campus data is often restricted, we generated highly realistic synthetic data. We strictly distinguished between our simulated activity data and the externally sourced, real-world emission factors we used for conversion. For validation, we built an automated data validation pipeline that checks for missing values, data type consistencies, and outliers before any calculation occurs, ensuring pipeline robustness.

### 2. Carbon Accounting
**Q: What categories of emissions does your model cover, and how are they calculated?**
**A:** We focus on four primary categories: Electricity, Travel, Waste, and Procurement. The calculation is fundamentally `Activity Data × Emission Factor`. We used documented, standard emission factors to convert raw activities (e.g., kWh of electricity or miles traveled) into CO2 equivalent (CO2e) emissions.

### 3. Forecasting & Modeling
**Q: Why did you choose the Holt-Winters model for forecasting, and how did you evaluate it?**
**A:** We chose Holt-Winters (Triple Exponential Smoothing) because campus emissions—particularly electricity and travel—often exhibit strong seasonality (e.g., lower emissions during summer breaks) and trends over time. We compared its performance against a naive baseline model. Evaluation was conducted using standard regression metrics, primarily Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), to quantify prediction accuracy.

### 4. Uncertainty Estimation
**Q: Forecasting the future is never 100% accurate. How do you handle uncertainty?**
**A:** We integrated Monte Carlo simulations. Instead of providing a single deterministic point forecast, we run thousands of simulated scenarios sampling from historical variance distributions. This allows us to provide confidence intervals (e.g., 95% certainty that emissions will fall within a specific range), offering decision-makers a realistic view of risk.

### 5. Optimization & Scenarios
**Q: Explain how your scenario optimization works under the hood.**
**A:** The optimization module aims to maximize emission reductions while adhering to a strict financial budget. It essentially solves a variation of the Knapsack Problem. We define various interventions (e.g., installing solar panels, transitioning to EV fleets), each with an associated cost and an expected emission reduction. The algorithm evaluates different portfolios of these interventions to find the optimal combination that yields the highest reduction without exceeding the user-defined budget constraint.

### 6. Software Engineering & Architecture
**Q: What software engineering best practices did you follow?**
**A:** We adhered to the principle of favoring simple, understandable implementations. The codebase is heavily modularized into distinct packages (`forecasting`, `scenarios`, `uncertainty`, `dashboard`). We implemented structured logging, comprehensive unit tests, and robust failure-mode testing to ensure reliability. We deliberately avoided unnecessary complexities like microservices or Kubernetes, focusing instead on a highly functional and maintainable monolithic academic prototype.

### 7. Limitations & Future Scope
**Q: What are the limitations of your current prototype?**
**A:** Currently, the system relies on synthetic activity data. While the emission factors are real, the consumption patterns are simulated. Furthermore, our budget optimization assumes fixed costs and static reduction rates for interventions over time. In a real-world scenario, economies of scale and changing technology costs would require a dynamic cost model. Integrating real-time IoT sensor data would be the logical next step for a production-grade system.
