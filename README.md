# FlowMatching-Tabular

An extension of the original [FlowMatching-BDT](https://github.com/radiradev/flowmatching-bdt) library, allowing the use of **any scikit-learn-compatible regressor** in the flow-matching framework for generative modeling. This enhancement improves flexibility and experimentation for conditional or unconditional synthetic data generation.

---

## Features

- Replace the default `XGBRegressor` with any compatible regressor, including:
  - Linear models (LinearRegression, Ridge, Lasso)
  - Tree-based models (RandomForest, ExtraTrees, GradientBoosting)
  - k-Nearest Neighbors
  - Any other sklearn-style regressor
- Maintain original FlowMatching-BDT interface:
  - `.fit(X, conditions=None)`  
  - `.predict(num_samples, conditions=None)`
- Lightweight and memory-friendly options for faster experimentation
- Supports conditional data generation

---

## Installation

Clone the repository:

```bash
git clone git@github.com: FloAI/flowmatching_tabular.git
cd flowmatching_tabular
