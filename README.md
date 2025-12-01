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
## Supported Scikit-Learn-Style Regressors

You can use any scikit-learn-compatible regressor with this FlowMatching-BDT extension.

 1️⃣ Linear Models

- LinearRegression – Ordinary least squares regression
- Ridge – L2-regularized linear regression
- Lasso – L1-regularized linear regression
- ElasticNet – Combination of L1 and L2 regularization
- BayesianRidge – Bayesian regression with L2 priors
- SGDRegressor – Linear model trained via stochastic gradient descent
- HuberRegressor – Robust to outliers
- PoissonRegressor – Generalized linear model for count data
- TweedieRegressor – Flexible GLM (Poisson, Gamma, Gaussian)

 2️⃣ Tree-Based Models

- DecisionTreeRegressor – Single decision tree
- RandomForestRegressor – Ensemble of trees using bagging
- ExtraTreesRegressor – Extremely randomized trees
- GradientBoostingRegressor – Gradient boosting of decision trees
- HistGradientBoostingRegressor – Fast histogram-based gradient boosting

 3️⃣ Neighbors-Based Models

- KNeighborsRegressor – Predicts using average of K nearest neighbors
- RadiusNeighborsRegressor – Uses neighbors within a radius

 4️⃣ Other Models

- SVR – Support Vector Regression (linear or kernel-based)
- MLPRegressor – Multi-layer perceptron (neural network)
- GaussianProcessRegressor – Non-parametric regression using Gaussian processes
- DecisionTreeRegressor / BaggingRegressor – Ensemble wrappers for custom trees

Note: Any regressor that implements the scikit-learn interface (fit(X, y) + predict(X)) can be used.
