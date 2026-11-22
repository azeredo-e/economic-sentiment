# Forecast port: fidelity notes

The Python port of the thesis' R pipeline is only trusted once it reproduces Table 3 with the
original indices (M1). These are the known places where ports drift.

## 1. Boosting = componentwise linear L2 boosting
The thesis (Eqs. 10–11) uses componentwise linear L2 boosting: at each iteration, fit the
residuals against each single regressor, keep the best one, and update the fit by a shrunken
step (learning rate ν). This is a linear model, about 30 lines to write by hand.

Do **not** substitute `sklearn.ensemble.GradientBoostingRegressor` (regression trees). It is a
different, nonlinear model and its results would not be comparable to Table 3.
The original (`09_Get_Models.R`) calls `mboost::glmboost` with `boost_control(mstop = 300, nu = 0.1)`
and picks the optimal `mstop` by cross-validation (`cv_seq`). *TODO:* read the CV scheme
(fold construction, `mstop` grid) from the R file before implementing.

## 2. SARIMA benchmark
Every relative RMSE is divided by the SARIMA forecast, so a different benchmark changes every
number. R's `forecast::auto.arima` (stepwise search, BIC, Ljung-Box check) and
`pmdarima.auto_arima` do not search the same model set or use the same defaults.

Options, to be decided at M1 (not yet chosen):
1. **Run the benchmark through R** (`Rscript` + `forecast`), cache the forecasts, compare with the
   rest in Python. Exact benchmark; needs R and the `forecast` package (not installed on this
   machine yet, nor are `glmnet` / `sentometrics`).
2. **Re-implement the auto.arima search in Python** with statsmodels. Self-contained, but it must
   be validated against R output on the same series before use.

Whichever is picked, M1 must compare the benchmark's chosen orders and RMSEs with the thesis.

## 3. Other places to watch
- LASSO/ENET: `glmnet` and scikit-learn standardise and parametrise the penalty differently.
  Check λ grids and the CV scheme (time-series CV) against the R code.
- Rolling-window definition, lag construction (4 lags) and stationarity transforms.
