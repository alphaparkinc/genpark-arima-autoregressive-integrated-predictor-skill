# genpark-arima-autoregressive-integrated-predictor-skill

Agent Skill implementing **ARIMA(1, d, 0) Time Series Modeling** with discrete difference transformations, autocorrelation coefficient estimation, and recursive forecasting.

## Architectural Overview
```mermaid
flowchart TD
    Series["Raw Non-Stationary Series"] --> Diff["Order-d Differencing: Delta^d Y_t"]
    Diff --> AutoCorr["Estimate Lag-1 Autocorrelation Phi"]
    AutoCorr --> AR["AR(1) Forecast on Stationary Increments"]
    AR --> Invert["Cumulative Inverse Differencing"]
    Invert --> Pred["Final Absolute Forecasts"]
```
