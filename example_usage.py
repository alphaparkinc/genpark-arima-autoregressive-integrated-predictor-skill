from client import ARIMAPredictor

series = [10, 12, 14, 16, 18, 20, 22, 24]
res = ARIMAPredictor.fit_predict(series, d=1, horizon=3)

print("ARIMA Model Results:")
print(f"AR(1) Parameter Phi: {res['phi']:.4f}")
print(f"Forecasts: {[round(f, 2) for f in res['forecasts']]}")
