"""ARIMA Autoregressive Integrated Predictor Engine.
100% Python Standard Library.
"""

class ARIMAPredictor:
    """ARIMA(1, d, 0) differencing and Yule-Walker autoregressive forecaster."""
    @staticmethod
    def difference(series, d=1):
        res = list(series)
        for _ in range(d):
            res = [res[i+1] - res[i] for i in range(len(res) - 1)]
        return res

    @staticmethod
    def invert_difference(orig_series, diff_forecasts, d=1):
        forecasts = list(diff_forecasts)
        last_val = orig_series[-1]
        inv = []
        for df in forecasts:
            next_val = last_val + df
            inv.append(next_val)
            last_val = next_val
        return inv

    @classmethod
    def fit_predict(cls, series, d=1, horizon=2):
        diff_data = cls.difference(series, d=d)
        mean_diff = sum(diff_data) / len(diff_data)
        num = sum((diff_data[i] - mean_diff) * (diff_data[i+1] - mean_diff) for i in range(len(diff_data) - 1))
        den = sum((x - mean_diff)**2 for x in diff_data) or 1.0
        phi = num / den

        diff_forecasts = []
        cur_diff = diff_data[-1]
        for _ in range(horizon):
            cur_diff = mean_diff + phi * (cur_diff - mean_diff)
            diff_forecasts.append(cur_diff)

        forecasts = cls.invert_difference(series, diff_forecasts, d=d)
        return {"phi": phi, "mean_diff": mean_diff, "forecasts": forecasts}
