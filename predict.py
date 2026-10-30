import joblib
import pandas as pd
import numpy as np

model = joblib.load("house_price_model.joblib")
prices = [110000, 112000, 115000, 118000, 121000, 125000]

if len(prices) != 6:
    raise ValueError("Exactly six monthly prices are required!")

prices  = np.asarray(prices, dtype=float)

if not np.isfinite(prices).all():
    raise ValueError("All values must be finite number!")

if (prices <= 0).any():
    raise ValueError("Prices must be positive!")

current_median = prices[-1]
lag1 = prices[-2]
lag2 = prices[-3]
ma6 = np.mean(prices)
new_data = pd.DataFrame([{"median": current_median, "median_lag1": lag1, "median_lag2": lag2, "median_ma6": ma6}])
prediction = model.predict(new_data)
print(prediction)
