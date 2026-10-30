import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression

df = pd.read_csv("txhousing.csv")
df = df.sort_values(by=["city", "year", "month"]).reset_index(drop=True)
df["next_month_target"] = df.groupby("city")["median"].shift(-1)
df["month_number"] = df["year"] * 12 + df["month"]
df["next_month_number"] = df.groupby("city")["month_number"].shift(-1)
df["month_gap"] = df["next_month_number"] - df["month_number"]
print(df.head(10))
df["median_lag1"] = df.groupby("city")["median"].shift(1)
df["median_lag2"] = df.groupby("city")["median"].shift(2)
df["lag1_month_number"] = df.groupby("city")["month_number"].shift(1)
df["lag2_month_number"] = df.groupby("city")["month_number"].shift(2)
df["lag1_gap"] = df["month_number"] - df["lag1_month_number"]
df["lag2_gap"] = df["month_number"] - df["lag2_month_number"]
print(df[["city", "year", "month", "median", "median_lag1", "median_lag2", "lag1_gap", "lag2_gap"]].head(10))
df["median_ma6"] = df.groupby("city")["median"].transform(lambda prices: prices.rolling(window=6, min_periods=6).mean())
df["ma6_start_month"] = df.groupby("city")["month_number"].shift(5)
df["ma6_gap"] = df["month_number"] - df["ma6_start_month"]
valid_ma6_dates = df["ma6_gap"] == 5
valid_ma6_values = df["median_ma6"].notna()
valid_ma6 = valid_ma6_dates & valid_ma6_values
print(df[["city", "year", "month", "median", "median_lag1", "median_lag2", "lag1_gap", "lag2_gap", "median_ma6"]].head(10))
print(valid_ma6.sum())
valid_gap = df["month_gap"] == 1
valid_target = df["next_month_target"].notna()
valid_median = df["median"].notna()
valid_lags = (df["lag1_gap"] == 1) & (df["lag2_gap"] == 2)
valid_lag1 = df["median_lag1"].notna()
valid_lag2 = df["median_lag2"].notna()
valid_four_mask = valid_ma6 & valid_gap & valid_target & valid_median & valid_lags & valid_lag1 & valid_lag2
four_df = df[valid_four_mask].copy()
print(f"Final data set shape: {four_df.shape}")
print(f"Final data set rows: {len(four_df)}")
x = four_df[["median", "median_lag1", "median_lag2", "median_ma6"]]
y = four_df["next_month_target"]
print(f"X shape: {x.shape}")
print(f"Y shape: {y.shape}")
print(f"Missing X values: {x.isna().sum()}")
print(f"Missing Y values: {y.isna().sum()}")
test_start = 2014 * 12 + 1
train_mask = four_df["next_month_number"] < test_start
test_mask = four_df["next_month_number"] >= test_start
x_train = x[train_mask]
y_train = y[train_mask]
x_test = x[test_mask]
y_test = y[test_mask]
print(f"Training X: {x_train.shape}")
print(f"Training Y: {y_train.shape}")
print(f"Test X: {x_test.shape}")
print(f"Test Y: {y_test.shape}")

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
actual = y_test.to_numpy()
print(f"Prediction: {predictions.shape}")
print(f"Actual: {actual.shape}")

errors = predictions - actual
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
actual_mean = np.mean(actual)
ss_res = np.sum((errors) ** 2)
ss_tot = np.sum((actual - actual_mean) **2)
r2 = 1 - (ss_res/ss_tot)
print(f"Test MAE: {mae}")
print(f"Test RMSE: {rmse}")
print(f"Test R2: {r2}")

joblib.dump(model, "house_price_model.joblib")
loaded_model = joblib.load("house_price_model.joblib")
loaded_predictions = loaded_model.predict(x_test)
same_predictions = np.allclose(predictions, loaded_predictions)
print(f"Model loaded successfully: {same_predictions}")