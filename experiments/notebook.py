import pandas as pd

df = pd.read_csv("txhousing.csv")

df.head(10)

df.shape

df.columns

df.dtypes

df.describe()

na = df.isna().sum()

na[na > 0]

total_rows = df.shape[0]

missing_percentage = (na / total_rows) * 100

print(missing_percentage.round(2))

inventory = df["inventory"]
missing_inventory = inventory.isna()
print(df[missing_inventory].head(10))

median = df["median"]
missing_median = median.isna()
print(df[missing_median].head(10))

missing_both = missing_inventory & missing_median
print(missing_both.sum())

df = df.sort_values(by=["city", "year", "month"]).reset_index(drop=True)
df["next_month_target"] = df.groupby("city")["median"].shift(-1)
df[["city", "year", "month", "median", "next_month_target"]].head(10)
df[df["city"] == "Abilene"].head()

df["month_number"] = df["year"] * 12 + df["month"]
df["next_month_number"] = df.groupby("city")["month_number"].shift(-1)
df["month_gap"] = df["next_month_number"] - df["month_number"]
df.loc[df["city"] == "Abilene", ["month_number", "next_month_number", "month_gap"]].head(10)

df["month_gap"].value_counts(dropna=False)
mask = df["month_gap"] > 1
mask.sum()
df[mask].head(10)

valid_gap = df["month_gap"] == 1
valid_target = df["next_month_target"].notna()
valid_median = df["median"].notna()
valid_mask = valid_gap & valid_target & valid_median
valid_df = df[valid_mask].copy()
valid_mask.sum()
valid_df["next_month_target"].isna().sum()
valid_df.head(100)

X = valid_df[["median"]]
y = valid_df["next_month_target"]
print(X.shape)
print(y.shape)
print(X.head())
print(y.head())

validation_start = 2012 * 12 + 1
test_start = 2014 * 12 + 1

train_mask = valid_df["next_month_number"] < validation_start
validation_mask = (valid_df["next_month_number"] >= validation_start) & (valid_df["next_month_number"] < test_start)
test_mask = valid_df["next_month_number"] >= test_start

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
predictions = model.predict(X_val)
print(predictions[:10])
print(y_val.head(10))

import numpy as np

actual = y_val.to_numpy()
error = predictions - actual
abs_error = np.abs(error)
mae = np.mean(abs_error)
print(error[:5])
print(mae)

squared_error = error ** 2
mse = np.mean(squared_error)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

actual_mean = np.mean(actual)
ss_re = np.sum(squared_error)
ss_tot = np.sum((actual - actual_mean) ** 2)
r2 = 1 - (ss_re / ss_tot)
print(f"R2: {r2.round(2)}")

baseline_prediction = X_val["median"].to_numpy()
baseline_error = baseline_prediction - actual
print("Baseline shape:", baseline_prediction.shape)
print("Actual shape:", actual.shape)
print("Error shape:", baseline_error.shape)
baseline_mae = np.mean(np.abs(baseline_error))
baseline_mse = np.mean(baseline_error ** 2)
baseline_rmse = np.sqrt(baseline_mse)
print(f"Baseline MAE: {baseline_mae}")
print(f"Baseline RMSE: {baseline_rmse}")

linear_abs_error = np.abs(error)
baseline_abs_error = np.abs(baseline_error)
linear_better = linear_abs_error < baseline_abs_error
linear_wins = np.sum(linear_better)
baseline_wins = np.sum(linear_abs_error > baseline_abs_error)
tie = np.sum(linear_abs_error == baseline_abs_error)
print(f"Linear Regression wins: {linear_wins}")
print(f"Baseline wins: {baseline_wins}")
print(f"Ties: {tie}")

df["median_lag1"] = df.groupby("city")["median"].shift(1)
df["median_lag2"] = df.groupby("city")["median"].shift(2)
print(df.loc[df["city"] == "Abilene", ["year", "month","median", "median_lag1", "median_lag2"]].head(10))

df["lag1_month_number"] = df.groupby("city")["month_number"].shift(1)
df["lag2_month_number"] = df.groupby("city")["month_number"].shift(2)
df["lag1_gap"] = df["month_number"] - df["lag1_month_number"]
df["lag2_gap"] = df["month_number"] - df["lag2_month_number"]
valid_lags = (df["lag1_gap"] == 1) & (df["lag2_gap"] == 2)
print(valid_lags.sum())

valid_lag1 = df["median_lag1"].notna()
valid_lag2 = df["median_lag2"].notna()
valid_three_mask = valid_lags & valid_lag1 & valid_lag2 & valid_mask
three_df = df[valid_three_mask].copy()
x3 = three_df[["median", "median_lag1", "median_lag2"]]
y3 = three_df["next_month_target"]
print(f"Shape x3: {x3.shape}")
print(f"Shape y3: {y3.shape}")
print(x3.head())

train_mask3 = three_df["next_month_number"] < validation_start
val_mask3 = (three_df["next_month_number"] >= validation_start) & (three_df["next_month_number"] < test_start)
x3_train = x3.loc[train_mask3]
y3_train = y3.loc[train_mask3]
x3_val = x3.loc[val_mask3]
y3_val = y3.loc[val_mask3]
print(x3_train.isna().sum())
print(x3_val.isna().sum())
print(f"Training X: {x3_train.shape}")
print(f"Training Y: {y3_train.shape}")
print(f"Validation X: {x3_val.shape}")
print(f"Validation Y: {y3_val.shape}")

one_feature_train = x3_train[["median"]]
one_feature_val = x3_val[["median"]]
model1 = LinearRegression()
model1.fit(one_feature_train, y3_train)
predictions1= model1.predict(one_feature_val)
model2 = LinearRegression()
model2.fit(x3_train, y3_train)
predictions2 = model2.predict(x3_val)
actual3 = y3_val.to_numpy()
prediction3 = x3_val["median"].to_numpy()
mae1 = np.mean(np.abs(predictions1 - actual3))
mae2 = np.mean(np.abs(predictions2 - actual3))
mae3 = np.mean(np.abs(prediction3 - actual3))
print(f"One feature MAE: {mae1}")
print(f"Three feature MAE: {mae2}")
print(f"Baseline MAE: {mae3}")
print(f"Model 1 co-efficients: {model1.coef_}, {model1.intercept_}")
print(f"Model 2 co-efficients: {model2.coef_}, {model2.intercept_}")

df["median_ma6"] = df.groupby("city")["median"].transform(lambda prices: prices.rolling(window=6, min_periods=6).mean())
print(df.loc[df["city"] == "Abilene", ["year", "month", "median", "median_ma6"]].head(10))
print(df["median_ma6"].isna().sum())

df["ma6_start_month"] = df.groupby("city")["month_number"].shift(5)
df["ma6_gap"] = df["month_number"] - df["ma6_start_month"]
valid_ma6_dates = df["ma6_gap"] == 5
print(valid_ma6_dates.sum())
valid_ma6 = valid_ma6_dates & (df["median_ma6"].notna())

print(f"Valid MA6: {valid_ma6.sum()}")
valid_four_mask = valid_three_mask & valid_ma6
four_df = df[valid_four_mask].copy()
x4 = four_df[["median", "median_lag1", "median_lag2", "median_ma6"]]
y4 = four_df["next_month_target"]
print(f"X4 shape: {x4.shape}")
print(f"Y4 shape: {y4.shape}")
print(x4.isna().sum())

train_mask4 = four_df["next_month_number"] < validation_start
val_mask4 = (four_df["next_month_number"] >= validation_start) & (four_df["next_month_number"] < test_start)
x4_train = x4.loc[train_mask4]
y4_train = y4.loc[train_mask4]
x4_val = x4.loc[val_mask4]
y4_val = y4.loc[val_mask4]
print(x4_train.shape)
print(y4_train.shape)
print(x4_val.shape)
print(y4_val.shape)

x3_same_train = x4_train[["median", "median_lag1", "median_lag2"]]
x3_same_val = x4_val[["median", "median_lag1", "median_lag2"]]
print(x3_same_train.shape)
print(x3_same_val.shape)

model3 = LinearRegression()
model4 = LinearRegression()
model3.fit(x3_same_train, y4_train)
model4.fit(x4_train, y4_train)
prediction3 = model3.predict(x3_same_val)
prediction4 = model4.predict(x4_val)
actual4 = y4_val.to_numpy()
mae3 = np.mean(np.abs(prediction3 - actual4))
mae4 = np.mean(np.abs(prediction4 - actual4))
print(f"Three features MAE: {mae3}")
print(f"Four featurres MAE: {mae4}")
print(f"Model 3 co-efficients: {model3.coef_, model3.intercept_}")
print(f"Model 4 co-efficients: {model4.coef_, model4.intercept_}")

baseline4 = x4_val["median"].to_numpy()
print(prediction3.shape)
print(prediction4.shape)
print(baseline4.shape)
print(actual4.shape)
baseline4_error = baseline4 - actual4
baseline4_mae = np.mean(np.abs(baseline4_error))
baseline4_rmse = np.sqrt(np.mean(baseline4_error ** 2))
print(f"Baseline MAE: {baseline4_mae}")
print(f"Baseline RMSE: {baseline4_rmse}")
rmse3 = np.sqrt(np.mean((prediction3 - actual4) ** 2))
rmse4 = np.sqrt(np.mean((prediction4 - actual4) ** 2))
print(f"Three features MAE: {mae3}")
print(f"Three features RMSE: {rmse3}")
print(f"Four featurres MAE: {mae4}")
print(f"Four featurres RMSE: {rmse4}")

diagnostics = four_df.loc[val_mask4, ["city", "year", "month", "next_month_target"]].copy()
diagnostics["predictions"] = prediction4
diagnostics["error"] = prediction4 - actual4
diagnostics["absolute_error"] = np.abs(diagnostics["error"])
largest_error = diagnostics.sort_values(by=["absolute_error"], ascending=False)
print(largest_error.head(10))

city_errors = diagnostics.groupby("city")["absolute_error"].agg(["count", "mean", "max"])
city_errors = city_errors.sort_values(by=["mean"], ascending=False)
print(city_errors.head(10))
print(city_errors.tail(10))

city_train = four_df.loc[train_mask4, "city"]
city_val = four_df.loc[val_mask4, "city"]

train_encoded = pd.get_dummies(city_train, dtype=int)
val_encoded = pd.get_dummies(city_val, dtype=int)
val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
x_city_train = pd.concat([train_encoded, x4_train], axis=1)
x_city_val = pd.concat([val_encoded, x4_val], axis=1)
print(x_city_train.shape)
print(x_city_val.shape)
print(x_city_train.head())

city_model = LinearRegression()
city_model.fit(x_city_train, y4_train)
city_predictions = city_model.predict(x_city_val)
city_error = city_predictions - actual4
city_mae = np.mean(np.abs(city_error))
city_rmse = np.sqrt(np.mean(city_error ** 2))
print(f"City Model MAE: {city_mae}")
print(f"City Model RMSE: {city_rmse}")
print(f"Previous Model MAE: {mae4}")
print(f"Previous Model RMSE {rmse4}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(actual4, prediction4, alpha=0.4)
min_value = min(actual4.min(), prediction4.min())
max_value = max(actual4.max(), prediction4.max())
plt.plot([min_value,max_value], [min_value, max_value], linestyle="--")
plt.grid(True)
plt.xlabel("Actual Median Price ($)")
plt.ylabel("Prediction Median Price ($)")
plt.title("Actual vs Predicted Housing Prices")
plt.show()

residuals = actual4 - prediction4
plt.figure(figsize=(8, 6))
plt.scatter(prediction4, residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.grid(True)
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

final_train_mask = four_df["next_month_number"] < test_start
final_test_mask = four_df["next_month_number"] >= test_start
x_final_train = four_df.loc[final_train_mask, ["median", "median_lag1", "median_lag2", "median_ma6"]]
y_final_train = four_df.loc[final_train_mask, "next_month_target"]
x_final_test = four_df.loc[final_test_mask, ["median", "median_lag1", "median_lag2", "median_ma6"]]
y_final_test = four_df.loc[final_test_mask, "next_month_target"]
print(f"X training shape: {x_final_train.shape}")
print(f"Y training shape: {y_final_train.shape}")
print(f"X testing shape: {x_final_test.shape}")
print(f"Y testing shape: {y_final_test.shape}")

final_model = LinearRegression()
final_model.fit(x_final_train, y_final_train)
test_predictions = final_model.predict(x_final_test)
test_actual = y_final_test.to_numpy()
print(f"Predictions: {test_predictions.shape}")
print(f"Actual: {test_actual.shape}")
print(test_predictions[:5])
print(test_actual[:5])

test_error = test_predictions - test_actual
test_mae = np.mean(np.abs(test_error))
test_rmse = np.sqrt(np.mean(test_error ** 2))
test_mean = np.mean(test_actual)
ss_res = np.sum(test_error ** 2)
ss_tot = np.sum((test_actual - test_mean) ** 2)
test_r2 = 1 - (ss_res / ss_tot)
print(f"Test MAE: {test_mae}")
print(f"Test RMSE: {test_rmse}")
print(f"Test R2: {test_r2}")

baseline_test_predictions  = x_final_test["median"].to_numpy()
baseline_test_error = baseline_test_predictions - test_actual
baseline_test_mae = np.mean(np.abs(baseline_test_error))
baseline_test_rmse = np.sqrt(np.mean(baseline_test_error ** 2))
print(f"Baseline test MAE: {baseline_test_mae}")
print(f"Baseline test RMSE: {baseline_test_rmse}")

import joblib

joblib.dump(final_model, "house_price_model.joblib")
loaded_model = joblib.load("house_price_model.joblib")
loaded_predictions = loaded_model.predict(x_final_test)
print(f"Original: {test_predictions[:5]}")
print(f"Loaded: {loaded_predictions[:5]}")
print(np.allclose(test_predictions, loaded_predictions))
print(loaded_model.n_features_in_)
print(loaded_model.feature_names_in_)