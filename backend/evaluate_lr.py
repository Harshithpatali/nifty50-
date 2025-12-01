import os
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from math import sqrt
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, explained_variance_score,
    max_error
)

print(">>> Evaluation Script started")

# ----- Load .env -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ----- DB Connection -----
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

# ----- Pull feature table -----
df = pd.read_sql("SELECT * FROM nifty50_features;", conn)
conn.close()

print("Rows pulled:", len(df))

# Make sure order is by date
df = df.sort_values("Date").reset_index(drop=True)

# ----- Define X, y -----
feature_cols = [
    "Prev_Close", "Return_Sign", "Volatility_5d",
    "MA_5", "MA_20", "HL_Diff", "Days",
    "Day", "Month", "Year", "Weekday",
    "Volume_MA_5", "Volume"
]

X = df[feature_cols]
y = df["Close"]

# ----- Train/test split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ----- Train model -----
model = LinearRegression()
model.fit(X_train, y_train)

# ----- Predictions -----
y_pred_test = model.predict(X_test)

# ----- Metrics -----
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
smape = np.mean(2 * np.abs(y_test - y_pred_test) / (np.abs(y_test) + np.abs(y_pred_test))) * 100
evs = explained_variance_score(y_test, y_pred_test)
max_err = max_error(y_test, y_pred_test)
bias = np.mean(y_pred_test - y_test)
resid_std = np.std(y_test - y_pred_test)
dw = sm.stats.durbin_watson(y_test - y_pred_test)

# ----- Adjusted R² -----
n = len(y_test)
p = X_train.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# ----- Coeff table -----
coeff_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

# ----- Print results -----
print("\n📊 **Evaluation Metrics**")
print("R²:", r2)
print("Adjusted R²:", adj_r2)
print("Explained Variance Score:", evs)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape, "%")
print("sMAPE:", smape, "%")
print("Bias:", bias)
print("Residual Std Dev:", resid_std)
print("Max Error:", max_err)
print("Durbin–Watson:", dw)
print("\n🔍 **Feature Impact Table**")
print(coeff_table)
print("\n>>> Evaluation Script finished")
