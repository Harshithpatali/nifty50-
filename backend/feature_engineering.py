import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean", "nifty50_clean.csv")

df = pd.read_csv(CLEAN_PATH)

# Ensure Date is datetime
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# ---- Feature 1: Previous Close (Lag 1) ----
df["Prev_Close"] = df["Close"].shift(1)

# ---- Feature 2: Daily Return (percentage change) ----
df["Return"] = df["Close"].pct_change()

# ---- Feature 3: Return Direction (1 = Up, 0 = Down) ----
df["Return_Sign"] = (df["Return"] > 0).astype(int)

# ---- Feature 4: Rolling Volatility (5-day standard deviation) ----
df["Volatility_5d"] = df["Return"].rolling(window=5).std()

# ---- Feature 5: Moving Averages ----
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()

# ---- Feature 6: Price Trend (1 if above MA20, else 0) ----
df["Close_Above_MA20"] = (df["Close"] > df["MA_20"]).astype(int)

# ---- Feature 7: Rolling High/Low difference ----
df["HL_Diff"] = df["High"] - df["Low"]

# ---- Feature 8: Days Since Start (numerical time feature) ----
df["Days"] = (df["Date"] - df["Date"].min()).dt.days

# ---- Feature 9: Date breakdown features ----
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Weekday"] = df["Date"].dt.weekday  # 0=Monday, 6=Sunday
df["DayName"] = df["Date"].dt.day_name()

# ---- Feature 10: Volume rolling average ----
df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()

# ---- Remove initial NaN rows caused by shifting/rolling ----
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ---- Save feature engineered file ----
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)
FEATURE_PATH = os.path.join(FEATURE_DIR, "nifty50_features.csv")
df.to_csv(FEATURE_PATH, index=False)

print("Feature Engineering Done ✅")
print(df.head())
print("Saved engineered file at:", FEATURE_PATH)
