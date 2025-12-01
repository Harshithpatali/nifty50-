import os
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "nifty50_index_daily.csv")

df = pd.read_csv(RAW_PATH, header=[0,1])

# remove multi head index
df.columns = df.columns.get_level_values(0)

# rename for exact format
df.rename(columns={"Date": "Date",
                   "Close": "Close",
                   "High": "High",
                   "Low": "Low",
                   "Open": "Open",
                   "Volume": "Volume"}, inplace=True)

# keep only required columns
df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]

df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)
CLEAN_PATH = os.path.join(CLEAN_DIR, "nifty50_clean.csv")
df.to_csv(CLEAN_PATH, index=False)

print("Cleaned Data ✅")
print(df.head())
print("Saved to:", CLEAN_PATH)
