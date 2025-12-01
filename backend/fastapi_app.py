import os
from math import sqrt

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    max_error,
)

# ==================== ENV & DB SETUP ====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# In local dev (Windows) we use .env.
# In Docker we rely on environment variables from docker-compose.
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

if not all([POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD]):
    raise ValueError("Postgres credentials missing in .env")

FEATURE_TABLE = "nifty50_features"

FEATURE_COLS = [
    "Prev_Close",
    "Return_Sign",
    "Volatility_5d",
    "MA_5",
    "MA_20",
    "HL_Diff",
    "Days",
    "Day",
    "Month",
    "Year",
    "Weekday",
    "Volume_MA_5",
    "Volume",
]


def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Nifty50 Linear Regression API",
    description="Predict next-day Nifty50 close using Linear Regression on features stored in PostgreSQL.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Nifty50 LR API is running"}


def load_feature_data() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE};", conn)
    finally:
        conn.close()

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature table is empty. Run Airflow pipeline first.",
        )

    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def train_lr_and_predict(df: pd.DataFrame):
    # Ensure required columns exist
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing feature columns in DB table: {missing}",
        )

    X = df[FEATURE_COLS]
    y = df["Close"]

    # Time-series friendly split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions on test
    y_pred_test = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = sqrt(mse)
    mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)
    smape = float(
        np.mean(
            2 * np.abs(y_test - y_pred_test)
            / (np.abs(y_test) + np.abs(y_pred_test))
        )
        * 100
    )
    evs = explained_variance_score(y_test, y_pred_test)
    max_err = max_error(y_test, y_pred_test)
    bias = float(np.mean(y_pred_test - y_test))
    resid_std = float(np.std(y_test - y_pred_test))

    # Next-day prediction: use last row of X
    latest_row = X.tail(1)
    next_day_pred = float(model.predict(latest_row)[0])

    metrics = {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "explained_variance": evs,
        "max_error": max_err,
        "bias": bias,
        "residual_std": resid_std,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return next_day_pred, metrics


@app.get("/predict/next-day")
def predict_next_day():
    """
    Train a Linear Regression model on latest data from PostgreSQL
    and return the predicted next-day Close price + evaluation metrics.
    """
    df = load_feature_data()
    next_price, metrics = train_lr_and_predict(df)

    # Last known date & close for context
    last_row = df.sort_values("Date").iloc[-1]
    last_date = (
        last_row["Date"].isoformat()
        if hasattr(last_row["Date"], "isoformat")
        else str(last_row["Date"])
    )
    last_close = float(last_row["Close"])

    return {
        "last_date": last_date,
        "last_close": last_close,
        "predicted_next_day_close": next_price,
        "model_metrics": metrics,
    }
