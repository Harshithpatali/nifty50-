import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# ================== STREAMLIT CONFIG ==================
st.set_page_config(
    page_title="Nifty50 LR Predictor",
    page_icon="📈",
    layout="centered",
)

st.title("📈 Nifty50 Next-Day Price Prediction")
st.caption("Streamlit Cloud version • Linear Regression + yfinance (no FastAPI / no Postgres)")

st.markdown("---")


# ================== DATA & FEATURES ==================
@st.cache_data(ttl=60 * 60)
def load_nifty_data():
    # Adjust ticker if you used a different one in training (e.g. ^NSEI)
    df = yf.download("^NSEI", period="2y", interval="1d")
    df.reset_index(inplace=True)
    df.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj_Close",
            "Volume": "Volume",
        },
        inplace=True,
    )
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date")

    # Same idea as your DB features
    df["Prev_Close"] = df["Close"].shift(1)

    # Return sign: sign of daily return (Close - Prev_Close)
    df["Return_Sign"] = np.sign(df["Close"] - df["Prev_Close"])

    # 5-day volatility of returns (approx)
    returns = df["Close"].pct_change()
    df["Volatility_5d"] = returns.rolling(5).std()

    # Moving averages
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    # High-Low difference
    df["HL_Diff"] = df["High"] - df["Low"]

    # Days since start
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    # Date breakdown
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Weekday"] = df["Date"].dt.weekday  # Monday=0

    # Volume MA 5
    df["Volume_MA_5"] = df["Volume"].rolling(5).mean()

    # Drop rows with NaN from rolling/shift
    df = df.dropna().reset_index(drop=True)

    return df


def prepare_xy(df_feat: pd.DataFrame):
    feature_cols = [
        "Prev_Close", "Return_Sign", "Volatility_5d",
        "MA_5", "MA_20", "HL_Diff", "Days",
        "Day", "Month", "Year", "Weekday", "Volume_MA_5", "Volume"
    ]
    X = df_feat[feature_cols]
    y = df_feat["Close"]
    return X, y, feature_cols


# ================== MODEL TRAINING ==================
@st.cache_data(ttl=60 * 60)
def train_model(df_feat: pd.DataFrame):
    X, y, feature_cols = prepare_xy(df_feat)

    # Time-series aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Test MAE": mae,
        "Test MSE": mse,
        "Test RMSE": rmse,
    }

    return model, metrics, X, y, feature_cols


# ================== MAIN LOGIC ==================
with st.spinner("Downloading Nifty50 data and training model..."):
    raw_df = load_nifty_data()
    feat_df = create_features(raw_df)
    model, metrics, X_all, y_all, feature_cols = train_model(feat_df)

# Latest row for prediction
latest_row = feat_df.iloc[-1]
X_latest = latest_row[feature_cols].to_frame().T
predicted_price = model.predict(X_latest)[0]

last_date = latest_row["Date"]
last_close = latest_row["Close"]

# ================== UI DISPLAY ==================
st.subheader("📅 Latest Market Snapshot")
col1, col2 = st.columns(2)
with col1:
    if isinstance(last_date, (pd.Timestamp, datetime)):
        last_date_str = last_date.strftime("%Y-%m-%d")
    else:
        last_date_str = str(last_date)
    st.metric("Last Date", last_date_str)
with col2:
    st.metric("Last Close", f"{last_close:,.2f}")

st.markdown("---")

st.subheader("🔮 Predicted Next-Day Close")
delta = predicted_price - last_close
st.metric(
    "Predicted Close (Next Session)",
    f"{predicted_price:,.2f}",
    f"{delta:+.2f} vs last close",
)

st.markdown("---")
st.subheader("📊 Model Evaluation Metrics")

metrics_df = pd.DataFrame(
    [{"Metric": k, "Value": float(v)} for k, v in metrics.items()]
)
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")
st.subheader("📈 Recent Nifty50 Close Prices (last 60 days)")
st.line_chart(raw_df.set_index("Date")["Close"].tail(60))

with st.expander("Show feature-engineered data (tail)"):
    st.dataframe(feat_df.tail(10), use_container_width=True)

st.markdown("---")
st.caption(
    "This Streamlit Cloud app runs everything inside one script: "
    "yfinance → feature engineering → Linear Regression → prediction.\n"
    "Your local FastAPI + Postgres pipeline is still valid for offline / local use."
)
