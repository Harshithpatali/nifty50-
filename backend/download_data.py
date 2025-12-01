import os
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd

# Load env (for later DB usage if needed)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


def download_nifty50_index_history(
    ticker: str = "^NSEI",  # Nifty 50 index ticker on Yahoo Finance
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical Nifty 50 index data from Yahoo Finance.

    :param ticker: Yahoo Finance ticker for Nifty 50
    :param start: start date (YYYY-MM-DD)
    :param end: end date (YYYY-MM-DD) or None for today
    :param interval: data interval ("1d", "1h", "5m", etc.)
    :return: DataFrame with OHLCV data
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print(f"Downloading data for {ticker} from {start} to {end} at interval {interval}...")
    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError("No data returned from yfinance. Check ticker or dates.")

    df.reset_index(inplace=True)
    return df


def save_to_csv(df: pd.DataFrame, filename: str) -> str:
    """
    Save DataFrame to CSV in ../data/raw directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved data to: {file_path}")
    return file_path


if __name__ == "__main__":
    # You can adjust these later based on your needs
    df_nifty = download_nifty50_index_history(
        ticker="^NSEI",
        start="2015-01-01",
        interval="1d",
    )
    print(df_nifty.head())

    save_to_csv(df_nifty, "nifty50_index_daily.csv")
