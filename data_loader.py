import pandas as pd
import numpy as np
import yfinance as yf
import os, time
from config import *

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
    "MRK", "PEP", "COST", "ADBE", "CRM", "TMO", "ACN", "MCD", "CSCO",
    "ABT", "DHR", "LIN", "TXN", "NEE", "PM", "RTX", "HON", "UPS",
    "BMY", "AMGN", "QCOM", "IBM", "CAT", "GE", "SBUX", "BA", "GS",
    "BLK", "AXP", "SPGI", "PLD", "AMT", "SYK", "GILD", "MDT", "ADP",
    "ISRG", "VRTX", "TJX", "CB", "MO", "DUK", "SO", "EXC", "SRE",
    "HIG", "ALL", "PGR", "AFL", "MET", "PRU", "TRV", "EME", "PWR",
    "ROK", "AME", "PH", "ITW", "DOV", "IR", "XYL", "RSG", "WM",
    "ECL", "SHW", "PPG", "LYB", "DOW", "VLO", "PSX", "MPC", "HES",
    "OXY", "COP", "EOG", "HAL", "SLB", "ICE", "MSCI", "MCO", "NDAQ",
    "WFC", "C", "MS", "BAC", "USB", "PNC", "COF", "LMT", "NOC", "GD",
]

def download_prices():
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = list(dict.fromkeys(TICKERS))
    print(f"Downloading {len(tickers)} tickers in batches...")

    all_series = []

    # Download in batches of 20 with pauses
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}: {batch[0]} to {batch[-1]}...")
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
            )
            # New yfinance returns ("Close", "AAPL") MultiIndex columns
            close = raw["Close"]
            if isinstance(close, pd.Series):
                # Only one ticker came back
                close = close.to_frame(name=batch[0])
            # Drop tickers with less than 200 rows
            close = close.dropna(axis=1, thresh=200)
            all_series.append(close)
            print(f"  Got {close.shape[1]} tickers")
        except Exception as e:
            print(f"  Batch failed: {e}")

        print(f"  Sleeping 10s...")
        time.sleep(10)

    if not all_series:
        raise ValueError("No data downloaded.")

    combined = pd.concat(all_series, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.index.name = "date"
    combined.columns.name = "ticker"
    combined = combined.sort_index()
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.8))
    combined = combined.ffill(limit=3)

    combined.to_parquet(f"{DATA_DIR}/prices.parquet")
    print(f"\nSaved {combined.shape[1]} tickers, {combined.shape[0]} days")
    print(f"Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    return combined

def load_prices():
    df = pd.read_parquet(f"{DATA_DIR}/prices.parquet")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df.columns.name = "ticker"
    return df

def compute_returns(prices):
    daily_ret = prices.pct_change()
    fwd_ret = daily_ret.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)
    return daily_ret, fwd_ret

if __name__ == "__main__":
    download_prices()