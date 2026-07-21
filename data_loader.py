import pandas as pd
import numpy as np
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
    # Import here so if curl_cffi isn't installed we get a clear error
    try:
        import yfinance as yf
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome110")
    except ImportError:
        raise ImportError("Run: pip install yfinance curl_cffi")

    os.makedirs(DATA_DIR, exist_ok=True)
    all_series = []
    failed = []

    tickers = list(dict.fromkeys(TICKERS))
    print(f"Downloading {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker, session=session)
            hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
            if hist.empty or len(hist) < 200:
                failed.append(ticker)
                continue
            close = hist["Close"].copy()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close.name = ticker
            all_series.append(close)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tickers)} — {len(all_series)} successful")
            time.sleep(0.5)
        except Exception as e:
            failed.append(ticker)

    print(f"\nFailed: {len(failed)}, Successful: {len(all_series)}")

    if not all_series:
        raise ValueError("No data downloaded.")

    combined = pd.concat(all_series, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    combined.columns.name = "ticker"
    combined = combined.sort_index()
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.8))
    combined = combined.ffill(limit=3)

    combined.to_parquet(f"{DATA_DIR}/prices.parquet")
    print(f"Saved {combined.shape[1]} tickers, {combined.shape[0]} days")
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