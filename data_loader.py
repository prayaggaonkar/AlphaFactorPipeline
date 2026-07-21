import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, os, time
from config import *

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    table = pd.read_html(io.StringIO(response.text))[0]
    tickers = table["Symbol"].str.replace(".", "-").tolist()
    return tickers

def download_prices():
    tickers = get_sp500_tickers()[:UNIVERSE_SIZE]
    print(f"Downloading {len(tickers)} tickers one by one...")

    all_series = []
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=False
            )
            if df.empty or len(df) < 200:
                failed.append(ticker)
                continue
            close = df["Close"].squeeze()
            close.name = ticker
            all_series.append(close)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(tickers)} done, {len(all_series)} successful...")
                time.sleep(1)  # brief pause every 10 tickers
        except Exception as e:
            failed.append(ticker)

    print(f"\nFailed: {len(failed)} tickers: {failed[:10]}...")
    combined = pd.concat(all_series, axis=1)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.8))
    combined = combined.ffill(limit=3)

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_parquet(f"{DATA_DIR}/prices.parquet")
    print(f"Saved {combined.shape[1]} tickers, {combined.shape[0]} days")
    return combined

def load_prices():
    return pd.read_parquet(f"{DATA_DIR}/prices.parquet")

def compute_returns(prices):
    daily_ret = prices.pct_change()
    fwd_ret = daily_ret.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)
    return daily_ret, fwd_ret

if __name__ == "__main__":
    download_prices()