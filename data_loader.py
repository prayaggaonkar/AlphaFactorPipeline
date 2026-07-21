## data_loader.py
import yfinance as yf
import requests
import pandas as pd
import os
import io
from config import *
import time

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    table = pd.read_html(io.StringIO(response.text))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def download_prices():
    tickers = get_sp500_tickers()[:150]
    print(f"Downloading {len(tickers)} tickers in batches...")

    batch_size = 10
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}: {batch[0]} to {batch[-1]}...")
        try:
            # Download one ticker at a time to avoid threading issues
            batch_frames = []
            for ticker in batch:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(
                        start=START_DATE,
                        end=END_DATE,
                        auto_adjust=True
                    )["Close"]
                    if len(hist) > 100:   # skip if barely any data
                        hist.name = ticker
                        batch_frames.append(hist)
                except Exception:
                    pass
            if batch_frames:
                all_data.append(pd.concat(batch_frames, axis=1))
        except Exception as e:
            print(f"Batch failed: {e}")
        time.sleep(2)

    combined = pd.concat(all_data, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.8))
    combined = combined.ffill(limit=3)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)  # strip timezone

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_parquet(f"{DATA_DIR}/prices.parquet")
    print(f"Saved {combined.shape[1]} tickers, {combined.shape[0]} days")
    return combined

def load_prices():
    """Load prices from disk (run download_prices first)."""
    return pd.read_parquet(f"{DATA_DIR}/prices.parquet")

def compute_returns(prices):
    """
    Compute daily percentage returns and forward returns.
    Forward return = how much the stock moves in the NEXT N days.
    This is what we're trying to predict.
    """
    ## Daily return: (today's price - yesterday's price) / yesterday's price
    daily_ret = prices.pct_change()

    ## Forward return: shift by -FORWARD_DAYS so today's row
    ## shows what will happen over the NEXT 5 days
    fwd_ret = daily_ret.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)

    return daily_ret, fwd_ret

## Run this file directly to download data
if __name__ == "__main__":
    download_prices()