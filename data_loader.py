## data_loader.py
import yfinance as yf
import requests
import pandas as pd
import os
import io
from config import *

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
    """Download adjusted close prices and save to disk."""
    tickers = get_sp500_tickers()[:UNIVERSE_SIZE]
    print(f"Downloading {len(tickers)} tickers...")

    df = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusts for splits/dividends automatically
        progress=True
    )["Close"]

    ## Drop columns (tickers) with more than 20% missing data
    df = df.dropna(axis=1, thresh=int(len(df) * 0.8))

    ## Forward-fill gaps up to 3 days (handles weekends, holidays)
    df = df.ffill(limit=3)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(f"{DATA_DIR}/prices.parquet")
    print(f"Saved {df.shape[1]} tickers, {df.shape[0]} days")
    return df

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