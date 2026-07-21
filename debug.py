# run this as a standalone script: debug.py
from data_loader import load_prices, compute_returns
import yfinance as yf
from config import *

prices = load_prices()
print("=== DATA CHECK ===")
print(f"Shape:      {prices.shape}")
print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Tickers:    {list(prices.columns[:5])}...")
print(f"Sample prices (AAPL):\n{prices['AAPL'].tail()}")

daily_ret, fwd_ret = compute_returns(prices)
print(f"\n=== RETURNS CHECK ===")
print(f"Daily returns sample:\n{daily_ret['AAPL'].dropna().tail()}")
print(f"Forward returns sample:\n{fwd_ret['AAPL'].dropna().tail()}")
print(f"Forward ret NaN %: {fwd_ret.isna().mean().mean()*100:.1f}%")