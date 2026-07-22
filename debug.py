# test.py
import yfinance as yf
import pandas as pd

print("Testing yfinance download...")
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01", 
                  progress=False, auto_adjust=True)
print(df.tail())
print(f"Got {len(df)} rows")