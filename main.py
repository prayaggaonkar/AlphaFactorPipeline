## main.py  — run this file to execute the whole project
from data_loader import load_prices, compute_returns
from factor_lib  import build_factor_matrix
from model       import train_and_evaluate
from backtest    import build_portfolio
import yfinance as yf
from config import UNIVERSE_SIZE, START_DATE, END_DATE

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

print("=== Step 1: Load data ===")
prices = load_prices()      # run data_loader.py first to download
volume = yf.download(
    list(prices.columns), start=START_DATE, end=END_DATE,
    auto_adjust=True, progress=False
)["Volume"].reindex(columns=prices.columns)

print("=== Step 2: Compute returns ===")
daily_ret, fwd_ret = compute_returns(prices)

print("=== Step 3: Build factors ===")
factor_matrix = build_factor_matrix(prices, volume)

print("=== Step 4: Train model ===")
predictions, model, mean_ic = train_and_evaluate(factor_matrix, fwd_ret)

print("=== Step 5: Backtest ===")
metrics, net_pnl, cum_returns = build_portfolio(predictions, daily_ret)

print("\nDone! Check backtest/results.png and models/shap_importance.png")