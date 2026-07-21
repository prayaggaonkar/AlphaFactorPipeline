import os
from data_loader import load_prices, compute_returns
from factor_lib  import build_factor_matrix
from model       import train_and_evaluate
from backtest    import build_portfolio

print("=== Step 1: Load data ===")
prices = load_prices()
print(f"Prices shape: {prices.shape}")
print(f"Date range:   {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Index name:   {prices.index.name}")
print(f"Columns name: {prices.columns.name}")

print("\n=== Step 2: Compute returns ===")
daily_ret, fwd_ret = compute_returns(prices)

print("\n=== Step 3: Build factors ===")
factor_matrix = build_factor_matrix(prices)

print("\n=== Step 4: Train model ===")
predictions, model, mean_ic = train_and_evaluate(factor_matrix, fwd_ret)

print("\n=== Step 5: Backtest ===")
os.makedirs("backtest", exist_ok=True)
metrics, net_pnl, cum_returns = build_portfolio(predictions, daily_ret)

print("\nDone!")