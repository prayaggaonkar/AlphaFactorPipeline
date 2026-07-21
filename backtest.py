## backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *

def build_portfolio(predictions: pd.Series,
                    daily_returns: pd.DataFrame) -> dict:

    pred_dates = predictions.index.get_level_values("date").unique()

    weight_df = pd.DataFrame(0.0, index=daily_returns.index,
                             columns=daily_returns.columns)

    for date in pred_dates[::REBAL_FREQ]:
        if date not in daily_returns.index:
            continue

        sig = predictions.xs(date, level="date").dropna()

        q20 = sig.quantile(0.20)
        q80 = sig.quantile(0.80)

        longs  = sig[sig >= q80].index
        shorts = sig[sig <= q20].index

        if len(longs) == 0 or len(shorts) == 0:
            continue

        ## Find the integer position of this date and fill forward REBAL_FREQ rows
        idx_pos = daily_returns.index.get_loc(date)
        end_pos = min(idx_pos + REBAL_FREQ, len(daily_returns))
        rows    = daily_returns.index[idx_pos:end_pos]

        ## Build weight vector for this rebalance
        w = pd.Series(0.0, index=daily_returns.columns)
        valid_longs  = [t for t in longs  if t in w.index]
        valid_shorts = [t for t in shorts if t in w.index]

        if valid_longs:
            w[valid_longs]  =  0.5 / len(valid_longs)
        if valid_shorts:
            w[valid_shorts] = -0.5 / len(valid_shorts)

        ## Use .loc to assign cleanly — avoids ChainedAssignmentError
        weight_df.loc[rows, :] = w.values

    ## Shift weights by 1 day: use yesterday's weights on today's returns
    weight_df = weight_df.shift(1).fillna(0.0)

    ## Daily P&L
    daily_pnl = (weight_df * daily_returns).sum(axis=1)

    ## Transaction costs on rebalance days
    turnover  = weight_df.diff().abs().sum(axis=1)
    tc        = turnover * (COST_BPS / 10000)
    net_pnl   = daily_pnl - tc

    ## Cumulative returns
    cum_returns = (1 + net_pnl).cumprod()

    ## Save for Streamlit dashboard
    import os
    os.makedirs("backtest", exist_ok=True)
    cum_returns.to_frame("cum_returns").to_parquet("backtest/cum_returns.parquet")
    net_pnl.to_frame("net_pnl").to_parquet("backtest/net_pnl.parquet")

    ## Performance metrics
    ann_return = net_pnl.mean() * 252
    ann_vol    = net_pnl.std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd     = (cum_returns / cum_returns.cummax() - 1).min()

    metrics = {
        "annualized_return": round(ann_return * 100, 2),
        "annualized_vol":    round(ann_vol * 100, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "max_drawdown":      round(max_dd * 100, 2),
        "win_rate":          round((net_pnl > 0).mean() * 100, 1),
    }

    print("\n── Backtest Results ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    ## Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    cum_returns.plot(ax=ax1, color="steelblue", label="Strategy")
    ax1.set_title("Cumulative Returns")
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.7)
    ax1.legend()

    roll_sharpe = (net_pnl.rolling(63).mean() /
                   net_pnl.rolling(63).std() * np.sqrt(252))
    roll_sharpe.plot(ax=ax2, color="darkorange")
    ax2.set_title("Rolling 63-day Sharpe")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.savefig("backtest/results.png", dpi=150)
    print("\nChart saved to backtest/results.png")

    return metrics, net_pnl, cum_returns