import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from config import *


def build_portfolio(predictions, daily_returns):

    pred_dates = predictions.index.get_level_values("date").unique().sort_values()
    rebal_dates = pred_dates[::REBAL_FREQ]

    # Trim daily_returns to only the period where we have predictions
    first_pred = pred_dates[0]
    daily_returns = daily_returns.loc[first_pred:]

    weight_df = pd.DataFrame(
        0.0,
        index=daily_returns.index,
        columns=daily_returns.columns
    )

    for date in rebal_dates:
        if date not in daily_returns.index:
            continue

        sig = predictions.xs(date, level="date").dropna()

        q20 = sig.quantile(0.20)
        q80 = sig.quantile(0.80)

        longs  = sig[sig >= q80].index
        shorts = sig[sig <= q20].index

        if len(longs) == 0 or len(shorts) == 0:
            continue

        w = pd.Series(0.0, index=daily_returns.columns)
        valid_longs  = [t for t in longs  if t in daily_returns.columns]
        valid_shorts = [t for t in shorts if t in daily_returns.columns]

        if valid_longs:
            w[valid_longs]  =  0.5 / len(valid_longs)
        if valid_shorts:
            w[valid_shorts] = -0.5 / len(valid_shorts)

        idx_pos = daily_returns.index.get_loc(date)
        end_pos = min(idx_pos + REBAL_FREQ, len(daily_returns))

        for j in range(idx_pos, end_pos):
            weight_df.loc[daily_returns.index[j]] = w

    weight_df = weight_df.shift(1).fillna(0.0)

    daily_pnl = (weight_df * daily_returns).sum(axis=1)

    turnover = weight_df.diff().abs().sum(axis=1)
    tc       = turnover * (COST_BPS / 10000)
    net_pnl  = daily_pnl - tc

    # Trim to only active trading days
    active = weight_df.abs().sum(axis=1) > 0
    net_pnl_active = net_pnl[active]

    cum_returns = (1 + net_pnl).cumprod()

    os.makedirs("backtest", exist_ok=True)
    cum_returns.to_frame("cum_returns").to_parquet("backtest/cum_returns.parquet")
    net_pnl.to_frame("net_pnl").to_parquet("backtest/net_pnl.parquet")

    # Compute metrics on active days only
    ann_return = net_pnl_active.mean() * 252
    ann_vol    = net_pnl_active.std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd     = (cum_returns / cum_returns.cummax() - 1).min()

    metrics = {
        "annualized_return": round(ann_return * 100, 2),
        "annualized_vol":    round(ann_vol * 100, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "max_drawdown":      round(max_dd * 100, 2),
        "win_rate":          round((net_pnl_active > 0).mean() * 100, 1),
        "active_days":       int(active.sum()),
    }

    print("\n── Backtest Results ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    cum_returns.plot(ax=ax1, color="steelblue", label="Strategy")
    ax1.set_title("Cumulative Returns")
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.7)
    ax1.legend()

    roll_sharpe = (
        net_pnl.rolling(63).mean() /
        (net_pnl.rolling(63).std() + 1e-8) *
        np.sqrt(252)
    )
    roll_sharpe.plot(ax=ax2, color="darkorange")
    ax2.set_title("Rolling 63-day Sharpe")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.savefig("backtest/results.png", dpi=150)
    plt.close()
    print("Chart saved to backtest/results.png")

    return metrics, net_pnl, cum_returns