## backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *

def build_portfolio(predictions: pd.Series,
                    daily_returns: pd.DataFrame) -> dict:
    """
    Convert model predictions into a portfolio and simulate returns.

    predictions: MultiIndex Series (date, ticker) -> predicted return
    daily_returns: DataFrame (date x ticker) -> actual daily returns
    """
    ## Get all dates where we have predictions
    pred_dates = predictions.index.get_level_values("date").unique()

    weights_list = []

    for date in pred_dates[::REBAL_FREQ]:    # rebalance every N days
        sig = predictions.xs(date, level="date").dropna()

        ## Rank into quintiles (5 groups)
        q20 = sig.quantile(0.20)   # bottom 20%
        q80 = sig.quantile(0.80)   # top 20%

        longs  = sig[sig >= q80].index   # buy these
        shorts = sig[sig <= q20].index   # short these

        if len(longs) == 0 or len(shorts) == 0:
            continue

        w = pd.Series(0.0, index=daily_returns.columns)
        w[longs]  =  0.5 / len(longs)    # long leg = 50% of NAV
        w[shorts] = -0.5 / len(shorts)   # short leg = 50% of NAV

        weights_list.append({"date": date, "weights": w})

    ## Forward-fill weights between rebalance dates
    all_dates   = daily_returns.index
    weight_df   = pd.DataFrame(0.0, index=all_dates,
                               columns=daily_returns.columns)
    for item in weights_list:
        date = item["date"]
        if date in all_dates:
            weight_df.loc[date:].iloc[:REBAL_FREQ] = item["weights"].values

    weight_df = weight_df.shift(1)   # use yesterday's weights for today's returns

    ## Daily P&L: sum of (weight * return) across all stocks
    daily_pnl = (weight_df * daily_returns).sum(axis=1)

    ## Transaction costs: on rebalance days, pay COST_BPS per unit of turnover
    rebal_mask = pd.Series(False, index=all_dates)
    for item in weights_list:
        if item["date"] in all_dates:
            rebal_mask[item["date"]] = True

    turnover  = weight_df.diff().abs().sum(axis=1)
    tc        = turnover * (COST_BPS / 10000) * rebal_mask
    net_pnl   = daily_pnl - tc

    ## Cumulative returns
    cm_returns = (1 + net_pnl).cumprod()

    ## Performance metrics
    ann_return = net_pnl.mean() * 252
    ann_vol    = net_pnl.std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd     = (cm_returns / cm_returns.cummax() - 1).min()

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
    cm_returns.plot(ax=ax1, label="Strategy", color="steelblue")
    ax1.set_title("Cumulative returns")
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.7)
    ax1.legend()

    rolling_sharpe = (net_pnl.rolling(63).mean() /
                      net_pnl.rolling(63).std() * np.sqrt(252))
    rolling_sharpe.plot(ax=ax2, color="darkorange")
    ax2.set_title("Rolling 63-day Sharpe ratio")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.savefig("backtest/results.png", dpi=150)
    print("\nChart saved to backtest/results.png")

    return metrics, net_pnl, cm_returns