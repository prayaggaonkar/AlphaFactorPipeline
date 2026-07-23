import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from config import *


def build_portfolio(predictions, daily_returns):

    pred_dates = predictions.index.get_level_values("date").unique().sort_values()
    rebal_dates = pred_dates[::FORWARD_DAYS]

    daily_returns = daily_returns.loc[pred_dates[0]:]

    weights = pd.DataFrame(
        0.0,
        index=daily_returns.index,
        columns=list(daily_returns.columns)
    )

    current_weights = pd.Series(0.0, index=daily_returns.columns)

    for date in daily_returns.index:

        if date in rebal_dates:
            sig = predictions.xs(date, level="date").dropna()

            qLow = sig.quantile(0.10)
            qHigh = sig.quantile(0.90)

            longs = sig[sig >= qHigh].index
            shorts = sig[sig <= qLow].index

            current_weights[:] = 0.0

            valid_longs = [x for x in longs if x in daily_returns.columns]
            valid_shorts = [x for x in shorts if x in daily_returns.columns]

            if valid_longs:
                ranks = sig.loc[valid_longs].rank()
                rank_weights = ranks / ranks.sum()
                current_weights.loc[valid_longs] = 0.5 * rank_weights

            if valid_shorts:
                ranks = sig.loc[valid_shorts].rank(ascending=False)
                rank_weights = ranks / ranks.sum()
                current_weights.loc[valid_shorts] = -0.5 * rank_weights

        weights.loc[date, current_weights.index] = current_weights.values

    weights = weights.shift(1).fillna(0.0)

    gross_returns = (weights * daily_returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    transaction_costs = turnover * (COST_BPS / 10000)

    net_returns = gross_returns - transaction_costs

    active_returns = net_returns[weights.abs().sum(axis=1) > 0]

    #TARGET_VOL = 0.15
    #realized_vol = ( active_returns.std() * np.sqrt(252))
    #scaling_factor = TARGET_VOL / realized_vol
    #active_returns = active_returns * scaling_factor

    cumulative = (1 + active_returns).cumprod()

    annual_return = (
        cumulative.iloc[-1] ** (252 / len(active_returns))
        - 1
    )

    annual_vol = active_returns.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    drawdown = cumulative / cumulative.cummax() - 1

    metrics = {
        "annualized_return": round(annual_return * 100, 2),
        "annualized_vol": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(drawdown.min() * 100, 2),
        "win_rate": round((active_returns > 0).mean() * 100, 1),
        "active_days": len(active_returns)
    }

    print("\n── Backtest Results ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    os.makedirs("backtest", exist_ok=True)

    cumulative.to_frame("cum_returns").to_parquet(
        "backtest/cum_returns.parquet"
    )

    active_returns.to_frame("net_pnl").to_parquet(
        "backtest/net_pnl.parquet"
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    cumulative.plot(ax=axes[0])
    axes[0].set_title("Cumulative Returns")
    axes[0].axhline(1, linestyle="--", color="gray")

    rolling_sharpe = (
        active_returns.rolling(63).mean()
        /
        active_returns.rolling(63).std()
        *
        np.sqrt(252)
    )

    rolling_sharpe.plot(ax=axes[1])
    axes[1].set_title("Rolling 63-Day Sharpe")
    axes[1].axhline(0, linestyle="--", color="gray")

    plt.tight_layout()
    plt.savefig("backtest/results.png", dpi=150)
    plt.close()

    print("Chart saved to backtest/results.png")

    return metrics, active_returns, cumulative