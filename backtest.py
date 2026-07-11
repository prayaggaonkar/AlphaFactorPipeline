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

