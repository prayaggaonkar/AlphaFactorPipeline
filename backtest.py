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

