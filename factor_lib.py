import pandas as pd
import numpy as np
from config import *


def normalize(factor: pd.DataFrame) -> pd.DataFrame:
    ranked = factor.rank(axis=1, pct=True)
    return ranked - 0.5


def winsorize(factor: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    lo = factor.quantile(pct, axis=1)
    hi = factor.quantile(1 - pct, axis=1)
    return factor.clip(lower=lo, upper=hi, axis=0)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index/column names are set correctly before stacking."""
    df = df.copy()
    df.index.name = "date"
    df.columns.name = "ticker"
    return df


def momentum_1m(prices):
    raw = prices.pct_change(21).shift(1)
    return _prep(normalize(winsorize(raw)))


def momentum_3m(prices):
    raw = prices.pct_change(63).shift(1)
    return _prep(normalize(winsorize(raw)))


def momentum_12m_skip1m(prices):
    ret_12m = prices.pct_change(252).shift(1)
    ret_1m  = prices.pct_change(21).shift(1)
    raw = ret_12m - ret_1m
    return _prep(normalize(winsorize(raw)))


def reversal_1w(prices):
    raw = -prices.pct_change(5).shift(1)
    return _prep(normalize(winsorize(raw)))


def distance_from_ma(prices, window=20):
    ma  = prices.rolling(window).mean().shift(1)
    raw = -(prices.shift(1) / ma - 1)
    return _prep(normalize(winsorize(raw)))


def realized_vol(prices, window=21):
    daily_ret = prices.pct_change()
    raw = -daily_ret.rolling(window).std().shift(1)
    return _prep(normalize(winsorize(raw)))


def relative_volume(volume, window=20):
    avg_vol = volume.rolling(window).mean().shift(1)
    raw = (volume.shift(1) / avg_vol) - 1
    return _prep(normalize(winsorize(raw)))


def build_factor_matrix(prices: pd.DataFrame,
                        volume: pd.DataFrame = None) -> pd.DataFrame:
    factors = {
        "mom_1m":         momentum_1m(prices),
        "mom_3m":         momentum_3m(prices),
        "mom_12m_skip1m": momentum_12m_skip1m(prices),
        "reversal_1w":    reversal_1w(prices),
        "dist_ma20":      distance_from_ma(prices, 20),
        "dist_ma60":      distance_from_ma(prices, 60),
        "vol_21d":        realized_vol(prices, 21),
    }

    panels = []
    for name, df in factors.items():
        stacked = df.stack(future_stack=True)
        stacked.name = name
        panels.append(stacked)

    factor_matrix = pd.concat(panels, axis=1)
    factor_matrix.index.names = ["date", "ticker"]
    factor_matrix = factor_matrix.dropna()

    print(f"Factor matrix shape: {factor_matrix.shape}")
    print(f"Factor date range: {factor_matrix.index.get_level_values('date').min()} → {factor_matrix.index.get_level_values('date').max()}")
    return factor_matrix