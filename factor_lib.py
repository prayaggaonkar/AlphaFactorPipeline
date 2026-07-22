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

def vol_momentum(prices, window=21):
    """High recent vol = positive signal (confirmed in our data)"""
    daily_ret = prices.pct_change()
    raw = daily_ret.rolling(window).std().shift(1)
    return _prep(normalize(winsorize(raw)))

def dist_ma120(prices):
    """Distance from 120-day MA — longer horizon mean reversion"""
    ma  = prices.rolling(120).mean().shift(1)
    raw = -(prices.shift(1) / ma - 1)
    return _prep(normalize(winsorize(raw)))

def rsi_reversal(prices, window=14):
    """Low RSI = oversold = buy signal"""
    delta = prices.diff().shift(1)
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-8)
    rsi   = 100 - (100 / (1 + rs))
    raw   = -(rsi - 50)
    return _prep(normalize(winsorize(raw)))

def build_factor_matrix(prices: pd.DataFrame,
                        volume: pd.DataFrame = None) -> pd.DataFrame:
    factors = {
        "dist_ma20":      distance_from_ma(prices, 20),
        "dist_ma60":      distance_from_ma(prices, 60),
        "dist_ma120":     dist_ma120(prices),        # add this
        "reversal_1w":    reversal_1w(prices),
        "rsi_reversal":   rsi_reversal(prices),      # add this
        # momentum FLIPPED — negative IC means signal is inverted
        "mom_1m_flip":    _prep(normalize(winsorize(prices.pct_change(21).shift(1)))),
        "mom_3m_flip":    _prep(normalize(winsorize(prices.pct_change(63).shift(1)))),
        # vol FLIPPED — high vol was positive signal
        "vol_momentum":   vol_momentum(prices),      # add this (no negation)
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