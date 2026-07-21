## factor_lib.py — rebuilt around factors that showed positive IC
import pandas as pd
import numpy as np
from config import *

def normalize(factor):
    return factor.rank(axis=1, pct=True) - 0.5

def winsorize(factor, pct=0.01):
    lo = factor.quantile(pct, axis=1)
    hi = factor.quantile(1 - pct, axis=1)
    return factor.clip(lower=lo, upper=hi, axis=0)

## KEEP: positive IC
def dist_ma60(prices):
    """Distance from 60-day MA — strongest factor (IC 0.03)"""
    ma  = prices.rolling(60).mean().shift(1)
    raw = -(prices.shift(1) / ma - 1)   # negative = mean reversion
    return normalize(winsorize(raw))

def dist_ma20(prices):
    """Distance from 20-day MA"""
    ma  = prices.rolling(20).mean().shift(1)
    raw = -(prices.shift(1) / ma - 1)
    return normalize(winsorize(raw))

def dist_ma120(prices):
    """Distance from 120-day MA — longer horizon"""
    ma  = prices.rolling(120).mean().shift(1)
    raw = -(prices.shift(1) / ma - 1)
    return normalize(winsorize(raw))

def reversal_1m(prices):
    """1-month reversal — showed weak positive IC"""
    raw = -prices.pct_change(21).shift(1)
    return normalize(winsorize(raw))

def reversal_1w(prices):
    """1-week reversal"""
    raw = -prices.pct_change(5).shift(1)
    return normalize(winsorize(raw))

def bb_position(prices, window=20):
    """
    Bollinger Band position — where is price within its band?
    Below lower band = oversold = buy signal (mean reversion)
    """
    ma  = prices.rolling(window).mean().shift(1)
    std = prices.rolling(window).std().shift(1)
    raw = -(prices.shift(1) - ma) / (2 * std + 1e-8)
    return normalize(winsorize(raw))

def rsi_reversal(prices, window=14):
    """
    RSI — but used as a mean reversion signal.
    Low RSI (oversold) = positive signal.
    """
    delta = prices.diff().shift(1)
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-8)
    rsi   = 100 - (100 / (1 + rs))
    raw   = -(rsi - 50)   # flip: low RSI = positive signal
    return normalize(winsorize(raw))

## FLIP: vol_21d had negative IC meaning HIGH vol = positive signal
def vol_momentum(prices, window=21):
    """High recent volatility = positive signal (per our data)"""
    daily_ret = prices.pct_change()
    raw = daily_ret.rolling(window).std().shift(1)   # no negation
    return normalize(winsorize(raw))

def build_factor_matrix(prices, volume=None):
    factors = {
        "dist_ma20":    dist_ma20(prices),
        "dist_ma60":    dist_ma60(prices),
        "dist_ma120":   dist_ma120(prices),
        "reversal_1w":  reversal_1w(prices),
        "reversal_1m":  reversal_1m(prices),
        "bb_position":  bb_position(prices),
        "rsi_reversal": rsi_reversal(prices),
        "vol_momentum": vol_momentum(prices),
    }

    panels = []
    for name, df in factors.items():
        df.index.name   = "date"
        df.columns.name = "ticker"
        stacked = df.stack(future_stack=True)
        stacked.name = name
        panels.append(stacked)

    factor_matrix = pd.concat(panels, axis=1)
    factor_matrix.index.names = ["date", "ticker"]
    factor_matrix = factor_matrix.dropna()
    print(f"Factor matrix shape: {factor_matrix.shape}")
    return factor_matrix