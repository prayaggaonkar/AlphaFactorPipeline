import pandas as pd
import numpy as np
from config import *


def _prep(df):
    df = df.copy()
    df.index.name = "date"
    df.columns.name = "ticker"
    return df

def winsorize(factor, pct=0.01):
    lo = factor.quantile(pct, axis=1)
    hi = factor.quantile(1 - pct, axis=1)
    return factor.clip(lower=lo, upper=hi, axis=0)

def normalize(factor):
    return factor.rank(axis=1, pct=True) - 0.5

def finalize(raw):
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return _prep(normalize(winsorize(raw)))


# ── Momentum ──────────────────────────────────────────────────────

def momentum_1m(prices):
    return finalize(prices.pct_change(21).shift(1))

def momentum_3m(prices):
    return finalize(prices.pct_change(63).shift(1))

def momentum_6m(prices):
    return finalize(prices.pct_change(126).shift(1))

def momentum_12_1(prices):
    raw = prices.pct_change(252) - prices.pct_change(21)
    return finalize(raw.shift(1))


# ── Mean Reversion ────────────────────────────────────────────────

def reversal_5d(prices):
    return finalize(-prices.pct_change(5).shift(1))

def reversal_10d(prices):
    return finalize(-prices.pct_change(10).shift(1))

def distance_ma(prices, window):
    ma = prices.rolling(window).mean()
    raw = -(prices.shift(1) / ma.shift(1) - 1)
    return finalize(raw)

def rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    rsi_val = 100 - (100 / (1 + rs))
    return finalize(-(rsi_val - 50).shift(1))

def bollinger_zscore(prices, window=20):
    mean = prices.rolling(window).mean()
    std  = prices.rolling(window).std()
    raw  = -((prices - mean) / (std + 1e-8)).shift(1)
    return finalize(raw)


# ── Volatility ────────────────────────────────────────────────────

def realized_vol(prices, window=20):
    ret = prices.pct_change()
    # NO negation — high vol was positive signal in our data
    raw = ret.rolling(window).std().shift(1)
    return finalize(raw)

def volatility_ratio(prices):
    ret   = prices.pct_change()
    short = ret.rolling(20).std()
    long  = ret.rolling(120).std()
    raw   = (short / (long + 1e-8)).shift(1)
    return finalize(raw)


# ── Price Structure ───────────────────────────────────────────────

def breakout_52w(prices):
    high = prices.rolling(252).max()
    low  = prices.rolling(252).min()
    raw  = ((prices - low) / (high - low + 1e-8)).shift(1)
    return finalize(raw)


# ── Build Factor Matrix ───────────────────────────────────────────

def build_factor_matrix(prices, volume=None):

    '''
    factors = {
        # Momentum
        "mom_1m":        momentum_1m(prices),
        "mom_3m":        momentum_3m(prices),
        "mom_6m":        momentum_6m(prices),
        "mom_12_1":      momentum_12_1(prices),

        # Mean reversion
        "rev_10d":       reversal_10d(prices),
        "ma20_dist":     distance_ma(prices, 20),
        "ma50_dist":     distance_ma(prices, 50),
        "rsi":           rsi(prices),
        "bollinger":     bollinger_zscore(prices),

        # Volatility
        "realized_vol":  realized_vol(prices),
        "vol_ratio":     volatility_ratio(prices),

        # Trend
        "breakout_52w":  breakout_52w(prices),
    }

    '''
    factors = {
        "realized_vol": realized_vol(prices),
        "ma50_dist":    distance_ma(prices, 50),
        "rsi":          rsi(prices),
        "ma20_dist":    distance_ma(prices, 20),
        "bollinger":    bollinger_zscore(prices),
        "rev_10d":      reversal_10d(prices),
    }
    

    panels = []
    for name, df in factors.items():
        stacked = df.stack(future_stack=True)
        stacked.name = name
        panels.append(stacked)

    factor_matrix = pd.concat(panels, axis=1)
    factor_matrix.index.names = ["date", "ticker"]
    factor_matrix = factor_matrix.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Factor matrix shape: {factor_matrix.shape}")
    print(f"Factor dates: {factor_matrix.index.get_level_values('date').min()} → {factor_matrix.index.get_level_values('date').max()}")
    return factor_matrix