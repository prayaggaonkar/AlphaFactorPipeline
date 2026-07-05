## factor_lib.py
import pandas as pd
import numpy as np
from config import *

def normalize(factor: pd.DataFrame) -> pd.DataFrame:
    """
    Two steps:
    1. Cross-sectional rank: for each day, rank all stocks 0 to 1
       This removes market-wide effects (on a good day for stocks,
       we don't want ALL factors to look bullish)
    2. Subtract 0.5 so values range from -0.5 (worst) to +0.5 (best)
    Always do this to every factor before using it.
    """
    ranked = factor.rank(axis=1, pct=True)
    return ranked - 0.5

def winsorize(factor: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """
    Clip extreme values at the 1st and 99th percentile.
    Without this, one stock with a crazy outlier value can
    dominate the entire factor.
    """
    lo = factor.quantile(pct, axis=1)
    hi = factor.quantile(1 - pct, axis=1)
    return factor.clip(lower=lo, upper=hi, axis=0)

## ── MOMENTUM FACTORS ──────────────────────────────────────────────

def momentum_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """1-month return. Stocks up a lot recently tend to keep going up."""
    raw = prices.pct_change(21)        # 21 trading days ≈ 1 month
    raw = raw.shift(1)                 # CRITICAL: shift so we don't use today's price
    return normalize(winsorize(raw))

def momentum_3m(prices: pd.DataFrame) -> pd.DataFrame:
    """3-month return. Medium-term momentum."""
    raw = prices.pct_change(63).shift(1)
    return normalize(winsorize(raw))

def momentum_12m_skip1m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-month return, skipping the last month.
    Why skip? Very short-term momentum reverses (mean reverts).
    Classic Jegadeesh-Titman momentum: 12 months ago to 1 month ago.
    """
    ret_12m = prices.pct_change(252).shift(1)
    ret_1m  = prices.pct_change(21).shift(1)
    raw = ret_12m - ret_1m             # remove the last month
    return normalize(winsorize(raw))

## ── MEAN REVERSION FACTORS ────────────────────────────────────────

def reversal_1w(prices: pd.DataFrame) -> pd.DataFrame:
    """
    1-week reversal. Stocks that dropped hard last week often bounce.
    Note: we NEGATE it — losers are expected to bounce back,
    so a big negative return is a POSITIVE signal.
    """
    raw = -prices.pct_change(5).shift(1)   # negative = reversal
    return normalize(winsorize(raw))

def distance_from_ma(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    How far is the price from its moving average?
    Stocks far above their MA tend to pull back (mean reversion).
    Again negated: far above MA = negative signal.
    """
    ma  = prices.rolling(window).mean().shift(1)
    raw = -(prices.shift(1) / ma - 1)
    return normalize(winsorize(raw))

## ── VOLATILITY FACTORS ────────────────────────────────────────────

def realized_vol(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    How volatile has this stock been recently?
    High-vol stocks are often overpriced due to lottery-ticket demand.
    Negated: high vol = negative signal (low vol anomaly).
    """
    daily_ret = prices.pct_change()
    raw = -daily_ret.rolling(window).std().shift(1)
    return normalize(winsorize(raw))

## ── VOLUME FACTORS ────────────────────────────────────────────────

def relative_volume(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Is today's volume unusually high or low vs. recent average?
    High relative volume can signal institutional activity.
    """
    avg_vol = volume.rolling(window).mean().shift(1)
    raw = (volume.shift(1) / avg_vol) - 1
    return normalize(winsorize(raw))

## ── COMPUTE ALL FACTORS ───────────────────────────────────────────

def build_factor_matrix(prices: pd.DataFrame,
                        volume: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all factors and stack them into a single DataFrame
    where each column is a factor name.
    Returns a MultiIndex DataFrame: (date, ticker) -> factor values
    """
    factors = {
        "mom_1m":          momentum_1m(prices),
        "mom_3m":          momentum_3m(prices),
        "mom_12m_skip1m":  momentum_12m_skip1m(prices),
        "reversal_1w":     reversal_1w(prices),
        "dist_ma20":       distance_from_ma(prices, 20),
        "dist_ma60":       distance_from_ma(prices, 60),
        "vol_21d":         realized_vol(prices, 21),
        "rel_volume":      relative_volume(volume, 20),
    }

    ## Stack: turn wide (date x ticker) into long (date, ticker)
    panels = []
    for name, df in factors.items():
        stacked = df.stack()           # (date, ticker) index, one value per row
        stacked.name = name
        panels.append(stacked)

    factor_matrix = pd.concat(panels, axis=1).dropna()
    print(f"Factor matrix shape: {factor_matrix.shape}")
    return factor_matrix