'''
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

def mom_1m_flip(prices):
    """Momentum 1m flipped — negative IC in our data so we invert"""
    raw = prices.pct_change(21).shift(1)
    return _prep(normalize(winsorize(raw)))

def mom_3m_flip(prices):
    """Momentum 3m flipped"""
    raw = prices.pct_change(63).shift(1)
    return _prep(normalize(winsorize(raw)))

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
'''
import pandas as pd
import numpy as np
from config import *


# =========================
# Preprocessing
# =========================

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


# =========================
# Momentum Factors
# =========================

def momentum_1m(prices):
    return finalize(prices.pct_change(21).shift(1))

def momentum_3m(prices):
    return finalize(prices.pct_change(63).shift(1))

def momentum_6m(prices):
    return finalize(prices.pct_change(126).shift(1))

def momentum_12_1(prices):
    raw = prices.pct_change(252) - prices.pct_change(21)
    return finalize(raw.shift(1))

def momentum_acceleration(prices):
    mom3 = prices.pct_change(63)
    mom1 = prices.pct_change(21)
    return finalize((mom1 - mom3 / 3).shift(1))


# =========================
# Mean Reversion Factors
# =========================

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

    gain = (
        delta.clip(lower=0)
        .ewm(alpha=1/window)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .ewm(alpha=1/window)
        .mean()
    )

    rs = gain / (loss + 1e-8)
    rsi_value = 100 - (100 / (1 + rs))

    return finalize(-(rsi_value - 50))

def bollinger_zscore(prices, window=20):
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()

    raw = -(
        (prices.shift(1) - mean.shift(1))
        /
        std.shift(1)
    )

    return finalize(raw)


# =========================
# Trend Factors
# =========================

def ema_crossover(prices):
    ema20 = prices.ewm(span=20).mean()
    ema60 = prices.ewm(span=60).mean()

    raw = (ema20 / ema60 - 1).shift(1)

    return finalize(raw)

def ma200_distance(prices):
    ma = prices.rolling(200).mean()

    raw = (
        prices.shift(1) /
        ma.shift(1)
        - 1
    )

    return finalize(raw)

def trend_slope(prices, window=60):
    def slope(x):
        if np.isnan(x).any():
            return np.nan

        return np.polyfit(
            np.arange(len(x)),
            x,
            1
        )[0]

    raw = prices.rolling(window).apply(
        slope,
        raw=False
    )

    return finalize(raw.shift(1))


# =========================
# Volatility Factors
# =========================

def realized_vol(prices, window=20):
    ret = prices.pct_change()

    raw = (
        -ret
        .rolling(window)
        .std()
        .shift(1)
    )

    return finalize(raw)

def volatility_ratio(prices):
    ret = prices.pct_change()

    short = ret.rolling(20).std()
    long = ret.rolling(120).std()

    raw = -(short / long).shift(1)

    return finalize(raw)

def downside_volatility(prices):
    ret = prices.pct_change()

    downside = ret.where(ret < 0)

    raw = (
        -downside
        .rolling(20)
        .std()
        .shift(1)
    )

    return finalize(raw)

def parkinson_volatility(prices):
    high_low = np.log(
        prices.rolling(2).max()
        /
        prices.rolling(2).min()
    )

    raw = (
        -high_low
        .rolling(20)
        .std()
        .shift(1)
    )

    return finalize(raw)


# =========================
# Volume / Liquidity Factors
# =========================

def relative_volume(volume):
    avg = volume.rolling(20).mean()

    raw = (
        volume.shift(1)
        /
        avg.shift(1)
    )

    return finalize(raw)

def volume_trend(volume):
    short = volume.rolling(20).mean()
    long = volume.rolling(60).mean()

    raw = (
        short / long - 1
    ).shift(1)

    return finalize(raw)

def dollar_volume(prices, volume):
    raw = (
        prices * volume
    ).rolling(20).mean().shift(1)

    return finalize(raw)


# =========================
# Statistical Factors
# =========================

def skewness(prices):
    ret = prices.pct_change()

    raw = (
        ret
        .rolling(60)
        .skew()
        .shift(1)
    )

    return finalize(raw)

def kurtosis(prices):
    ret = prices.pct_change()

    raw = (
        ret
        .rolling(60)
        .kurt()
        .shift(1)
    )

    return finalize(raw)

def autocorrelation(prices):
    ret = prices.pct_change()

    raw = (
        ret
        .rolling(60)
        .corr(ret.shift(1))
        .shift(1)
    )

    return finalize(raw)


# =========================
# Price Structure Factors
# =========================

def breakout_52w(prices):
    high = prices.rolling(252).max()
    low = prices.rolling(252).min()

    raw = (
        prices.shift(1) - low.shift(1)
    ) / (
        high.shift(1)
        -
        low.shift(1)
        +
        1e-8
    )

    return finalize(raw)

def price_percentile(prices, window=60):
    raw = (
        prices
        .rolling(window)
        .rank(pct=True)
        .shift(1)
    )

    return finalize(raw)


# =========================
# Build Factor Matrix
# =========================

def build_factor_matrix(prices, volume=None):

    factors = {
        "mom_1m": momentum_1m(prices),
        "mom_3m": momentum_3m(prices),
        "mom_6m": momentum_6m(prices),
        "mom_12_1": momentum_12_1(prices),
        "mom_accel": momentum_acceleration(prices),

        "rev_5d": reversal_5d(prices),
        "rev_10d": reversal_10d(prices),
        "ma20_dist": distance_ma(prices, 20),
        "ma50_dist": distance_ma(prices, 50),
        "rsi": rsi(prices),
        "bollinger": bollinger_zscore(prices),

        "ema_cross": ema_crossover(prices),
        "ma200_dist": ma200_distance(prices),
        "trend_slope": trend_slope(prices),

        "realized_vol": realized_vol(prices),
        "vol_ratio": volatility_ratio(prices),
        "downside_vol": downside_volatility(prices),
        "parkinson_vol": parkinson_volatility(prices),

        "skew": skewness(prices),
        "kurtosis": kurtosis(prices),
        "autocorr": autocorrelation(prices),

        "breakout": breakout_52w(prices),
        "price_percentile": price_percentile(prices)
    }

    if volume is not None:
        factors.update({
            "relative_volume": relative_volume(volume),
            "volume_trend": volume_trend(volume),
            "dollar_volume": dollar_volume(prices, volume)
        })

    panels = []

    for name, factor in factors.items():
        stacked = factor.stack(future_stack=True)
        stacked.name = name
        panels.append(stacked)

    factor_matrix = pd.concat(
        panels,
        axis=1
    )

    factor_matrix.index.names = [
        "date",
        "ticker"
    ]

    factor_matrix = (
        factor_matrix
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    print(f"Factor matrix shape: {factor_matrix.shape}")

    print(
        f"Factor dates: "
        f"{factor_matrix.index.get_level_values('date').min()} → "
        f"{factor_matrix.index.get_level_values('date').max()}"
    )

    return factor_matrix