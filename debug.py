# debug.py
from data_loader import load_prices, compute_returns
from factor_lib import *
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

prices = load_prices()
daily_ret, fwd_ret = compute_returns(prices)

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

print(f"{'Factor':<20} {'Mean IC':>10} {'ICIR':>10} {'Positive %':>12}")
print("-" * 55)

for name, factor in factors.items():
    ics = []
    for date in factor.index[252::5]:
        f = factor.loc[date].dropna()
        if date not in fwd_ret.index:
            continue
        r = fwd_ret.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 20:
            continue
        ic, _ = spearmanr(f[common], r[common])
        ics.append(ic)
    ics = pd.Series(ics).dropna()
    if len(ics) == 0:
        print(f"{name:<20} {'no data':>10}")
        continue
    print(f"{name:<20} {ics.mean():>10.4f} {(ics.mean()/ics.std()):>10.2f} {(ics>0).mean()*100:>11.1f}%")