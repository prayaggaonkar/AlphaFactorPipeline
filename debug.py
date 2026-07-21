# debug_factors.py
from data_loader import load_prices, compute_returns
from factor_lib import *
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

prices = load_prices()
daily_ret, fwd_ret = compute_returns(prices)

factors = {
    "mom_1m":         momentum_1m(prices),
    "mom_3m":         momentum_3m(prices),
    "mom_12m_skip1m": momentum_12m_skip1m(prices),
    "reversal_1w":    reversal_1w(prices),
    "dist_ma20":      distance_from_ma(prices, 20),
    "dist_ma60":      distance_from_ma(prices, 60),
    "vol_21d":        realized_vol(prices, 21),
}

print(f"{'Factor':<20} {'Mean IC':>10} {'ICIR':>10} {'Positive %':>12}")
print("-" * 55)

for name, factor in factors.items():
    ics = []
    for date in factor.index[252::5]:   # sample every 5 days
        f = factor.loc[date].dropna()
        r = fwd_ret.loc[date].dropna() if date in fwd_ret.index else pd.Series()
        common = f.index.intersection(r.index)
        if len(common) < 20:
            continue
        ic, _ = spearmanr(f[common], r[common])
        ics.append(ic)
    ics = pd.Series(ics).dropna()
    print(f"{name:<20} {ics.mean():>10.4f} {(ics.mean()/ics.std()):>10.2f} {(ics>0).mean()*100:>11.1f}%")