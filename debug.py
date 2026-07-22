# debug_factors.py
from data_loader import load_prices, compute_returns
from factor_lib import *
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

prices = load_prices()
daily_ret, fwd_ret = compute_returns(prices)

factors = {
    "mom_1m":        momentum_1m(prices),
    "mom_3m":        momentum_3m(prices),
    "mom_6m":        momentum_6m(prices),
    "mom_12_1":      momentum_12_1(prices),
    "rev_5d":        reversal_5d(prices),
    "rev_10d":       reversal_10d(prices),
    "ma20_dist":     distance_ma(prices, 20),
    "ma50_dist":     distance_ma(prices, 50),
    "rsi":           rsi(prices),
    "bollinger":     bollinger_zscore(prices),
    "realized_vol":  realized_vol(prices),
    "breakout":      breakout_52w(prices),
}

print(f"Forward return period: {FORWARD_DAYS} days")
print(f"{'Factor':<20} {'Mean IC':>10} {'ICIR':>10} {'Positive%':>12}")
print("-" * 56)

results = {}
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
    mean_ic = ics.mean()
    icir = mean_ic / ics.std() if ics.std() > 0 else 0
    pos = (ics > 0).mean() * 100
    results[name] = mean_ic
    print(f"{name:<20} {mean_ic:>10.4f} {icir:>10.2f} {pos:>11.1f}%")

print("\n── Top positive factors ──")
for name, ic in sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {ic:.4f}")