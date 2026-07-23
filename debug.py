import pandas as pd
import numpy as np
import lightgbm as lgb

from model import information_coefficient
from backtest import build_portfolio
from data_loader import load_prices
from factor_lib import build_factor_matrix
from config import *


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

prices = load_prices()

factor_matrix = build_factor_matrix(prices)

# Forward returns target
fwd_returns = (
    prices
    .pct_change(FORWARD_DAYS)
    .shift(-FORWARD_DAYS)
)

fwd_returns = fwd_returns.loc[factor_matrix.index
                              .get_level_values("date")
                              .unique()]


target = fwd_returns.stack(future_stack=True)
target.index.names = ["date", "ticker"]
target.name = "target"


data = factor_matrix.join(target, how="inner").dropna()


print("\nFull dataset:")
print(data.index.get_level_values("date").min(),
      "→",
      data.index.get_level_values("date").max())


# ─────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────

TEST_START = pd.Timestamp("2024-01-01")


dates = (
    data.index
    .get_level_values("date")
    .unique()
    .sort_values()
)


feature_cols = [
    c for c in data.columns
    if c != "target"
]


# ─────────────────────────────────────────────
# Walk-forward 2024
# ─────────────────────────────────────────────

predictions = []


model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=50,
    reg_lambda=1.0,
    reg_alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)


test_dates = dates[dates >= TEST_START]


for i, date in enumerate(test_dates):

    # only train using data before this date
    train_dates = dates[dates < date]

    train = data[
        data.index.get_level_values("date")
        .isin(train_dates)
    ]


    # predict next rebalance period
    future_dates = test_dates[i:i+FORWARD_DAYS]


    test = data[
        data.index.get_level_values("date")
        .isin(future_dates)
    ]


    if len(train) < 500 or len(test) < 50:
        continue


    X_train = train[feature_cols]
    y_train = train["target"]

    X_test = test[feature_cols]


    model.fit(
        X_train,
        y_train
    )


    preds = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="prediction"
    )


    predictions.append(preds)

    print(
        f"Trained through {date.date()} | "
        f"Predicted {future_dates[0].date()} → {future_dates[-1].date()}"
    )


# combine all predictions

predictions = pd.concat(predictions)


print("\nPredictions:")
print(
    predictions.index.get_level_values("date").min(),
    "→",
    predictions.index.get_level_values("date").max()
)


# ─────────────────────────────────────────────
# IC on unseen 2024
# ─────────────────────────────────────────────

actual = data["target"]

ic = information_coefficient(
    predictions,
    actual
)

print(f"\n2024 Mean IC: {ic:.4f}")


# ─────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────

daily_returns = prices.pct_change()

daily_returns = daily_returns.loc["2024-01-01":]


metrics, returns, cumulative = build_portfolio(
    predictions,
    daily_returns
)


print("\n2024 OUT-OF-SAMPLE RESULTS")
for k,v in metrics.items():
    print(f"{k:<20}: {v}")