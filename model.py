'''
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config import *
import os


def information_coefficient(predictions, actual):
    daily_ic = []
    dates = predictions.index.get_level_values("date").unique()
    for date in dates:
        if date not in actual.index.get_level_values("date"):
            continue
        pred = predictions.xs(date, level="date")
        act  = actual.xs(date, level="date")
        common = pred.index.intersection(act.index)
        if len(common) < 10:
            continue
        ic, _ = spearmanr(pred.loc[common], act.loc[common])
        daily_ic.append(ic)
    return np.nanmean(daily_ic)


def train_and_evaluate(factor_matrix: pd.DataFrame,
                       fwd_returns: pd.DataFrame):

    fwd_stacked = fwd_returns.stack(future_stack=True)
    fwd_stacked.index.names = ["date", "ticker"]
    fwd_stacked.name = "target"

    factor_matrix = factor_matrix.copy()
    factor_matrix.index.names = ["date", "ticker"]

    data = factor_matrix.join(fwd_stacked, how="inner").dropna()

    print(f"Joined data shape: {data.shape}")
    print(f"Date range: {data.index.get_level_values('date').min()} → {data.index.get_level_values('date').max()}")
    print(f"Unique dates: {data.index.get_level_values('date').nunique()}")

    if data.empty:
        raise ValueError("Data is empty after join.")

    feature_cols = [c for c in data.columns if c != "target"]
    dates = data.index.get_level_values("date").unique().sort_values()

    all_predictions = []
    ic_by_fold = []

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

    for i in range(TRAIN_WINDOW, len(dates) - TEST_WINDOW, 21):
        train_dates = dates[i - TRAIN_WINDOW: i]
        test_dates  = dates[i + FORWARD_DAYS: i + FORWARD_DAYS + TEST_WINDOW]

        if len(test_dates) == 0:
            continue

        train = data[data.index.get_level_values("date").isin(train_dates)]
        test  = data[data.index.get_level_values("date").isin(test_dates)]

        if len(train) < 500 or len(test) < 50:
            continue

        X_train = train[feature_cols]
        y_train = train["target"]
        X_test  = test[feature_cols]
        y_test  = test["target"]

        model.fit(X_train, y_train)

        preds = pd.Series(
            model.predict(X_test),
            index=X_test.index,
            name="prediction"
        )

        ic = information_coefficient(preds, y_test)
        ic_by_fold.append(ic)
        all_predictions.append(preds)
        print(f"Fold ending {test_dates[-1].date()} — IC: {ic:.4f}")

    if not all_predictions:
        raise ValueError(f"No folds generated. Have {len(dates)} dates, need > {TRAIN_WINDOW + TEST_WINDOW}.")

    all_preds = pd.concat(all_predictions)
    mean_ic   = np.nanmean(ic_by_fold)
    icir      = mean_ic / np.nanstd(ic_by_fold) if np.nanstd(ic_by_fold) > 0 else 0

    print(f"\nMean IC: {mean_ic:.4f}")
    print(f"ICIR:    {icir:.2f}  (target > 0.5)")

    print("\nComputing SHAP values...")
    os.makedirs("models", exist_ok=True)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("models/shap_importance.png", dpi=150)
    plt.close()
    print("SHAP chart saved to models/shap_importance.png")

    return all_preds, model, mean_ic
'''

## model.py — IC-weighted composite (no LightGBM)
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from config import *


def information_coefficient(predictions, actual):
    daily_ic = []
    dates = predictions.index.get_level_values("date").unique()
    for date in dates:
        if date not in actual.index.get_level_values("date"):
            continue
        pred = predictions.xs(date, level="date")
        act  = actual.xs(date, level="date")
        common = pred.index.intersection(act.index)
        if len(common) < 10:
            continue
        ic, _ = spearmanr(pred.loc[common], act.loc[common])
        daily_ic.append(ic)
    return np.nanmean(daily_ic)


def train_and_evaluate(factor_matrix: pd.DataFrame,
                       fwd_returns: pd.DataFrame):

    fwd_stacked = fwd_returns.stack(future_stack=True)
    fwd_stacked.index.names = ["date", "ticker"]
    fwd_stacked.name = "target"

    factor_matrix = factor_matrix.copy()
    factor_matrix.index.names = ["date", "ticker"]

    data = factor_matrix.join(fwd_stacked, how="inner").dropna()

    print(f"Joined data shape: {data.shape}")
    print(f"Date range: {data.index.get_level_values('date').min()} → {data.index.get_level_values('date').max()}")

    if data.empty:
        raise ValueError("Data is empty after join.")

    feature_cols = [c for c in data.columns if c != "target"]
    dates = data.index.get_level_values("date").unique().sort_values()

    all_predictions = []
    ic_by_fold = []

    for i in range(TRAIN_WINDOW, len(dates) - TEST_WINDOW, 21):
        train_dates = dates[i - TRAIN_WINDOW: i]
        test_dates  = dates[i + FORWARD_DAYS: i + FORWARD_DAYS + TEST_WINDOW]

        if len(test_dates) == 0:
            continue

        train = data[data.index.get_level_values("date").isin(train_dates)]
        test  = data[data.index.get_level_values("date").isin(test_dates)]

        if len(train) < 500 or len(test) < 50:
            continue

        # Compute IC for each factor on training data
        factor_ics = {}
        for col in feature_cols:
            col_ics = []
            for date in train_dates:
                if date not in train.index.get_level_values("date"):
                    continue
                slice_ = train.xs(date, level="date")[[col, "target"]].dropna()
                if len(slice_) < 10:
                    continue
                ic, _ = spearmanr(slice_[col], slice_["target"])
                col_ics.append(ic)
            factor_ics[col] = np.nanmean(col_ics) if col_ics else 0.0

        # Only use factors with positive IC — zero-weight the rest
        weights = {col: max(ic, 0) for col, ic in factor_ics.items()}
        total = sum(weights.values())

        if total == 0:
            # All factors negative — equal weight as fallback
            weights = {col: 1/len(feature_cols) for col in feature_cols}
            total = 1.0

        weights = {col: w/total for col, w in weights.items()}

        print(f"  Fold weights: " + " | ".join(f"{k}:{v:.2f}" for k,v in sorted(weights.items(), key=lambda x: -x[1])[:3]))

        # Compute weighted composite score on test data
        X_test = test[feature_cols]
        composite = sum(X_test[col] * w for col, w in weights.items())
        composite.name = "prediction"

        y_test = test["target"]
        ic = information_coefficient(composite, y_test)
        ic_by_fold.append(ic)
        all_predictions.append(composite)
        print(f"Fold ending {test_dates[-1].date()} — IC: {ic:.4f}")

    if not all_predictions:
        raise ValueError(f"No folds generated.")

    all_preds = pd.concat(all_predictions)
    mean_ic   = np.nanmean(ic_by_fold)
    icir      = mean_ic / np.nanstd(ic_by_fold) if np.nanstd(ic_by_fold) > 0 else 0

    print(f"\nMean IC: {mean_ic:.4f}")
    print(f"ICIR:    {icir:.2f}  (target > 0.5)")

    # Dummy model for compatibility with main.py
    class DummyModel:
        pass

    return all_preds, DummyModel(), mean_ic