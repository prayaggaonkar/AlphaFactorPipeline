import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config import *


def information_coefficient(predictions, actual):
    daily_ic=[]
    dates = predictions.index.get_level_values("date").unique()

    for date in dates:
        if date not in actual.index.get_level_values("date"):
            continue

        pred = predictions.xs(date, level="date")
        act  = actual.xs(date, level="date")

        common = pred.index.intersection(act.index)

        if len(common)<10:
            continue
        
        ic,_ = spearmanr(
            pred.loc[common],
            act.loc[common]
        )

        daily_ic.append(ic)

    return np.nanmean(daily_ic)


def train_and_evaluate(factor_matrix: pd.DataFrame,
                       fwd_returns: pd.DataFrame):

    # Stack forward returns and force index names
    fwd_stacked = fwd_returns.stack(future_stack=True)
    fwd_stacked.index.names = ["date", "ticker"]
    fwd_stacked.name = "target"

    # Force factor matrix index names to match
    factor_matrix = factor_matrix.copy()
    factor_matrix.index.names = ["date", "ticker"]

    data = factor_matrix.join(fwd_stacked, how="inner").dropna()

    print(f"Joined data shape: {data.shape}")
    print(f"Date range: {data.index.get_level_values('date').min()} → {data.index.get_level_values('date').max()}")
    print(f"Unique dates: {data.index.get_level_values('date').nunique()}")

    if data.empty:
        raise ValueError("Data is empty after join — index names still don't match.")

    feature_cols = [c for c in data.columns if c != "target"]
    dates = data.index.get_level_values("date").unique().sort_values()

    all_predictions = []
    ic_by_fold = []

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.02,
        num_leaves=24,
        min_child_samples=50,
        reg_lambda=2.0,
        reg_alpha=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    '''
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
    )
    '''

    for i in range(TRAIN_WINDOW, len(dates) - TEST_WINDOW, 21):
        train_dates = dates[i - TRAIN_WINDOW: i]
        test_dates = dates[i + FORWARD_DAYS:i + FORWARD_DAYS + TEST_WINDOW]

        train = data[data.index.get_level_values("date").isin(train_dates)]
        test  = data[data.index.get_level_values("date").isin(test_dates)]

        if len(train) < 500 or len(test) < 50:
            continue

        X_train = train[feature_cols]
        y_train = train["target"]
        X_test  = test[feature_cols]
        y_test  = test["target"]

        #groups = train.groupby(level="date").size()

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
    icir      = mean_ic / np.nanstd(ic_by_fold)

    print(f"\nMean IC: {mean_ic:.4f}")
    print(f"ICIR:    {icir:.2f}  (target > 0.5)")

    print("\nComputing SHAP values...")
    import os
    os.makedirs("models", exist_ok=True)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("models/shap_importance.png", dpi=150)
    plt.close()
    print("SHAP chart saved to models/shap_importance.png")

    return all_preds, model, mean_ic