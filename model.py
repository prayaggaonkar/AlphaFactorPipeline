## model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config import *

def information_coefficient(predictions: pd.Series,
                             actual: pd.Series) -> float:
    """
    IC = Spearman rank correlation between predictions and actual returns.
    Measures how well your model ranks stocks (not the exact values).
    IC > 0.05 is good. IC > 0.08 is excellent.
    """
    common = predictions.index.intersection(actual.index)
    if len(common) < 10:
        return np.nan
    ic, _ = spearmanr(predictions[common], actual[common])
    return ic

def train_and_evaluate(factor_matrix: pd.DataFrame,
                       fwd_returns: pd.DataFrame):
    """
    Walk-forward training:
    - Train on 252 days of data
    - Predict the next 63 days
    - Step forward 21 days and repeat
    """
    ## Align forward returns with factor matrix
    fwd_stacked = fwd_returns.stack()
    fwd_stacked.index.names = ["date", "ticker"]
    fwd_stacked.name = "target"

    ## Combine features and target into one DataFrame
    data = factor_matrix.join(fwd_stacked, how="inner").dropna()

    feature_cols = [c for c in data.columns if c != "target"]
    dates = data.index.get_level_values("date").unique().sort_values()

    all_predictions = []
    ic_by_fold = []

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,   # at least 50 samples per leaf
        reg_lambda=1.0,         # regularization — prevents overfitting
        subsample=0.8,          # use 80% of rows per tree
        colsample_bytree=0.8,   # use 80% of features per tree
        random_state=42,
        verbose=-1              # suppress output
    )

    ## Walk-forward loop
    for i in range(TRAIN_WINDOW, len(dates) - TEST_WINDOW, 21):
        train_dates = dates[i - TRAIN_WINDOW : i]
        test_dates  = dates[i + 5 : i + 5 + TEST_WINDOW]  # 5-day gap!

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

        ## Compute IC for this fold
        ic = information_coefficient(preds, y_test)
        ic_by_fold.append(ic)
        all_predictions.append(preds)

        print(f"Fold ending {test_dates[-1].date()} — IC: {ic:.4f}")

    all_preds = pd.concat(all_predictions)
    mean_ic = np.nanmean(ic_by_fold)
    icir    = np.nanmean(ic_by_fold) / np.nanstd(ic_by_fold)

    print(f"\nMean IC: {mean_ic:.4f}")
    print(f"ICIR:    {icir:.2f}  (target > 0.5)")

    ## SHAP: explain the final model
    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      show=False)
    plt.tight_layout()
    plt.savefig("models/shap_importance.png", dpi=150)
    print("SHAP chart saved to models/shap_importance.png")

    return all_preds, model, mean_ic