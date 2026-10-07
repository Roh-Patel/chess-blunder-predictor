"""Compare XGBoost with LightGBM on the cached baseline features (same rows, features, early stopping)."""
import time

import chess
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from chessblunder.features import board_features
from train_baseline import BASE, CLOCK, fit_and_score, paired_gain


def fit_xgb(cols, train, val):
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50, random_state=0, n_jobs=-1,
    )
    model.fit(train[cols], train.y, eval_set=[(val[cols], val.y)], verbose=False)
    return model.predict_proba(val[cols], iteration_range=(0, model.best_iteration + 1))[:, 1]


def score_row(name, lib, p, y, seconds):
    return {"model": name, "lib": lib, "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p), "seconds": round(seconds)}


def main() -> None:
    df = pd.read_parquet("data/processed/baseline_features.parquet")
    train = df[df.split == "train"]
    val = df[df.split == "val"].reset_index(drop=True)

    board_cols = list(board_features(chess.STARTING_FEN))
    sets = {
        "A: rating + ply": BASE,
        "B: A + clock": BASE + CLOCK,
        "C: A + board": BASE + board_cols,
        "D: A + board + clock": BASE + board_cols + CLOCK,
    }
    rows, preds = [], {}
    for name, cols in sets.items():
        t0 = time.perf_counter()
        _, p_lgb, _ = fit_and_score(name, cols, train, val)
        t1 = time.perf_counter()
        p_xgb = fit_xgb(cols, train, val)
        t2 = time.perf_counter()
        rows += [score_row(name, "lightgbm", p_lgb, val.y, t1 - t0),
                 score_row(name, "xgboost", p_xgb, val.y, t2 - t1)]
        preds[name] = (p_lgb, p_xgb)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    p_lgb, p_xgb = preds["D: A + board + clock"]
    print("\nModel D: log-loss gain of XGBoost over LightGBM (positive = XGBoost better; 95% CI resamples games)")
    for label, mask in (("all rows", np.ones(len(val), dtype=bool)),
                        ("clock <= 60s", (val.clock_before <= 60).to_numpy())):
        gain, (lo, hi) = paired_gain(val, p_lgb, p_xgb, mask)
        print(f"{label:14s} n={mask.sum():>8,}  gain={gain:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")


if __name__ == "__main__":
    main()
