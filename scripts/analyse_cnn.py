"""Analyse saved CNN predictions: does the CNN use the clock, and does blending with LightGBM help?"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from train_baseline import load_training_table, paired_gain
from train_cnn import lightgbm_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/moves.parquet")
    parser.add_argument("--cache", default="data/processed/baseline_features.parquet")
    parser.add_argument("--val-rows", type=int, default=1_000_000)  # must match the CNN runs
    parser.add_argument("--tag", default="clock", help="tag of the CNN run with clock features")
    parser.add_argument("--tag-noclock", default="noclock", help="tag of the matching run without clock features")
    args = parser.parse_args()

    df = load_training_table(args.data)
    val = df[df.split == "val"].sample(min(args.val_rows, (df.split == "val").sum()), random_state=0)
    del df
    y = val.y.to_numpy()
    p_clock = np.load(f"data/processed/cnn_val_preds_{args.tag}.npy")
    p_noclock = np.load(f"data/processed/cnn_val_preds_{args.tag_noclock}.npy")
    p_lgb = lightgbm_reference(val, True, args.cache)

    slices = (("all rows", np.ones(len(val), dtype=bool)),
              ("clock <= 120s", (val.clock_before <= 120).to_numpy()),
              ("clock <= 60s", (val.clock_before <= 60).to_numpy()))

    print("Does the CNN use the clock? Log-loss gain of CNN-with-clock over CNN-without (positive = clock helps)")
    for name, mask in slices:
        gain, (lo, hi) = paired_gain(val, p_noclock, p_clock, mask)
        print(f"{name:14s} n={mask.sum():>8,}  gain={gain:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")

    blend = 0.5 * p_clock + 0.5 * p_lgb  # one fixed weight, so nothing is tuned on the validation set
    print("\nEqual-weight blend of CNN (clock) and LightGBM (clock)")
    for label, p in (("CNN", p_clock), ("LightGBM", p_lgb), ("blend", blend)):
        print(f"{label:9s} PR-AUC={average_precision_score(y, p):.4f}  log loss={log_loss(y, p):.4f}")
    print("Log-loss gain of the blend over LightGBM alone (positive = blend better)")
    for name, mask in slices:
        gain, (lo, hi) = paired_gain(val, p_lgb, blend, mask)
        print(f"{name:14s} n={mask.sum():>8,}  gain={gain:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")


if __name__ == "__main__":
    main()
