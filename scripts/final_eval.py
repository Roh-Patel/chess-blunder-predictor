"""Final evaluation on the held-out TEST split: LightGBM vs the hybrid-CNN seed ensemble."""
import argparse

import chess
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

from chessblunder.encoding import N_PLANES, get_planes, scalar_features
from chessblunder.features import board_features
from chessblunder.labelling import BLUNDER_DROP
from chessblunder.model import BlunderNet
from train_baseline import BASE, CLOCK, attach_fens, build_features, fit_and_score, load_training_table
from train_cnn import check_rows, compare_by_band, metrics_row, predict, print_gains


def build_test_inputs(args):
    """Everything both models need for the test rows, built exactly as for training and validation."""
    df = load_training_table(args.data)
    train = df[df.split == "train"].sample(min(args.train_rows, (df.split == "train").sum()), random_state=0)
    test = df[df.split == "test"].sample(min(args.test_rows, (df.split == "test").sum()), random_state=0)
    del df

    cached = pd.read_parquet(args.cache)
    c_train, c_val = cached[cached.split == "train"], cached[cached.split == "val"]
    check_rows(train, c_train, "Training")
    board_cols = list(board_features(chess.STARTING_FEN))

    # Standardisation statistics from the training rows only
    tr_x = np.hstack([scalar_features(train, True), c_train.loc[train.index, board_cols].to_numpy(dtype=np.float32)])
    mu, sd = tr_x.mean(axis=0), tr_x.std(axis=0) + 1e-6
    np.savez("models/scaler.npz", mu=mu, sd=sd)  # the demo needs these too

    test = attach_fens(test, args.data)
    test_full = build_features(test)  # adds the hand-crafted board features
    x_test = np.hstack([scalar_features(test_full, True), test_full[board_cols].to_numpy(dtype=np.float32)])
    x_test = ((x_test - mu) / sd).astype(np.float32)
    planes = get_planes(test, "test")

    cols = BASE + board_cols + CLOCK
    model, _, _ = fit_and_score("LightGBM", cols, c_train, c_val)  # trained on train, early-stopped on val
    p_lgb = model.predict(test_full[cols], num_iteration=model.best_iteration)
    return test_full, x_test, planes, p_lgb


def cnn_predictions(args, planes, x_test, device) -> dict:
    thresholds = sorted(set(args.thresholds))
    primary = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - BLUNDER_DROP))
    planes_t, x_t = torch.from_numpy(planes).to(device), torch.from_numpy(x_test).to(device)
    preds = {}
    for tag in args.tags:
        state = torch.load(f"models/cnn_{tag}.pt", map_location=device)
        model = BlunderNet(N_PLANES, x_test.shape[1], channels=args.channels, blocks=args.blocks,
                           n_outputs=len(thresholds)).to(device)
        model.load_state_dict(state)
        preds[tag] = predict(model, planes_t, x_t, device, primary)
    return preds


def report(test: pd.DataFrame, p_lgb: np.ndarray, preds: dict) -> None:
    y = test.y.to_numpy()
    p_ens = np.mean(list(preds.values()), axis=0)  # average of probabilities
    np.save("data/processed/test_preds_ensemble.npy", p_ens)
    print(f"test rows: {len(test):,}  blunder rate: {y.mean():.4f}  "
          f"constant-prediction log loss: {log_loss(y, np.full(len(y), y.mean())):.4f}")
    rows = ([metrics_row("LightGBM", y, p_lgb)] + [metrics_row(t, y, p) for t, p in preds.items()]
            + [metrics_row(f"CNN mean of {len(preds)}", y, p_ens)])
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print_gains("Log-loss gain of the CNN ensemble over LightGBM", test, p_lgb, p_ens)
    compare_by_band(test, p_ens, p_lgb)

    frame = pd.DataFrame({"pred": p_ens, "obs": y}, index=test.index)
    print("\nCalibration of the ensemble: deciles of predicted probability")
    deciles = pd.qcut(frame.pred, 10, duplicates="drop")
    print(frame.groupby(deciles, observed=True).agg(n=("obs", "size"), mean_pred=("pred", "mean"),
                                                    observed=("obs", "mean")).round(4).to_string())
    print("\nPredicted vs observed blunder rate by clock remaining")
    clock_bins = pd.cut(test.clock_before, [0, 30, 60, 120, 300, 601])
    print(frame.groupby(clock_bins, observed=True).agg(n=("obs", "size"), mean_pred=("pred", "mean"),
                                                       observed=("obs", "mean")).round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/moves.parquet")
    parser.add_argument("--cache", default="data/processed/baseline_features.parquet")
    parser.add_argument("--train-rows", type=int, default=4_000_000)  # must match the training runs
    parser.add_argument("--test-rows", type=int, default=2_000_000)  # more than exists, so all test rows are used
    parser.add_argument("--tags", nargs="+", default=[f"big_hybrid_s{i}" for i in range(5)])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test, x_test, planes, p_lgb = build_test_inputs(args)
    preds = cnn_predictions(args, planes, x_test, device)
    report(test, p_lgb, preds)


if __name__ == "__main__":
    main()
