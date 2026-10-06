"""Stage 3: gradient-boosted baseline (no engine eval as input) with feature-group ablations."""
import argparse

import chess
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from tqdm import tqdm

from chessblunder.features import board_features
from chessblunder.labelling import BLUNDER_DROP

COLUMNS = ["game_id", "ply", "fen", "mover_is_white", "mover_elo", "opp_elo",
           "clock_before", "opp_clock", "time_spent", "wp_before", "wp_drop", "split"]

RATING = ["mover_elo", "opp_elo", "elo_diff"]
BASE = RATING + ["ply", "mover_is_white"]
CLOCK = ["clock_before", "opp_clock", "clock_diff"]


def load_training_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    # Berserk games (arena feature, halves a player's starting clock) look like a ~300s first move
    first = df[df.ply <= 2].groupby("game_id").time_spent.max()
    berserk = set(first[first > 200].index)
    df = df[~df.game_id.isin(berserk)]
    # Skip opening plies, and positions where a blunder is impossible by construction.s
    df = df[(df.ply > 10) & (df.wp_before >= BLUNDER_DROP)].copy()
    df["y"] = (df.wp_drop >= BLUNDER_DROP).astype(int)
    df["elo_diff"] = df.mover_elo - df.opp_elo
    df["clock_diff"] = df.clock_before - df.opp_clock
    df["mover_is_white"] = df.mover_is_white.astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame([board_features(fen) for fen in tqdm(df.fen, unit=" positions")], index=df.index)
    return pd.concat([df.drop(columns=["fen"]), feats], axis=1)


def fit_and_score(name, cols, train, val):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    model = lgb.train(
        params, lgb.Dataset(train[cols], train.y), num_boost_round=2000,
        valid_sets=[lgb.Dataset(val[cols], val.y)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p = model.predict(val[cols], num_iteration=model.best_iteration)
    return model, p, {
        "model": name, "n_feats": len(cols),
        "pr_auc": average_precision_score(val.y, p),
        "roc_auc": roc_auc_score(val.y, p),
        "log_loss": log_loss(val.y, p),
    }


def row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_gain(val: pd.DataFrame, p_without, p_with, mask, n_boot=1000, seed=0):
    """Mean per-row log-loss improvement of the 'with' model, plus a 95% CI that resamples whole games."""
    y = val.y.to_numpy()
    d = row_logloss(y, p_without) - row_logloss(y, p_with)
    per_game = pd.DataFrame({"game": val.game_id.to_numpy()[mask], "d": d[mask]}).groupby("game").d.agg(["sum", "count"])
    sums, counts = per_game["sum"].to_numpy(), per_game["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(per_game), size=(n_boot, len(per_game)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return d[mask].mean(), np.percentile(boots, [2.5, 97.5])


def report_clock_effect(val: pd.DataFrame, preds: dict) -> None:
    print("\nLog-loss gain from adding clock features (positive = clock helps; 95% CI resamples games)")
    pairs = (("A->B (no board)", "A: rating + ply", "B: A + clock"),
             ("C->D (with board)", "C: A + board", "D: A + board + clock"))
    slices = (("all rows", np.ones(len(val), dtype=bool)),
              ("clock <= 120s", (val.clock_before <= 120).to_numpy()),
              ("clock <= 60s", (val.clock_before <= 60).to_numpy()))
    for label, a, b in pairs:
        for slice_name, mask in slices:
            if mask.sum() == 0:
                continue
            gain, (lo, hi) = paired_gain(val, preds[a], preds[b], mask)
            print(f"{label:18s} {slice_name:14s} n={mask.sum():>8,}  gain={gain:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")


def report_by_position_band(val: pd.DataFrame, p: np.ndarray) -> None:
    """Does the model still beat the base rate among positions of similar prior win probability?"""
    print("\nModel D by win probability before the move (is it just detecting 'steep part of the curve'?)")
    band = pd.cut(val.wp_before, [0.15, 0.35, 0.65, 0.85, 1.0], include_lowest=True)
    for name, idx in val.groupby(band, observed=True).groups.items():
        y = val.y.loc[idx]
        if y.nunique() < 2:
            continue
        pr = average_precision_score(y, pd.Series(p, index=val.index).loc[idx])
        print(f"wp_before {str(name):14s} n={len(idx):>8,}  base rate={y.mean():.4f}  PR-AUC={pr:.4f}  lift={pr / y.mean():.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/moves.parquet")
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--val-rows", type=int, default=200_000)
    parser.add_argument("--cache", default="data/processed/baseline_features.parquet")
    args = parser.parse_args()

    df = load_training_table(args.data)
    train = build_features(df[df.split == "train"].sample(min(args.train_rows, (df.split == "train").sum()), random_state=0))
    val = build_features(df[df.split == "val"].sample(min(args.val_rows, (df.split == "val").sum()), random_state=0))
    pd.concat([train.assign(split="train"), val.assign(split="val")]).to_parquet(args.cache)

    board_cols = list(board_features(chess.STARTING_FEN))
    sets = {
        "A: rating + ply": BASE,
        "B: A + clock": BASE + CLOCK,
        "C: A + board": BASE + board_cols,
        "D: A + board + clock": BASE + board_cols + CLOCK,
    }
    base_rate = val.y.mean()
    print(f"\ntrain rows: {len(train):,}  val rows: {len(val):,}  val blunder rate: {base_rate:.4f}")
    print(f"constant-prediction log loss: {log_loss(val.y, np.full(len(val), base_rate)):.4f}")

    results, preds, last_model = [], {}, None
    for name, cols in sets.items():
        last_model, preds[name], res = fit_and_score(name, cols, train, val)
        results.append(res)
    print(pd.DataFrame(results).round(4).to_string(index=False))
    report_clock_effect(val, preds)
    report_by_position_band(val, preds["D: A + board + clock"])
    imp = pd.Series(last_model.feature_importance("gain"), index=sets["D: A + board + clock"])
    
    print("\nTop features (model D, gain):")
    print((imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
