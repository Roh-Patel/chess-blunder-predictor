"""Average the saved predictions of several CNN runs and compare the average with a reference run and with LightGBM."""
import argparse

import numpy as np
import pandas as pd

from train_baseline import load_training_table
from train_cnn import lightgbm_reference, metrics_row, print_gains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True, help="tags of the CNN runs to average")
    parser.add_argument("--ref", default=None, help="tag of a single run to compare against (default: first tag)")
    parser.add_argument("--data", default="data/processed/moves.parquet")
    parser.add_argument("--cache", default="data/processed/baseline_features.parquet")
    parser.add_argument("--val-rows", type=int, default=1_000_000)  # must match the CNN runs
    args = parser.parse_args()

    df = load_training_table(args.data)
    val = df[df.split == "val"].sample(min(args.val_rows, (df.split == "val").sum()), random_state=0)
    del df
    y = val.y.to_numpy()

    def load(tag: str) -> np.ndarray:
        p = np.load(f"data/processed/cnn_val_preds_{tag}.npy")
        if len(p) != len(val):
            raise SystemExit(f"{tag}: {len(p):,} predictions but {len(val):,} validation rows (different --val-rows?)")
        return p

    preds = {tag: load(tag) for tag in args.tags}
    ensemble = np.mean(list(preds.values()), axis=0)  # average of probabilities
    ref_tag = args.ref or args.tags[0]
    print(pd.DataFrame([metrics_row(t, y, p) for t, p in preds.items()]
                       + [metrics_row(f"mean of {len(preds)}", y, ensemble)]).round(4).to_string(index=False))

    print_gains(f"Log-loss gain of the average over {ref_tag}", val, load(ref_tag), ensemble)
    print_gains("Log-loss gain of the average over LightGBM", val, lightgbm_reference(val, True, args.cache), ensemble)


if __name__ == "__main__":
    main()
