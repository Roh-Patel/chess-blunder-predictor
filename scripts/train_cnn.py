"""Stage 4: residual CNN on board planes + ratings/clock/ply, compared with the LightGBM baseline."""
import argparse
import time

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from chessblunder.encoding import N_PLANES, get_planes, scalar_features
from chessblunder.features import board_features
from chessblunder.model import BlunderNet
from train_baseline import BASE, CLOCK, fit_and_score, load_training_table, paired_gain


def predict(model, planes, scalars, device, batch_size=4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(0, len(planes), batch_size):
            logits = model(planes[i:i + batch_size].float(), scalars[i:i + batch_size])
            out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def train_model(model, train_t, val_t, val_y, args, device) -> dict:
    """Train with early stopping on validation log loss. This gives the best weights."""
    tr_planes, tr_scalars, tr_y = train_t
    va_planes, va_scalars = val_t
    n = len(tr_y)
    steps = n // args.batch_size  # the incomplete last batch is dropped (a different one each epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps)
    loss_fn = nn.BCEWithLogitsLoss()
    best_ll, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=device)
        running = torch.zeros((), device=device)
        for step in range(steps):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(tr_planes[idx].float(), tr_scalars[idx])
            loss = loss_fn(logits.float(), tr_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach()
        p = predict(model, va_planes, va_scalars, device)
        ll, pr = log_loss(val_y, p), average_precision_score(val_y, p)
        print(f"epoch {epoch:2d}  train loss {running.item() / steps:.4f}  val log loss {ll:.4f}  "
              f"val PR-AUC {pr:.4f}  ({time.perf_counter() - t0:.0f}s)")
        if ll < best_ll:
            best_ll, bad_epochs = ll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break
    return best_state


def lightgbm_reference(val: pd.DataFrame, use_clock: bool, cache: str) -> np.ndarray:
    """Refit the matching LightGBM model on the cached baseline features; predictions aligned to `val`."""
    cached = pd.read_parquet(cache)
    c_train, c_val = cached[cached.split == "train"], cached[cached.split == "val"]
    ok = val.index.isin(c_val.index).all()
    if ok:
        aligned = c_val.loc[val.index]
        ok = (aligned.game_id.to_numpy() == val.game_id.to_numpy()).all() and (aligned.ply.to_numpy() == val.ply.to_numpy()).all()
    if not ok:
        raise SystemExit("Validation rows differ from the cached baseline features; rerun train_baseline.py first.")
    cols = BASE + list(board_features(chess.STARTING_FEN)) + (CLOCK if use_clock else [])
    _, p, _ = fit_and_score("LightGBM", cols, c_train, c_val)
    return pd.Series(p, index=c_val.index).loc[val.index].to_numpy()


def metrics_row(name: str, y, p) -> dict:
    return {"model": name, "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p)}


def compare_by_band(val: pd.DataFrame, p_cnn, p_lgb) -> None:
    print("\nPR-AUC by win probability before the move")
    edges = [0.15, 0.35, 0.65, 0.85, 1.0]
    codes = pd.cut(val.wp_before, edges, labels=False, include_lowest=True).to_numpy()
    y = val.y.to_numpy()
    for k in range(len(edges) - 1):
        m = codes == k
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        print(f"wp_before {edges[k]:.2f}-{edges[k + 1]:.2f}  n={m.sum():>8,}  base rate={y[m].mean():.4f}  "
              f"CNN={average_precision_score(y[m], p_cnn[m]):.4f}  LightGBM={average_precision_score(y[m], p_lgb[m]):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/moves.parquet")
    parser.add_argument("--cache", default="data/processed/baseline_features.parquet")
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--val-rows", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-clock", action="store_true", help="leave the clock features out")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_clock = not args.no_clock
    tag = "clock" if use_clock else "noclock"
    print(f"device: {device}   clock features: {use_clock}")

    # Same rows as the baseline: same filters and the same seeded sampling
    df = load_training_table(args.data)
    train = df[df.split == "train"].sample(min(args.train_rows, (df.split == "train").sum()), random_state=0)
    val = df[df.split == "val"].sample(min(args.val_rows, (df.split == "val").sum()), random_state=0)
    del df
    print(f"train rows: {len(train):,}  val rows: {len(val):,}  val blunder rate: {val.y.mean():.4f}")

    tr_x, va_x = scalar_features(train, use_clock), scalar_features(val, use_clock)
    mu, sd = tr_x.mean(axis=0), tr_x.std(axis=0) + 1e-6  # standardise with training statistics only
    tr_x, va_x = (tr_x - mu) / sd, (va_x - mu) / sd
    tr_planes, va_planes = get_planes(train, "train"), get_planes(val, "val")

    # Everything should fit in GPU memory (~1.6 GB of uint8 planes), so batches are sliced on GPU directly
    train_t = (torch.from_numpy(tr_planes).to(device), torch.from_numpy(tr_x).to(device),
               torch.tensor(train.y.to_numpy(), dtype=torch.float32, device=device))
    val_t = (torch.from_numpy(va_planes).to(device), torch.from_numpy(va_x).to(device))
    val_y = val.y.to_numpy()

    model = BlunderNet(N_PLANES, tr_x.shape[1], args.channels, args.blocks).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    best_state = train_model(model, train_t, val_t, val_y, args, device)
    model.load_state_dict(best_state)
    torch.save(best_state, f"models/cnn_{tag}.pt")

    p_cnn = predict(model, *val_t, device)
    np.save(f"data/processed/cnn_val_preds_{tag}.npy", p_cnn)
    p_lgb = lightgbm_reference(val, use_clock, args.cache)

    print(f"\nconstant-prediction log loss: {log_loss(val_y, np.full(len(val_y), val_y.mean())):.4f}")
    print(pd.DataFrame([metrics_row(f"CNN ({tag})", val_y, p_cnn),
                        metrics_row(f"LightGBM ({tag})", val_y, p_lgb)]).round(4).to_string(index=False))

    print("\nLog-loss gain of CNN over LightGBM (positive = CNN better; 95% CI resamples games)")
    for label, mask in (("all rows", np.ones(len(val), dtype=bool)),
                        ("clock <= 120s", (val.clock_before <= 120).to_numpy()),
                        ("clock <= 60s", (val.clock_before <= 60).to_numpy())):
        gain, (lo, hi) = paired_gain(val, p_lgb, p_cnn, mask)
        print(f"{label:14s} n={mask.sum():>8,}  gain={gain:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")
    compare_by_band(val, p_cnn, p_lgb)


if __name__ == "__main__":
    main()
