"""Stage 2: parse filtered games into a per-move Parquet dataset with blunder labels."""
import argparse
import hashlib
from pathlib import Path

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chessblunder.extract import game_to_rows
from chessblunder.pgn_stream import open_text_stream

CHUNK_ROWS = 200_000


def assign_split(game_id: str) -> str:
    """Deterministic 80/10/10 split by game, so no game ever spans two splits."""
    bucket = int(hashlib.md5(game_id.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def flush(buffer: list[dict], writer, out_path: str):
    table = pa.Table.from_pylist(buffer)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    buffer.clear()
    return writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="data/raw/rapid_10p0_evals.pgn.zst")
    parser.add_argument("--out", default="data/processed/moves.parquet")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stream = open_text_stream(args.source)
    writer, buffer = None, []
    n_games = n_skipped = 0

    with tqdm(unit=" games") as bar:
        while (game := chess.pgn.read_game(stream)) is not None:
            bar.update(1)
            if game.errors:  # illegal or garbled moves
                n_skipped += 1
                continue
            rows = game_to_rows(game)
            if not rows:
                continue
            split = assign_split(rows[0]["game_id"])
            for row in rows:
                row["split"] = split
            buffer.extend(rows)
            n_games += 1
            if len(buffer) >= CHUNK_ROWS:
                writer = flush(buffer, writer, args.out)
            if args.max_games and n_games >= args.max_games:
                break

    if buffer:
        writer = flush(buffer, writer, args.out)
    if writer:
        writer.close()
    print(f"Wrote {n_games:,} games to {args.out} ({n_skipped:,} skipped)")


if __name__ == "__main__":
    main()