"""Stage 1: stream a Lichess dump, keep only 10+0 rapid games with evals and clocks."""
import argparse
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm

from chessblunder.pgn_stream import iter_games, keep_game, open_text_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path or URL of a lichess_db_standard_rated_YYYY-MM.pgn.zst")
    parser.add_argument("--out", default="data/raw/rapid_10p0_evals.pgn.zst")
    parser.add_argument("--max-kept", type=int, default=100_000)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scanned = kept = 0
    stream = open_text_stream(args.source)
    with open(args.out, "wb") as fh, zstd.ZstdCompressor(level=6).stream_writer(fh) as writer, \
            tqdm(unit=" games") as bar:
        for headers, movetext, raw in iter_games(stream):
            scanned += 1
            if keep_game(headers, movetext):
                writer.write(raw.encode("utf-8"))
                kept += 1
                if kept >= args.max_kept:
                    break
            if scanned % 10_000 == 0:
                bar.update(10_000)
                bar.set_postfix(kept=kept)
    print(f"Scanned {scanned:,} games, kept {kept:,}")


if __name__ == "__main__":
    main()
