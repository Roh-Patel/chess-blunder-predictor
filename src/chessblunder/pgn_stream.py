"""Stream a Lichess .pgn.zst dump and yield only the rapid games with clock data and
engine analysis."""
import io
import urllib.request
from typing import Iterator

import zstandard as zstd

TARGET_TIME_CONTROL = "600+0"  # 10 minutes, no increment
USABLE_TERMINATIONS = {"Normal", "Time forfeit"}


def open_text_stream(source: str) -> io.TextIOBase:
    """Open a local .pgn.zst file or an https URL as text, decompressing on the fly."""
    if source.startswith("http"):
        raw = urllib.request.urlopen(source)
    else:
        raw = open(source, "rb")
    
    reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def iter_games(stream) -> Iterator[tuple[dict, str, str]]:
    """Yield (headers, movetext, raw_pgn) for every game in the stream."""
    headers: dict[str, str] = {}
    raw_lines: list[str] = []
    movetext: list[str] = []
    for line in stream:
        line = line.rstrip("\n")
        if line == "":
            if movetext:  # a blank line after the moves marks the end of a game
                yield headers, " ".join(movetext), "\n".join(raw_lines) + "\n\n"
                headers, raw_lines, movetext = {}, [], []
            else:  # blank separator between headers and moves
                raw_lines.append(line)
            continue
        raw_lines.append(line)
        if line.startswith("["):
            key, _, value = line[1:-1].partition(" ")
            headers[key] = value.strip('"')
        else:
            movetext.append(line)
    if movetext:  # last game if the file doesn't end with a blank line
        yield headers, " ".join(movetext), "\n".join(raw_lines) + "\n\n"


def keep_game(headers: dict, movetext: str) -> bool:
    """True for rated 10+0 rapid games with engine evals, clocks, and known ratings."""
    return (
        headers.get("TimeControl") == TARGET_TIME_CONTROL
        and headers.get("Event", "").startswith("Rated Rapid")
        and headers.get("Termination") in USABLE_TERMINATIONS
        and headers.get("WhiteElo", "?").isdigit()
        and headers.get("BlackElo", "?").isdigit()
        and "[%eval" in movetext
        and "[%clk" in movetext
    )
