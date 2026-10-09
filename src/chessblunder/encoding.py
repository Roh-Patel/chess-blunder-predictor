"""Encode positions as CNN inputs: board planes (always from the mover's point of view) and scalar features."""
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

N_PLANES = 17  # 12 piece planes + 4 castling-rights planes + 1 en-passant plane
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _mask_to_plane(mask: int) -> np.ndarray:
    """Convert a 64-bit bitboard into an 8x8 array of 0/1 (row = rank, column = file)."""
    raw = np.frombuffer(mask.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(8, 8)


def encode_fen(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        board = board.mirror()  # flip the board and swap colours: the mover is now 'White' at the bottom
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.uint8)

    for i, piece_type in enumerate(_PIECE_TYPES):
        planes[i] = _mask_to_plane(board.pieces_mask(piece_type, chess.WHITE))      # mover's pieces
        planes[6 + i] = _mask_to_plane(board.pieces_mask(piece_type, chess.BLACK))  # opponent's pieces
    planes[12] = board.has_kingside_castling_rights(chess.WHITE)
    planes[13] = board.has_queenside_castling_rights(chess.WHITE)
    planes[14] = board.has_kingside_castling_rights(chess.BLACK)
    planes[15] = board.has_queenside_castling_rights(chess.BLACK)

    if board.ep_square is not None:
        planes[16, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1

    return planes


def encode_many(fens: list[str]) -> np.ndarray:
    out = np.empty((len(fens), N_PLANES, 8, 8), dtype=np.uint8)
    for i, fen in enumerate(tqdm(fens, unit=" positions")):
        out[i] = encode_fen(fen)
    return out


def get_planes(df: pd.DataFrame, split: str, cache_dir: str = "data/processed") -> np.ndarray:
    """Encode a dataframe's FENs, caching the result on disk."""
    path = Path(cache_dir) / f"planes_{split}_{len(df)}.npy"
    if path.exists():
        return np.load(path)
    planes = encode_many(df.fen.tolist())
    np.save(path, planes)
    return planes


def scalar_features(df: pd.DataFrame, use_clock: bool) -> np.ndarray:
    """Ratings, ply and (optionally) clock features as a float32 matrix, before standardisation."""
    cols = {
        "mover_elo": df.mover_elo, "opp_elo": df.opp_elo, "elo_diff": df.elo_diff,
        "ply": df.ply, "mover_is_white": df.mover_is_white,
    }
    if use_clock:
        cols |= {
            "clock": df.clock_before / 600, "log_clock": np.log1p(df.clock_before),
            "opp_clock": df.opp_clock / 600, "log_opp_clock": np.log1p(df.opp_clock),
            "clock_diff": df.clock_diff / 600,
        }
    return pd.DataFrame(cols).to_numpy(dtype=np.float32)
