"""Turn a position, two ratings and two clocks into model inputs, and run the CNN ensemble."""
import io
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd

from chessblunder.encoding import N_PLANES, encode_fen, scalar_features
from chessblunder.features import board_features
from chessblunder.labelling import BLUNDER_DROP

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]  # must match the training runs
PRIMARY = min(range(len(THRESHOLDS)), key=lambda i: abs(THRESHOLDS[i] - BLUNDER_DROP))
BOARD_COLS = list(board_features(chess.STARTING_FEN))
TRAIN_MIN_PLY = 11  # the models were trained on positions after the first 10 plies


def ply_from_board(board: chess.Board) -> int:
    """Ply number of the move about to be played (1 = White's first move)."""
    return 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1) + 1


def board_from_moves(text: str) -> chess.Board:
    """Final position of a move list such as '1. e4 e5 2. Nf3 Nc6' (a full PGN also works)."""
    text = text.strip()
    if not text:
        raise ValueError("enter a move list, for example: 1. e4 e5 2. Nf3 Nc6")
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.errors or game.next() is None:  # nothing parsed, illegal move, or no moves at all
        raise ValueError("could not read the moves; check the notation")
    return game.end().board()


def build_inputs(fen: str, mover_elo: float, opp_elo: float, mover_clock: float, 
                 opp_clock: float, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(planes, scalars) for one position, same code as the training pipeline."""
    board = chess.Board(fen)
    row = pd.DataFrame([{
        "mover_elo": mover_elo, "opp_elo": opp_elo, "elo_diff": mover_elo - opp_elo,
        "ply": ply_from_board(board), "mover_is_white": int(board.turn == chess.WHITE),
        "clock_before": float(mover_clock), "opp_clock": float(opp_clock),
        "clock_diff": float(mover_clock) - float(opp_clock),
    }])
    handcrafted = pd.DataFrame([board_features(fen)])[BOARD_COLS].to_numpy(dtype=np.float32)
    x = np.hstack([scalar_features(row, True), handcrafted])
    x = ((x - mu) / sd).astype(np.float32)
    return encode_fen(fen)[None], x


class BlunderEnsemble:
    """The saved CNN seeds plus the standardisation statistics; averages their probabilities."""

    def __init__(self, model_dir: str = "models", pattern: str = "cnn_big_hybrid_s*.pt",
                 channels: int = 64, blocks: int = 6):
        import torch
        from chessblunder.model import BlunderNet

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = sorted(Path(model_dir).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no checkpoints matching {pattern} in {model_dir}/")
        scaler = np.load(Path(model_dir) / "scaler.npz")
        self.mu, self.sd = scaler["mu"], scaler["sd"]
        self.models = []
        for path in paths:
            model = BlunderNet(N_PLANES, len(self.mu), channels=channels, blocks=blocks,
                               n_outputs=len(THRESHOLDS)).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            self.models.append(model.eval())

    def predict_inputs(self, planes: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Blunder probabilities, shape (n_models, n_positions)."""
        torch = self._torch
        p = torch.from_numpy(planes).float().to(self.device)
        s = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            return np.stack([torch.sigmoid(m(p, s)[:, PRIMARY]).cpu().numpy() for m in self.models])
