"""Hand-crafted board features for the baseline model (no engine evaluation involved)."""
import chess

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PIECE_NAMES = {chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}


def board_features(fen: str) -> dict:
    """Features of a position, all from the point of view of the side about to move."""
    board = chess.Board(fen)
    me = board.turn
    them = not me
    f: dict[str, float] = {}

    # Piece counts and material
    material = {me: 0, them: 0}
    for piece_type, name in PIECE_NAMES.items():
        n_me = len(board.pieces(piece_type, me))
        n_them = len(board.pieces(piece_type, them))
        f[f"my_{name}"] = n_me
        f[f"opp_{name}"] = n_them
        material[me] += PIECE_VALUES[piece_type] * n_me
        material[them] += PIECE_VALUES[piece_type] * n_them
    f["material_diff"] = material[me] - material[them]
    f["total_material"] = material[me] + material[them]

    # Mobility and forcing moves
    legal = list(board.legal_moves)
    f["in_check"] = int(board.is_check())
    f["n_legal"] = len(legal)
    f["n_captures"] = sum(board.is_capture(m) for m in legal)

    # Attacked / hanging pieces and king exposure, for each side
    for side, prefix in ((me, "my"), (them, "opp")):
        attacker = not side
        n_attacked = n_hanging = max_hanging = 0
        for sq in chess.scan_forward(board.occupied_co[side]):
            piece_type = board.piece_type_at(sq)
            if piece_type == chess.KING:
                continue
            if board.is_attacked_by(attacker, sq):
                n_attacked += 1
                if not board.is_attacked_by(side, sq):  # attacked and undefended
                    n_hanging += 1
                    max_hanging = max(max_hanging, PIECE_VALUES[piece_type])
        f[f"{prefix}_attacked"] = n_attacked
        f[f"{prefix}_hanging"] = n_hanging
        f[f"{prefix}_max_hanging"] = max_hanging
        king_sq = board.king(side)
        f[f"{prefix}_king_zone_attacked"] = sum(
            board.is_attacked_by(attacker, sq) for sq in board.attacks(king_sq)
        )
    return f
