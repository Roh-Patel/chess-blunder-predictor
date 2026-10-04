"""Turn one parsed Lichess game into a labelled row per move."""
import chess
import chess.pgn

from chessblunder.labelling import MATE_CP, win_prob


def game_to_rows(game: chess.pgn.Game) -> list[dict]:
    h = game.headers
    base, _, inc = h["TimeControl"].partition("+")
    increment = float(inc or 0)
    # Seconds on each side's clock as of their own last move (= what they have when it's their turn)
    clock = {chess.WHITE: float(base), chess.BLACK: float(base)}
    elo = {chess.WHITE: int(h["WhiteElo"]), chess.BLACK: int(h["BlackElo"])}
    game_id = h["Site"].rsplit("/", 1)[-1]

    rows = []
    board = game.board()
    prev_cp = 0.0  # evaluation of the position before the move, from White's point of view
    for ply, node in enumerate(game.mainline(), start=1):
        mover = board.turn
        sign = 1 if mover == chess.WHITE else -1
        fen_before = board.fen()
        clock_before = clock[mover]
        opp_clock = clock[not mover]

        score = node.eval()  # [%eval] after this move, White's point of view
        clk_after = node.clock()  # [%clk] mover's clock after this move
        cp_after = score.white().score(mate_score=MATE_CP) if score is not None else None

        if cp_after is not None and prev_cp is not None and clk_after is not None:
            wp_before = win_prob(sign * prev_cp)
            wp_after = win_prob(sign * cp_after)
            rows.append({
                "game_id": game_id,
                "ply": ply,
                "fen": fen_before,
                "move_uci": node.move.uci(),
                "mover_is_white": mover == chess.WHITE,
                "mover_elo": elo[mover],
                "opp_elo": elo[not mover],
                "clock_before": clock_before,
                "opp_clock": opp_clock,
                "time_spent": clock_before - clk_after + increment,  # analysis only, not model input
                "cp_before": float(sign * prev_cp),
                "cp_after": float(sign * cp_after),
                "wp_before": wp_before,
                "wp_after": wp_after,
                "wp_drop": wp_before - wp_after,
            })

        prev_cp = float(cp_after) if cp_after is not None else None
        if clk_after is not None:
            clock[mover] = clk_after
        board.push(node.move)
    return rows
