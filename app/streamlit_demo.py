"""Streamlit demo: how likely is the player to move to blunder, given the position, ratings and clocks?"""
import base64

import chess
import chess.svg
import numpy as np
import pandas as pd
import streamlit as st

from chessblunder.inference import (TRAIN_MIN_PLY, BlunderEnsemble, board_from_moves, build_inputs,
                                    ply_from_board)

EXAMPLES = {
    "(your own position)": ("moves", ""),
    "Ruy Lopez, closed (middlegame)": (
        "moves", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O "
                 "9. h3 Nb8 10. d4 Nbd7"),
    "Italian game, open and tactical": (
        "moves", "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 Nxe4 "
                 "8. O-O Bxc3 9. d5 Bf6 10. Re1 Ne7 11. Rxe4 d6 12. Bg5 Bxg5 13. Nxg5 O-O"),
    "Rook endgame": ("fen", "8/5pk1/6p1/8/3R4/6P1/5PK1/r7 w - - 0 40"),
}
CLOCK_GRID = [5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 600]
TYPICAL_RATE = 0.098  # blunder rate among the evaluated test positions


@st.cache_resource
def load_model() -> BlunderEnsemble:
    return BlunderEnsemble("models")


st.set_page_config(page_title="Chess blunder predictor", page_icon="♟️")
st.title("♟️ Chess blunder predictor")
st.write(
    "Estimates the chance that the **player to move** blunders on their next move, meaning they give up "
    "at least 15 percentage points of win probability, in a 10-minute rapid game. It uses the position, "
    "both ratings and both clocks, and never sees an engine evaluation."
)

example = st.selectbox("Start from an example", list(EXAMPLES))
default_mode, default_text = EXAMPLES[example]
mode = st.radio("Enter the position as", ["Moves", "FEN"], index=0 if default_mode == "moves" else 1,
                horizontal=True, key=f"mode_{example}")
text = st.text_area(
    "Moves (for example 1. e4 e5 2. Nf3 Nc6 ...)" if mode == "Moves" else "FEN",
    value=default_text if mode.lower() == default_mode else "", height=110, key=f"text_{example}_{mode}",
)

try:
    board = board_from_moves(text) if mode == "Moves" else chess.Board(text.strip())
except ValueError as err:
    st.info(f"Enter a position to get a prediction. ({err})")
    st.stop()
if not board.is_valid():
    st.error("That position is not legal.")
    st.stop()
if board.is_game_over():
    st.warning("The game is over in this position, so there is no next move to predict.")
    st.stop()

fen = board.fen()
mover, other = ("White", "Black") if board.turn == chess.WHITE else ("Black", "White")

left, right = st.columns(2)
with left:
    st.subheader("Players and clocks")
    mover_elo = st.number_input(f"{mover} (to move) rating", 600, 3000, 1500, step=25)
    opp_elo = st.number_input(f"{other} rating", 600, 3000, 1500, step=25)
    mover_clock = st.slider(f"{mover} clock, seconds left", 1, 600, 300)
    opp_clock = st.slider(f"{other} clock, seconds left", 1, 600, 300)
with right:
    lastmove = board.move_stack[-1] if board.move_stack else None
    svg = chess.svg.board(board, size=380, lastmove=lastmove)
    encoded = base64.b64encode(svg.encode("utf-8")).decode()
    st.markdown(f'<img src="data:image/svg+xml;base64,{encoded}" width="380"/>', unsafe_allow_html=True)

if ply_from_board(board) < TRAIN_MIN_PLY:
    st.warning("This is an opening position. The models were trained on positions after move 5, so treat the "
               "result with caution.")

model = load_model()
planes, x = build_inputs(fen, mover_elo, opp_elo, mover_clock, opp_clock, model.mu, model.sd)
probs = model.predict_inputs(planes, x)[:, 0]
st.metric(f"Probability that {mover} blunders", f"{probs.mean():.1%}",
          f"{probs.mean() / TYPICAL_RATE:.1f}x the typical rate ({TYPICAL_RATE:.1%})", delta_color="off")
st.caption(f"Average of {len(probs)} networks (their estimates range from {probs.min():.1%} to {probs.max():.1%}).")

st.subheader("What would a different clock do?")
grid = [build_inputs(fen, mover_elo, opp_elo, c, opp_clock, model.mu, model.sd) for c in CLOCK_GRID]
curve = model.predict_inputs(np.concatenate([g[0] for g in grid]), np.concatenate([g[1] for g in grid])).mean(axis=0)
st.line_chart(pd.DataFrame({"blunder probability": curve},
                           index=pd.Index(CLOCK_GRID, name=f"{mover}'s clock (seconds left)")))
st.caption("Same position, ratings and opponent clock; only the mover's clock changes. In testing, the clock "
           "mainly improved predictions in the last minute or so, so expect little movement above that.")

with st.expander("About this model"):
    st.markdown(
        "- Trained on about 250,000 Lichess rapid 10+0 games that have engine analysis, using positions after "
        "move 5 where the player to move was not already lost (there is no engine here to check that).\n"
        "- A blunder is a move that costs at least 15 points of win probability, using Lichess's evaluation-to-win-"
        "probability curve.\n"
        "- It estimates how error-prone the *situation* is for a player of this rating and clock. It does not "
        "judge whether a specific move is good.\n"
        "- On held-out games it ranks positions much better than a gradient-boosted baseline and its "
        "probabilities are well calibrated on average."
    )
