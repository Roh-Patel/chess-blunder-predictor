# Chess Blunder Predictor

Predicts the probability that a human player blunders on their next move,
given the position, their rating, and their remaining clock time.
Trained on Lichess games (CC0 licensed).

## Status
Work in progress.

## Setup
1. `python -m venv .venv` then `.venv\Scripts\activate`
2. Install PyTorch for your CUDA version from https://pytorch.org
3. `pip install -r requirements.txt`
4. `pip install -e .`