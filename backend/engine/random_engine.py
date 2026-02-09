import random

import chess

from backend.engine.base import BaseEngine


class RandomEngine(BaseEngine):
    def select_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))
