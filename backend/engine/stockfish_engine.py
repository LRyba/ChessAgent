"""Stockfish UCI engine wrapper."""

import chess
import chess.engine

from backend.engine.base import BaseEngine


class StockfishEngine(BaseEngine):
    """Wraps a Stockfish binary via UCI protocol."""

    def __init__(
        self, path: str = "stockfish", depth: int = 1, time_limit: float = 0.1
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.limit = chess.engine.Limit(depth=depth, time=time_limit)

    def select_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, self.limit)
        return result.move

    def close(self):
        self.engine.quit()
