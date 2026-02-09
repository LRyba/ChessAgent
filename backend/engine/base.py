import abc

import chess


class BaseEngine(abc.ABC):
    @abc.abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move: ...
