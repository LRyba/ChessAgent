"""Board representation and move encoding utilities for chess SL/RL training."""

import chess
import numpy as np

# Channel mapping: 0-5 = white pieces, 6-11 = black pieces, 12 = side-to-move
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

NUM_CHANNELS = 13       # 12 piece planes + 1 side-to-move
NUM_MOVE_CLASSES = 4096  # 64 * 64 (from_square * 64 + to_square)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a chess.Board to a (13, 8, 8) float32 tensor.

    Channels 0-5:  white pawn, knight, bishop, rook, queen, king
    Channels 6-11: black pawn, knight, bishop, rook, queen, king
    Channel 12:    side-to-move (all 1s if white to move, all 0s if black)
    """
    tensor = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = square // 8  # 0-7
        file = square % 8   # 0-7
        channel = PIECE_TO_CHANNEL[piece.piece_type]
        if piece.color == chess.BLACK:
            channel += 6
        tensor[channel, rank, file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    return tensor


def move_to_index(move: chess.Move) -> int:
    """Encode a move as from_square * 64 + to_square (0..4095)."""
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board | None = None) -> chess.Move:
    """Decode an index back to a chess.Move.

    If board is provided, automatically adds queen promotion when needed.
    """
    from_square = index // 64
    to_square = index % 64
    move = chess.Move(from_square, to_square)

    if board is not None:
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            to_rank = to_square // 8
            if to_rank == 0 or to_rank == 7:
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask (4096,) — True for legal moves."""
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
