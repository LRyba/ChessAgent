"""Self-play game generation for reinforcement learning training."""

import dataclasses

import chess
import numpy as np
import torch

from backend.training.data import (
    board_to_tensor,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


@dataclasses.dataclass
class GameRecord:
    """Record of a single self-play game."""

    states: list[np.ndarray]   # (13, 8, 8) per move
    actions: list[int]         # move index 0..4095
    log_probs: list[float]     # log pi(a|s)
    sides: list[bool]          # True = white moved, False = black moved
    result: float              # +1 white won, 0 draw, -1 black won
    num_moves: int
    step_rewards: list[float] | None = None  # per-move rewards (reward shaping)


# Piece values for reward shaping (standard chess values)
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_balance(board: chess.Board) -> float:
    """Material advantage for white, normalized to roughly [-1, 1].

    Total material at start = 39 per side. We normalize by 39 so that
    a full material advantage maps to ~1.0.
    """
    score = 0
    for piece_type, value in _PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score / 39.0


def compute_result(board: chess.Board) -> float:
    """Compute game result from white's perspective."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


@torch.no_grad()
def play_one_game(
    model: torch.nn.Module,
    device: torch.device,
    temperature: float = 1.0,
    max_moves: int = 200,
    reward_shaping: bool = False,
) -> GameRecord:
    """Play a single self-play game using sampling from the policy.

    The model plays both sides. Moves are sampled from the softmax
    distribution (with temperature) over legal moves.
    """
    model.eval()
    board = chess.Board()

    states: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    sides: list[bool] = []
    step_rewards: list[float] = [] if reward_shaping else None

    prev_balance = material_balance(board) if reward_shaping else 0.0

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Encode board
        state = board_to_tensor(board)
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)

        # Forward pass
        logits = model(tensor).squeeze(0)  # (4096,)

        # Mask illegal moves
        mask = torch.from_numpy(legal_move_mask(board)).to(device)
        logits[~mask] = float("-inf")

        # Apply temperature and sample
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=0)
        action_idx = torch.multinomial(probs, num_samples=1).item()

        # Record log probability
        log_prob = torch.log_softmax(scaled_logits, dim=0)[action_idx].item()

        is_white = board.turn == chess.WHITE

        # Store trajectory step
        states.append(state)
        actions.append(action_idx)
        log_probs.append(log_prob)
        sides.append(is_white)

        # Execute move
        move = index_to_move(action_idx, board)
        board.push(move)

        # Per-move reward: material delta from mover's perspective
        if reward_shaping:
            new_balance = material_balance(board)
            delta = new_balance - prev_balance
            # From mover's perspective: white wants positive, black wants negative
            step_rewards.append(delta if is_white else -delta)
            prev_balance = new_balance

    result = compute_result(board)

    return GameRecord(
        states=states,
        actions=actions,
        log_probs=log_probs,
        sides=sides,
        result=result,
        num_moves=len(states),
        step_rewards=step_rewards,
    )


@torch.no_grad()
def play_one_game_vs_opponent(
    learner: torch.nn.Module,
    opponent: torch.nn.Module,
    device: torch.device,
    learner_white: bool,
    temperature: float = 1.0,
    max_moves: int = 200,
    reward_shaping: bool = False,
) -> GameRecord:
    """Play a game: learner vs frozen opponent.

    Only the learner's moves are recorded in the trajectory.
    The opponent plays with the same sampling logic but its moves
    are not included in the training data.
    """
    learner.eval()
    opponent.eval()
    board = chess.Board()

    states: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    sides: list[bool] = []
    step_rewards: list[float] = [] if reward_shaping else None

    prev_balance = material_balance(board) if reward_shaping else 0.0

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        is_learner_turn = (board.turn == chess.WHITE) == learner_white
        model = learner if is_learner_turn else opponent
        is_white = board.turn == chess.WHITE

        # Encode board
        state = board_to_tensor(board)
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)

        # Forward pass
        logits = model(tensor).squeeze(0)

        # Mask illegal moves
        mask = torch.from_numpy(legal_move_mask(board)).to(device)
        logits[~mask] = float("-inf")

        # Apply temperature and sample
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=0)
        action_idx = torch.multinomial(probs, num_samples=1).item()

        # Only record learner's moves
        if is_learner_turn:
            log_prob = torch.log_softmax(scaled_logits, dim=0)[action_idx].item()
            states.append(state)
            actions.append(action_idx)
            log_probs.append(log_prob)
            sides.append(is_white)

        # Execute move
        move = index_to_move(action_idx, board)
        board.push(move)

        # Per-move reward for learner's moves only
        if reward_shaping:
            new_balance = material_balance(board)
            if is_learner_turn:
                delta = new_balance - prev_balance
                step_rewards.append(delta if is_white else -delta)
            prev_balance = new_balance

    result = compute_result(board)

    return GameRecord(
        states=states,
        actions=actions,
        log_probs=log_probs,
        sides=sides,
        result=result,
        num_moves=len(states),
        step_rewards=step_rewards,
    )
