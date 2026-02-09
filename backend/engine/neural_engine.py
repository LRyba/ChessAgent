"""Neural network chess engine using a trained ChessCNN model."""

import torch
import chess

from backend.engine.base import BaseEngine
from backend.models.chess_cnn import ChessCNN
from backend.training.data import board_to_tensor, index_to_move, legal_move_mask


class NeuralEngine(BaseEngine):
    """Selects moves using a trained ChessCNN with legal-move masking."""

    def __init__(self, model_path: str, temperature: float = 0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.temperature = temperature

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        logits = self.model(tensor).squeeze(0)

        # Mask illegal moves to -inf
        mask = torch.from_numpy(legal_move_mask(board)).to(self.device)
        logits[~mask] = float("-inf")

        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=0)
            index = torch.multinomial(probs, num_samples=1).item()
        else:
            index = logits.argmax().item()
        return index_to_move(index, board)
