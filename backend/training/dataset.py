"""PyTorch Dataset for loading pre-processed chess .npz batches."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """Dataset that loads all batch_*.npz files from a directory into memory.

    Each sample is a (board_tensor, move_index) pair:
        - board_tensor: float32 tensor of shape (13, 8, 8)
        - move_index: int64 scalar (0..4095)
    """

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        npz_files = sorted(data_dir.glob("batch_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No batch_*.npz files found in {data_dir}")

        all_boards = []
        all_moves = []
        for path in npz_files:
            data = np.load(path)
            # Store as uint8 (0/1) instead of float32 — 4× less RAM
            all_boards.append(data["boards"].astype(np.uint8))
            all_moves.append(data["moves"])

        self.boards = torch.from_numpy(np.concatenate(all_boards))
        self.moves = torch.from_numpy(np.concatenate(all_moves).astype(np.int64))

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.boards[idx].float(), self.moves[idx]
