"""CNN model for chess move prediction."""

import torch.nn as nn


class ChessCNN(nn.Module):
    """4-layer CNN for predicting chess moves from board state.

    Input:  (batch, 13, 8, 8)  — 12 piece planes + side-to-move
    Output: (batch, 4096)       — logits over from_sq * 64 + to_sq
    """

    def __init__(self, num_channels: int = 13, num_classes: int = 4096):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: 13 → 64
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Block 3: 128 → 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        return self.head(x)
