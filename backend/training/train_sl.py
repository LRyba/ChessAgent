"""Supervised-learning training script for ChessCNN.

Usage:
    python -m backend.training.train_sl \
        --data datasets/processed \
        --output models/sl_model.pt \
        --epochs 10 --batch-size 512 --lr 1e-3
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from backend.models.chess_cnn import ChessCNN
from backend.training.dataset import ChessDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ChessCNN with supervised learning")
    p.add_argument("--data", type=str, required=True, help="Directory with batch_*.npz files")
    p.add_argument("--output", type=str, default="models/sl_model.pt", help="Path to save best model")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    p.add_argument("--val-split", type=float, default=0.1, help="Validation fraction")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for boards, moves in loader:
        boards, moves = boards.to(device), moves.to(device)
        logits = model(boards)
        loss = criterion(logits, moves)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * boards.size(0)
        total_samples += boards.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    for boards, moves in loader:
        boards, moves = boards.to(device), moves.to(device)
        logits = model(boards)
        loss = criterion(logits, moves)
        total_loss += loss.item() * boards.size(0)
        # Top-1
        preds = logits.argmax(dim=1)
        correct_top1 += (preds == moves).sum().item()
        # Top-5
        _, top5 = logits.topk(5, dim=1)
        correct_top5 += (top5 == moves.unsqueeze(1)).any(dim=1).sum().item()
        total_samples += boards.size(0)
    return (
        total_loss / total_samples,
        correct_top1 / total_samples,
        correct_top5 / total_samples,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print(f"Loading dataset from {args.data} ...")
    dataset = ChessDataset(args.data)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = ChessCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Output dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>11} {'Top-1':>7} {'Top-5':>7} {'LR':>10} {'Time':>7}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} {train_loss:11.4f} {val_loss:11.4f} {top1:7.2%} {top5:7.2%} {lr:10.1e} {elapsed:6.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
