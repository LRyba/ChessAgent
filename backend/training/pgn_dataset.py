"""CLI tool: parse PGN file and produce .npz dataset batches for SL training."""

import argparse
import json
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .data import board_to_tensor, move_to_index


def should_skip_game(game: chess.pgn.Game) -> bool:
    """Skip games with no result or very few moves."""
    result = game.headers.get("Result", "*")
    if result == "*":
        return True
    return False


def process_game(game: chess.pgn.Game, skip_plies: int = 10) -> list[tuple[np.ndarray, int]]:
    """Extract (board_tensor, move_index) pairs from a game.

    Skips the first `skip_plies` half-moves (opening book territory).
    """
    samples = []
    board = game.board()
    ply = 0

    for move in game.mainline_moves():
        if ply >= skip_plies:
            tensor = board_to_tensor(board)
            index = move_to_index(move)
            samples.append((tensor, index))
        board.push(move)
        ply += 1

    return samples


def save_batch(
    boards: list[np.ndarray],
    moves: list[int],
    output_dir: Path,
    batch_num: int,
) -> Path:
    """Save a batch of samples as compressed .npz."""
    boards_arr = np.stack(boards)
    moves_arr = np.array(moves, dtype=np.int16)
    path = output_dir / f"batch_{batch_num:04d}.npz"
    np.savez_compressed(path, boards=boards_arr, moves=moves_arr)
    return path


def main():
    parser = argparse.ArgumentParser(description="Convert PGN to training dataset")
    parser.add_argument("--pgn", required=True, help="Path to PGN file")
    parser.add_argument("--output", required=True, help="Output directory for .npz files")
    parser.add_argument("--max-games", type=int, default=20000, help="Max games to process")
    parser.add_argument("--skip-plies", type=int, default=10, help="Skip first N half-moves")
    parser.add_argument("--batch-size", type=int, default=50000, help="Samples per .npz batch")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    boards_buf: list[np.ndarray] = []
    moves_buf: list[int] = []
    batch_num = 0
    total_samples = 0
    games_processed = 0
    games_skipped = 0

    start_time = time.time()

    with open(args.pgn, encoding="utf-8", errors="replace") as f:
        while games_processed < args.max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            if should_skip_game(game):
                games_skipped += 1
                continue

            samples = process_game(game, skip_plies=args.skip_plies)
            for tensor, index in samples:
                boards_buf.append(tensor)
                moves_buf.append(index)

            total_samples += len(samples)
            games_processed += 1

            if len(boards_buf) >= args.batch_size:
                path = save_batch(boards_buf, moves_buf, output_dir, batch_num)
                print(f"  Saved {path.name} ({len(boards_buf)} samples)")
                boards_buf.clear()
                moves_buf.clear()
                batch_num += 1

            if games_processed % 1000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{games_processed}/{args.max_games}] "
                    f"{total_samples} samples, {elapsed:.1f}s"
                )

    # Save remaining samples
    if boards_buf:
        path = save_batch(boards_buf, moves_buf, output_dir, batch_num)
        print(f"  Saved {path.name} ({len(boards_buf)} samples)")
        batch_num += 1

    elapsed = time.time() - start_time

    # Save metadata
    metadata = {
        "pgn": args.pgn,
        "max_games": args.max_games,
        "skip_plies": args.skip_plies,
        "batch_size": args.batch_size,
        "games_processed": games_processed,
        "games_skipped": games_skipped,
        "total_samples": total_samples,
        "num_batches": batch_num,
        "elapsed_seconds": round(elapsed, 1),
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nDone! {games_processed} games -> {total_samples} samples in {batch_num} batches")
    print(f"Time: {elapsed:.1f}s")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
