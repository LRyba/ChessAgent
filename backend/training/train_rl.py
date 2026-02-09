"""REINFORCE (policy gradient) training via self-play.

Usage:
    python -m backend.training.train_rl \
        --checkpoint models/sl_model.pt \
        --output models/rl_model.pt \
        --iterations 500 --games-per-iter 32 --lr 1e-5
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from backend.models.chess_cnn import ChessCNN
from backend.training.self_play import GameRecord, play_one_game, play_one_game_vs_opponent


def collect_games(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    temperature: float,
    max_moves: int,
    opponent: torch.nn.Module | None = None,
    reward_shaping: bool = False,
) -> list[GameRecord]:
    """Play num_games and return the records.

    If opponent is provided, plays learner vs frozen opponent
    (alternating colours). Otherwise, plays self-play.
    """
    if opponent is None:
        return [
            play_one_game(model, device, temperature, max_moves, reward_shaping)
            for _ in range(num_games)
        ]
    return [
        play_one_game_vs_opponent(
            model, opponent, device,
            learner_white=(i % 2 == 0),
            temperature=temperature,
            max_moves=max_moves,
            reward_shaping=reward_shaping,
        )
        for i in range(num_games)
    ]


def reinforce_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    games: list[GameRecord],
    device: torch.device,
    entropy_coeff: float,
    max_grad_norm: float,
    mini_batch_size: int = 1024,
) -> tuple[float, float]:
    """Perform a single REINFORCE update from collected games.

    Uses mini-batching with gradient accumulation to avoid VRAM OOM
    on GPUs with limited memory (e.g. 4GB RTX 3050).

    Returns (policy_loss, entropy) as floats for logging.
    """
    # Flatten trajectories
    all_states: list[np.ndarray] = []
    all_actions: list[int] = []
    all_rewards: list[float] = []

    for game in games:
        for i in range(game.num_moves):
            all_states.append(game.states[i])
            all_actions.append(game.actions[i])
            # Game-level reward from perspective of the player who made this move
            game_reward = game.result if game.sides[i] else -game.result
            # Per-step reward (material delta) if available
            step_reward = game.step_rewards[i] if game.step_rewards else 0.0
            all_rewards.append(game_reward + step_reward)

    if not all_states:
        return 0.0, 0.0

    # Normalize rewards (variance reduction)
    rewards = np.array(all_rewards, dtype=np.float32)
    std = rewards.std()
    if std > 0:
        rewards = (rewards - rewards.mean()) / (std + 1e-8)

    # Prepare tensors (keep on CPU, move mini-batches to GPU)
    states_np = np.stack(all_states)
    actions_np = np.array(all_actions, dtype=np.int64)
    rewards_np = rewards

    n = len(all_states)
    num_chunks = (n + mini_batch_size - 1) // mini_batch_size

    model.train()
    optimizer.zero_grad()

    total_policy_loss = 0.0
    total_entropy = 0.0

    for start in range(0, n, mini_batch_size):
        end = min(start + mini_batch_size, n)

        states_t = torch.from_numpy(states_np[start:end]).to(device)
        actions_t = torch.from_numpy(actions_np[start:end]).to(device)
        rewards_t = torch.from_numpy(rewards_np[start:end]).to(device)

        # Forward pass
        logits = model(states_t)  # (B, 4096)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Gather log probabilities for taken actions
        action_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Policy loss: -log(pi(a|s)) * R_normalized
        policy_loss = -(action_log_probs * rewards_t).mean()

        # Entropy bonus: encourages exploration
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # Scale loss by chunk fraction for correct gradient accumulation
        chunk_loss = (policy_loss - entropy_coeff * entropy) / num_chunks
        chunk_loss.backward()

        total_policy_loss += policy_loss.item()
        total_entropy += entropy.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return total_policy_loss / num_chunks, total_entropy / num_chunks


def train(args: argparse.Namespace) -> None:
    """Main REINFORCE training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SL checkpoint
    model = ChessCNN()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Frozen opponent — separate copy of SL weights, never updated
    opponent = None
    if args.frozen_opponent:
        opponent = ChessCNN()
        opponent.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        opponent.to(device)
        opponent.eval()
        print("Frozen opponent: enabled (plays as SL baseline)")

    if args.reward_shaping:
        print("Reward shaping: enabled (material-based reward for draws)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare logging directory
    log_dir = os.path.join("experiments", "rl_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "iteration", "white_wins", "black_wins", "draws",
        "avg_length", "policy_loss", "entropy", "time_s",
    ])

    # Print header
    print()
    print(f" {'Iter':>5}  {'W':>4}  {'B':>4}  {'D':>4}  "
          f"{'AvgLen':>6}  {'Loss':>8}  {'Entropy':>8}  {'Time':>6}")
    print("-" * 58)

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()

        # --- Collect games ---
        games = collect_games(
            model, device, args.games_per_iter,
            args.temperature, args.max_moves, opponent,
            args.reward_shaping,
        )

        # --- Statistics ---
        white_wins = sum(1 for g in games if g.result == 1.0)
        black_wins = sum(1 for g in games if g.result == -1.0)
        draws = sum(1 for g in games if g.result == 0.0)
        avg_length = sum(g.num_moves for g in games) / len(games)

        # --- REINFORCE update ---
        policy_loss, entropy = reinforce_update(
            model, optimizer, games, device,
            args.entropy_coeff, args.max_grad_norm,
        )

        elapsed = time.time() - t0

        # --- Log ---
        csv_writer.writerow([
            iteration, white_wins, black_wins, draws,
            f"{avg_length:.1f}", f"{policy_loss:.4f}",
            f"{entropy:.3f}", f"{elapsed:.1f}",
        ])
        csv_file.flush()

        print(f" {iteration:5d}  {white_wins:4d}  {black_wins:4d}  {draws:4d}  "
              f"{avg_length:6.1f}  {policy_loss:8.4f}  {entropy:8.3f}  {elapsed:5.1f}s")

        # --- Checkpoint ---
        if iteration % args.checkpoint_every == 0:
            ckpt_path = os.path.join(log_dir, f"rl_iter_{iteration}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

    # Save final model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nFinal model saved: {args.output}")

    csv_file.close()
    print(f"Training log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE self-play training for chess CNN",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SL model checkpoint (.pt)")
    parser.add_argument("--output", required=True,
                        help="Path to save the RL-trained model")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of training iterations (default: 500)")
    parser.add_argument("--games-per-iter", type=int, default=32,
                        help="Self-play games per iteration (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Max moves per game (default: 200)")
    parser.add_argument("--entropy-coeff", type=float, default=0.01,
                        help="Entropy bonus coefficient (default: 0.01)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N iterations (default: 50)")
    parser.add_argument("--frozen-opponent", action="store_true",
                        help="Play vs frozen SL copy instead of self-play")
    parser.add_argument("--reward-shaping", action="store_true",
                        help="Use material-based reward for non-checkmate endings")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
