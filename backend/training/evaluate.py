"""Evaluation script — run matches between chess engines and collect statistics."""

import argparse
import csv
import dataclasses
import logging
import sys
from pathlib import Path

import chess

from backend.engine.base import BaseEngine
from backend.engine.random_engine import RandomEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MatchGame:
    """Result of a single game."""

    result: float       # +1 white won, 0 draw, -1 black won
    num_moves: int
    moves: list[str]    # UCI move strings
    termination: str    # "checkmate", "stalemate", "draw", "max_moves"


@dataclasses.dataclass
class MatchResult:
    """Aggregate result of a multi-game match."""

    engine1_name: str
    engine2_name: str
    engine1_wins: int
    engine2_wins: int
    draws: int
    total_games: int
    avg_length: float
    games: list[MatchGame]


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_match_game(
    white: BaseEngine,
    black: BaseEngine,
    max_moves: int = 200,
) -> MatchGame:
    """Play one game between two engines. Returns a MatchGame."""
    board = chess.Board()
    moves: list[str] = []

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        engine = white if board.turn == chess.WHITE else black
        move = engine.select_move(board)
        moves.append(move.uci())
        board.push(move)

    # Determine termination and result
    result = 0.0
    if board.is_checkmate():
        termination = "checkmate"
        result = -1.0 if board.turn == chess.WHITE else 1.0
    elif board.is_stalemate():
        termination = "stalemate"
    elif board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        termination = "draw"
    else:
        termination = "max_moves"

    return MatchGame(
        result=result,
        num_moves=len(moves),
        moves=moves,
        termination=termination,
    )


# ---------------------------------------------------------------------------
# Full match
# ---------------------------------------------------------------------------

def run_match(
    name1: str,
    engine1: BaseEngine,
    name2: str,
    engine2: BaseEngine,
    num_games: int,
    max_moves: int = 200,
) -> MatchResult:
    """Play *num_games* between two engines, alternating colours each game."""
    e1_wins = 0
    e2_wins = 0
    draws = 0
    total_moves = 0
    games: list[MatchGame] = []

    for i in range(num_games):
        # Alternate colours for fairness
        if i % 2 == 0:
            game = play_match_game(engine1, engine2, max_moves)
            # engine1 is white
            if game.result > 0:
                e1_wins += 1
            elif game.result < 0:
                e2_wins += 1
            else:
                draws += 1
        else:
            game = play_match_game(engine2, engine1, max_moves)
            # engine1 is black
            if game.result < 0:
                e1_wins += 1
            elif game.result > 0:
                e2_wins += 1
            else:
                draws += 1

        total_moves += game.num_moves
        games.append(game)

        if (i + 1) % 50 == 0 or (i + 1) == num_games:
            logger.info(
                "  [%d/%d] %s %d – %d %s (draws %d)",
                i + 1, num_games, name1, e1_wins, e2_wins, name2, draws,
            )

    avg_len = total_moves / num_games if num_games else 0.0

    return MatchResult(
        engine1_name=name1,
        engine2_name=name2,
        engine1_wins=e1_wins,
        engine2_wins=e2_wins,
        draws=draws,
        total_games=num_games,
        avg_length=avg_len,
        games=games,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(mr: MatchResult) -> None:
    """Print a human-readable summary table."""
    w = max(len(mr.engine1_name), len(mr.engine2_name), 6)
    header = f"Match: {mr.engine1_name} vs {mr.engine2_name} ({mr.total_games} games)"
    sep = "-" * len(header)

    print(f"\n{header}")
    print(sep)
    print(f"{'':>{w}}  {'Wins':>6}  {'Losses':>6}  {'Draws':>6}  {'Win%':>6}")

    e1_pct = mr.engine1_wins / mr.total_games * 100 if mr.total_games else 0
    e2_pct = mr.engine2_wins / mr.total_games * 100 if mr.total_games else 0

    print(f"{mr.engine1_name:>{w}}  {mr.engine1_wins:>6}  {mr.engine2_wins:>6}  {mr.draws:>6}  {e1_pct:>5.1f}%")
    print(f"{mr.engine2_name:>{w}}  {mr.engine2_wins:>6}  {mr.engine1_wins:>6}  {mr.draws:>6}  {e2_pct:>5.1f}%")
    print(sep)
    print(f"Avg game length: {mr.avg_length:.1f} moves\n")


def save_csv(mr: MatchResult, csv_path: Path) -> None:
    """Append one row to the CSV results file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "match", "engine1", "engine2",
                "e1_wins", "e2_wins", "draws", "total", "avg_length",
            ])
        writer.writerow([
            f"{mr.engine1_name}_vs_{mr.engine2_name}",
            mr.engine1_name,
            mr.engine2_name,
            mr.engine1_wins,
            mr.engine2_wins,
            mr.draws,
            mr.total_games,
            f"{mr.avg_length:.1f}",
        ])

    logger.info("CSV saved to %s", csv_path)


def save_pgn(mr: MatchResult, pgn_path: Path) -> None:
    """Write all games to a PGN file."""
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pgn_path, "w") as f:
        for game_idx, game in enumerate(mr.games):
            # Determine white/black names based on alternation
            if game_idx % 2 == 0:
                white_name = mr.engine1_name
                black_name = mr.engine2_name
            else:
                white_name = mr.engine2_name
                black_name = mr.engine1_name

            # PGN result string
            if game.result > 0:
                result_str = "1-0"
            elif game.result < 0:
                result_str = "0-1"
            else:
                result_str = "1/2-1/2"

            f.write(f'[Event "Eval Match"]\n')
            f.write(f'[White "{white_name}"]\n')
            f.write(f'[Black "{black_name}"]\n')
            f.write(f'[Result "{result_str}"]\n')
            f.write(f'[Termination "{game.termination}"]\n')
            f.write("\n")

            # Write moves in SAN from UCI
            board = chess.Board()
            tokens: list[str] = []
            for i, uci in enumerate(game.moves):
                if i % 2 == 0:
                    tokens.append(f"{i // 2 + 1}.")
                move = chess.Move.from_uci(uci)
                tokens.append(board.san(move))
                board.push(move)

            # Wrap lines at ~80 chars
            line = ""
            for tok in tokens:
                if len(line) + len(tok) + 1 > 80:
                    f.write(line.rstrip() + "\n")
                    line = ""
                line += tok + " "
            if line.strip():
                f.write(line.rstrip() + "\n")

            f.write(f"{result_str}\n\n")

    logger.info("PGN saved to %s", pgn_path)


# ---------------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------------

def load_engine(
    name: str,
    stockfish_path: str = "stockfish",
    stockfish_depth: int = 1,
    temperature: float = 0.0,
) -> BaseEngine:
    """Load an engine by name. Supports 'random', 'stockfish', and neural models."""
    if name == "random":
        return RandomEngine()

    if name == "stockfish":
        from backend.engine.stockfish_engine import StockfishEngine
        return StockfishEngine(stockfish_path, depth=stockfish_depth)

    # Neural model — look for <name>_model.pt
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    model_path = models_dir / f"{name}_model.pt"
    if not model_path.exists():
        sys.exit(f"Error: model file not found: {model_path}")

    from backend.engine.neural_engine import NeuralEngine
    return NeuralEngine(str(model_path), temperature=temperature)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chess engines by playing matches.",
    )
    parser.add_argument("--engine1", required=True, help="First engine name (e.g. sl, rl, random, stockfish)")
    parser.add_argument("--engine2", required=True, help="Second engine name")
    parser.add_argument("--games", type=int, default=500, help="Number of games (default: 500)")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game (default: 200)")
    parser.add_argument("--save-pgn", action="store_true", help="Save games to PGN file")
    parser.add_argument("--stockfish-path", default="stockfish", help="Path to Stockfish binary")
    parser.add_argument("--stockfish-depth", type=int, default=1, help="Stockfish search depth (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature for neural engines, 0=argmax (default: 0.5)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading engines: %s, %s (temperature=%.2f)", args.engine1, args.engine2, args.temperature)
    e1 = load_engine(args.engine1, args.stockfish_path, args.stockfish_depth, args.temperature)
    e2 = load_engine(args.engine2, args.stockfish_path, args.stockfish_depth, args.temperature)

    logger.info("Starting match: %s vs %s (%d games)", args.engine1, args.engine2, args.games)
    result = run_match(args.engine1, e1, args.engine2, e2, args.games, args.max_moves)

    print_summary(result)

    # Save CSV
    eval_dir = Path(__file__).resolve().parent.parent.parent / "experiments" / "eval_logs"
    csv_path = eval_dir / "eval_results.csv"
    save_csv(result, csv_path)

    # Save PGN
    if args.save_pgn:
        pgn_path = eval_dir / f"{args.engine1}_vs_{args.engine2}.pgn"
        save_pgn(result, pgn_path)

    # Cleanup Stockfish if used
    for eng in (e1, e2):
        if hasattr(eng, "close"):
            eng.close()


if __name__ == "__main__":
    main()
