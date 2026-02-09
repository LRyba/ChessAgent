import logging
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.engine.base import BaseEngine
from backend.engine.random_engine import RandomEngine

app = FastAPI(title="Chess AI")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engines: dict[str, BaseEngine] = {}

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _find_models() -> dict[str, Path]:
    """Scan models/ dir and return mapping: engine_name -> model_path."""
    models = {}
    if MODELS_DIR.exists():
        for pt_file in MODELS_DIR.glob("*_model.pt"):
            # sl_model.pt -> "sl", sl_100k_model.pt -> "sl_100k"
            name = pt_file.stem.removesuffix("_model")
            models[name] = pt_file
    return models


def get_engine(name: str) -> BaseEngine:
    if name not in _engines:
        if name == "random":
            _engines[name] = RandomEngine()
        else:
            models = _find_models()
            if name not in models:
                raise HTTPException(status_code=400, detail=f"Unknown engine: {name}")
            from backend.engine.neural_engine import NeuralEngine
            _engines[name] = NeuralEngine(str(models[name]))
            logger.info("Loaded engine '%s' from %s", name, models[name])
    return _engines[name]


class MoveRequest(BaseModel):
    fen: str
    move: str
    engine: str = "random"


class MoveResponse(BaseModel):
    fen: str
    ai_move: str | None
    status: str


class NewGameResponse(BaseModel):
    fen: str
    status: str


def get_game_status(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        return "draw"
    return "playing"


@app.get("/engines")
def list_engines():
    """Return available engine names."""
    available = ["random"] + sorted(_find_models().keys())
    return {"engines": available}


@app.post("/new_game", response_model=NewGameResponse)
def new_game():
    board = chess.Board()
    return NewGameResponse(fen=board.fen(), status="playing")


@app.post("/move", response_model=MoveResponse)
def make_move(req: MoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {req.fen}")

    try:
        move = chess.Move.from_uci(req.move)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid UCI move: {req.move}")

    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail=f"Illegal move: {req.move}")

    board.push(move)

    status = get_game_status(board)
    if status != "playing":
        return MoveResponse(fen=board.fen(), ai_move=None, status=status)

    engine = get_engine(req.engine)
    ai_move = engine.select_move(board)
    board.push(ai_move)

    status = get_game_status(board)
    return MoveResponse(fen=board.fen(), ai_move=ai_move.uci(), status=status)
