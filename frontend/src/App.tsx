import { useRef, useState, useCallback, useEffect } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { newGame, sendMove, fetchEngines } from "./api";
import "./App.css";

type PieceDropArgs = {
  piece: { pieceType: string };
  sourceSquare: string;
  targetSquare: string | null;
};

type PendingPromotion = {
  from: string;
  to: string;
  fenBefore: string;
};

const PROMOTION_PIECES = [
  { piece: "q", label: "\u265B" },
  { piece: "r", label: "\u265C" },
  { piece: "b", label: "\u265D" },
  { piece: "n", label: "\u265E" },
] as const;

function statusLabel(status: string): string {
  switch (status) {
    case "checkmate":
      return "Checkmate!";
    case "stalemate":
      return "Stalemate!";
    case "draw":
      return "Draw!";
    default:
      return "";
  }
}

export default function App() {
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const [status, setStatus] = useState("playing");
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState("");
  const [pendingPromotion, setPendingPromotion] =
    useState<PendingPromotion | null>(null);
  const [engine, setEngine] = useState("random");
  const [availableEngines, setAvailableEngines] = useState<string[]>(["random"]);

  useEffect(() => {
    fetchEngines()
      .then((res) => setAvailableEngines(res.engines))
      .catch(() => {});
  }, []);

  const resetGame = useCallback(async () => {
    try {
      const res = await newGame();
      gameRef.current = new Chess(res.fen);
      setFen(res.fen);
      setStatus(res.status);
      setError("");
      setPendingPromotion(null);
    } catch {
      setError("Failed to start new game");
    }
  }, []);

  useEffect(() => {
    resetGame();
  }, [resetGame]);

  const executeMove = useCallback(
    (fenBefore: string, from: string, to: string, promotion?: string) => {
      const game = gameRef.current;

      try {
        game.move({ from, to, promotion });
      } catch {
        return;
      }

      setFen(game.fen());
      setError("");
      setIsThinking(true);

      sendMove(fenBefore, from + to + (promotion ?? ""), engine)
        .then((res) => {
          gameRef.current = new Chess(res.fen);
          setFen(res.fen);
          setStatus(res.status);
        })
        .catch((err: Error) => {
          gameRef.current = new Chess(fenBefore);
          setFen(fenBefore);
          setError(err.message);
        })
        .finally(() => {
          setIsThinking(false);
        });
    },
    [engine]
  );

  const handlePieceDrop = useCallback(
    ({ piece, sourceSquare, targetSquare }: PieceDropArgs): boolean => {
      if (isThinking || status !== "playing" || targetSquare === null) {
        return false;
      }

      const isPromotion =
        piece.pieceType === "wP" && targetSquare[1] === "8";

      if (isPromotion) {
        setPendingPromotion({
          from: sourceSquare,
          to: targetSquare,
          fenBefore: gameRef.current.fen(),
        });
        return true;
      }

      const fenBefore = gameRef.current.fen();
      try {
        gameRef.current.move({ from: sourceSquare, to: targetSquare });
      } catch {
        return false;
      }

      setFen(gameRef.current.fen());
      setError("");
      setIsThinking(true);

      sendMove(fenBefore, sourceSquare + targetSquare, engine)
        .then((res) => {
          gameRef.current = new Chess(res.fen);
          setFen(res.fen);
          setStatus(res.status);
        })
        .catch((err: Error) => {
          gameRef.current = new Chess(fenBefore);
          setFen(fenBefore);
          setError(err.message);
        })
        .finally(() => {
          setIsThinking(false);
        });

      return true;
    },
    [isThinking, status, engine]
  );

  const handlePromotionChoice = useCallback(
    (piece: string) => {
      if (!pendingPromotion) return;
      const { from, to, fenBefore } = pendingPromotion;
      setPendingPromotion(null);
      executeMove(fenBefore, from, to, piece);
    },
    [pendingPromotion, executeMove]
  );

  const cancelPromotion = useCallback(() => {
    setPendingPromotion(null);
  }, []);

  const gameOver = status !== "playing";

  return (
    <div className="app">
      <h1>Chess AI</h1>
      <div className="board-container">
        <Chessboard
          options={{
            position: fen,
            onPieceDrop: handlePieceDrop,
            boardOrientation: "white" as const,
            allowDragging:
              !isThinking && !gameOver && pendingPromotion === null,
            animationDurationInMs: 200,
          }}
        />
        {pendingPromotion && (
          <div className="promotion-overlay" onClick={cancelPromotion}>
            <div
              className="promotion-dialog"
              onClick={(e) => e.stopPropagation()}
            >
              <p>Promote to:</p>
              <div className="promotion-pieces">
                {PROMOTION_PIECES.map(({ piece, label }) => (
                  <button
                    key={piece}
                    className="promotion-btn"
                    onClick={() => handlePromotionChoice(piece)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      <div className={`status${error ? " error" : ""}`}>
        {error
          ? error
          : isThinking
            ? "AI is thinking..."
            : pendingPromotion
              ? "Choose promotion piece"
              : statusLabel(status)}
      </div>
      <div className="controls">
        <select
          value={engine}
          onChange={(e) => setEngine(e.target.value)}
          disabled={isThinking}
        >
          {availableEngines.map((eng) => (
            <option key={eng} value={eng}>
              {eng === "random" ? "Random" : eng.toUpperCase().replace("_", " ")}
            </option>
          ))}
        </select>
        <button onClick={resetGame}>New Game</button>
      </div>
    </div>
  );
}
