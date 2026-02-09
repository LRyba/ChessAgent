const API_BASE = "http://localhost:8000";

export interface NewGameResponse {
  fen: string;
  status: string;
}

export interface MoveResponse {
  fen: string;
  ai_move: string | null;
  status: string;
}

export interface EnginesResponse {
  engines: string[];
}

export async function newGame(): Promise<NewGameResponse> {
  const res = await fetch(`${API_BASE}/new_game`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to start new game");
  return res.json();
}

export async function sendMove(
  fen: string,
  move: string,
  engine: string
): Promise<MoveResponse> {
  const res = await fetch(`${API_BASE}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen, move, engine }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Move failed");
  }
  return res.json();
}

export async function fetchEngines(): Promise<EnginesResponse> {
  const res = await fetch(`${API_BASE}/engines`);
  if (!res.ok) throw new Error("Failed to fetch engines");
  return res.json();
}
