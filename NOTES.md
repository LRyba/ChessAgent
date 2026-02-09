# NOTES.md — Dziennik projektu

## Ukończone etapy

### Etap 0 — Przygotowanie środowiska

**Co zrobiliśmy:**
- Struktura projektu: `backend/{api,engine,training,models}`, `frontend/`, `datasets/`, `notebooks/`, `experiments/`
- Python 3.12.3 w `venv/`, PyTorch 2.10.0+cpu (dev na Linuxie), CUDA na Windowsie (RTX 3050)
- `requirements.txt` bez torcha — torch instalowany osobno przez `--index-url` (wersja CPU/CUDA zależy od maszyny)
- `verify_setup.py` — skrypt walidujący środowisko (Python >= 3.11, import pakietów, struktura katalogów, GPU detection)
- `Makefile` z targetami: `setup`, `install`, `verify`, `backend`, `frontend`

**Problemy i rozwiązania:**

| Problem | Rozwiązanie |
|---------|-------------|
| `numpy >= 2.4` powoduje `ImportError: cannot load module more than once per process` przy imporcie torcha | Pinowanie `numpy<2.3` w `requirements.txt` |
| `experiments/` jest w `.gitignore`, ale katalog potrzebny w repo | `git add -f experiments/.gitkeep` — force-add placeholdera |
| Torch ma różne warianty CPU/CUDA, nie można dać jednej linii w `requirements.txt` | Torch poza `requirements.txt`, instalowany osobno w `Makefile` z odpowiednim `--index-url` |

---

### Etap 1 — Aplikacja webowa (random AI)

**Co zrobiliśmy:**

Backend (FastAPI):
- `backend/engine/base.py` — abstrakcyjna klasa `BaseEngine` z metodą `select_move(board) -> Move`
- `backend/engine/random_engine.py` — `RandomEngine` losujący legalny ruch
- `backend/api/main.py` — endpointy `POST /new_game` i `POST /move`, CORS dla dev frontendu, detekcja statusu gry (mat, pat, remis)

Frontend (React + Vite):
- `react-chessboard` + `chess.js`
- Drag & drop figur, walidacja legalności po stronie klienta i serwera
- Dialog promocji pionka (hetman/wieża/goniec/skoczek)
- Status gry: "AI is thinking...", "Checkmate!", "Stalemate!", "Draw!"
- Przycisk "New Game"

**Decyzje projektowe:**
- Gracz zawsze gra białymi (upraszcza UI, wystarczy do celów pracy)
- Abstrakcyjna klasa `BaseEngine` pozwala łatwo podmienić silnik — random → SL → SL+RL
- Komunikacja REST (FEN + UCI move), bezstanowy serwer
- Walidacja ruchów podwójna: chess.js na froncie (UX) + python-chess na backendzie (bezpieczeństwo)

---

### Etap 2 — Parsowanie PGN i tworzenie datasetu

**Dane źródłowe:**
- `datasets/lichess_elite_2023-12.pgn` — 943K partii, 245 MB, gracze 2300–2900 ELO
- Wysokie ELO = partie uczą struktury gry zamiast chaosu (wg TIPS.md: jakość > ilość)
- Cel: 20K partii → ~1.3M pozycji treningowych

**Utworzone pliki:**

1. **`backend/training/data.py`** — narzędzia konwersji:
   - `board_to_tensor(board)` → tensor `(13, 8, 8)` float32
   - `move_to_index(move)` → int 0..4095
   - `index_to_move(index, board=None)` → `chess.Move`

2. **`backend/training/pgn_dataset.py`** — CLI do przetwarzania PGN:
   - Streaming `chess.pgn.read_game()` — game by game, nie ładuje całego pliku
   - Zapis batchami co 50K pozycji jako `.npz` (kompresowane NumPy)
   - `metadata.json` z parametrami i statystykami
   - Progress co 1000 gier

3. **`backend/training/dataset.py`** — PyTorch `ChessDataset`:
   - Ładuje wszystkie `batch_*.npz` z katalogu, konkatenuje w pamięci
   - Zwraca `(board_tensor, move_index)` — gotowe do DataLoadera

**Decyzje projektowe:**

| Decyzja | Uzasadnienie |
|---------|--------------|
| **13 kanałów** (12 figur + side-to-move) zamiast 12 z PLAN.md | Kanał 12 = `1.0` gdy ruch białych, `0.0` gdy czarnych. Bez tego model nie wie czyja kolej — krytyczne przy stałej perspektywie |
| **Stała perspektywa** (biały na dole), bez flipowania planszy | Prostsze, mniej błędów. Kanały 0–5 = białe, 6–11 = czarne. Model sam uczy się grać obiema stronami |
| **Obie strony** — pozycje zarówno z ruchów białych jak i czarnych | Podwaja dataset. Model widzi planszę z obu perspektyw |
| **Kodowanie ruchów: `from_sq * 64 + to_sq` = 4096 klas** | Najprostsze podejście. Nie koduje promocji — zawsze zakładamy hetmana (underpromocja < 0.1% partii) |
| **Pomijanie 10 pierwszych półruchów** | Unikanie overfittingu do debiutów (wg TIPS.md). Te same otwarcia powtarzają się tysiące razy |
| **Format `.npz`** (kompresowane NumPy) | Przenośne Linux ↔ Windows. Dobra kompresja na rzadkich tensorach binarnych. Prostsze niż HDF5 |
| **Zapis batchami** (50K pozycji/batch) | Nie ładujemy milionów tensorów do RAM naraz podczas tworzenia datasetu |
| **Cały dataset w pamięci** przy treningu | 1.3M pozycji × 13 × 8 × 8 × 4B ≈ 3.3 GB — zmieści się w RAM. Prostsza implementacja niż lazy loading |
| **Pomijanie gier z wynikiem `*`** | Niedokończone partie nie dają pełnego kontekstu |

**Weryfikacja:**
- Unit testy `data.py`: tensor shape `(13,8,8)`, 32 figury na starcie, side-to-move flip, round-trip move encoding, auto-promocja
- Test run 100 gier: 8615 pozycji w 1.3s, 1 batch `.npz`
- PyTorch DataLoader: batching i shuffling działają poprawnie

**Uruchomienie pełnego datasetu:** `make dataset`

---

### Etap 3 — Model SL (Supervised Learning)

**Architektura ChessCNN:**
- 4× ConvBlock: `Conv2d(3×3, pad=1) → BatchNorm2d → ReLU`, kanały: 13 → 64 → 128 → 128 → 256
- Bez poolingu — w szachach każde pole ma znaczenie, nie tracimy rozdzielczości
- `Flatten → Linear(16384, 1024) → ReLU → Dropout(0.3) → Linear(1024, 4096)`
- Bez Softmax na wyjściu — `CrossEntropyLoss` przyjmuje surowe logity
- ~21.5M parametrów

**Utworzone pliki:**

1. **`backend/models/chess_cnn.py`** — klasa `ChessCNN(nn.Module)`
2. **`backend/training/train_sl.py`** — skrypt treningowy CLI z early stopping, ReduceLROnPlateau, logowaniem top-1/top-5 accuracy
3. **`backend/engine/neural_engine.py`** — `NeuralEngine(BaseEngine)` z maskowaniem nielegalnych ruchów
4. **`backend/training/data.py`** — dodana funkcja `legal_move_mask(board)` → boolean mask `(4096,)`

**Integracja z webapp:**
- `backend/api/main.py` — dynamiczne wykrywanie modeli: każdy plik `*_model.pt` w `models/` staje się silnikiem (np. `sl_model.pt` → "sl", `sl_100k_model.pt` → "sl_100k")
- Cache silników — model ładowany raz, reużywany per-request
- `GET /engines` — zwraca listę dostępnych silników
- Frontend: dropdown z wyborem silnika, parametr `engine` w request body

**Hiperparametry treningu:**

| Parametr | Wartość |
|----------|---------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 512 |
| Epoki | 10 (z early stopping, patience=3) |
| Split | 90% train / 10% val |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |

**Wyniki treningu SL 20k (na RTX 3050, ~25 min):**

```
Train: 1,408,318  Val: 156,479

Epoch  Train Loss    Val Loss   Top-1   Top-5         LR    Time
-----------------------------------------------------------------
    1      4.3793      3.3992  20.74%  48.55%    1.0e-03  159.5s
    2      3.5028      3.0579  25.48%  54.96%    1.0e-03  152.4s
    3      3.2162      2.8935  27.97%  58.98%    1.0e-03  152.0s
    4      3.0060      2.7827  29.66%  61.15%    1.0e-03  152.0s
    5      2.8241      2.7097  31.02%  63.08%    1.0e-03  152.0s
    6      2.6681      2.6602  31.96%  64.48%    1.0e-03  152.2s
    7      2.5313      2.6442  32.66%  65.14%    1.0e-03  152.1s
    8      2.4119      2.6161  33.05%  65.59%    1.0e-03  152.2s
    9      2.3042      2.6267  33.10%  65.76%    1.0e-03  152.2s
   10      2.2069      2.6532  33.42%  66.08%    1.0e-03  152.1s
```

**Wnioski z treningu SL 20k:**
- Najlepszy model z epoki 8 (val loss = 2.6161)
- Top-1 accuracy ~33% — co trzeci ruch identyczny z ludzkim
- Top-5 accuracy ~66% — prawdziwy ruch w top-5 propozycji modelu w 2/3 przypadków
- Overfitting widoczny od epoki 9 (val loss rośnie, train loss nadal spada)
- LR nie został zredukowany — val loss spadał wystarczająco stabilnie
- Model gra sensowne ruchy, ale popełnia poważne błędy taktyczne — oczekiwane dla czystego SL

**Obserwacje z gry:**
- Model nauczył się podstaw: rozwój figur, kontrola centrum, bicie
- Błędy: oddawanie materiału, brak obrony przed matami — SL naśladuje statystycznie, nie rozumie konsekwencji
- To jest baseline — etap RL powinien poprawić decyzje taktyczne

**Trening SL 100k (w toku):**
- 100K gier zamiast 20K → ~7M pozycji treningowych (5×)
- Szacowany czas treningu: ~2h (5× dłuższy per epoka)
- Oczekiwanie: mniej overfittingu, wyższy top-1/top-5, lepsze uogólnianie
- Model zapisze się jako `models/sl_100k_model.pt` → silnik "sl_100k" w webapp

---

### Etap 4 — Reinforcement Learning (Self-Play)

**Algorytm: REINFORCE (policy gradient)**

Najprostszy policy gradient — model gra sam ze sobą, zbiera trajektorie, aktualizuje wagi na podstawie wyniku partii.

```
Dla każdej iteracji:
  1. Zagraj N partii (self-play, sampling z polityki)
  2. Zbierz trajektorie: (stan, akcja, strona, wynik)
  3. Oblicz loss: -log(pi(a|s)) * reward_normalized
  4. Backpropagation + aktualizacja wag
```

**Utworzone pliki:**

1. **`backend/training/self_play.py`** — generowanie partii self-play:
   - `GameRecord` dataclass — przechowuje trajektorię gry (states, actions, log_probs, sides, result)
   - `play_one_game(model, device, temperature, max_moves)` — rozgrywa jedną partię
   - Sampling z softmax (z temperaturą) zamiast argmax — eksploracja
   - Legal move masking — nielegalne ruchy maskowane `-inf`
   - `board.is_game_over(claim_draw=True)` — zapobiega nieskończonym cyklom

2. **`backend/training/train_rl.py`** — pętla treningowa REINFORCE z CLI:
   - Zbieranie gier → flatten trajektorii → REINFORCE update
   - Normalizacja nagród: `(R - mean) / (std + eps)` — redukcja wariancji
   - Entropy bonus (0.01) — zapobiega kolapsowi polityki
   - Gradient clipping (`max_norm=1.0`) — stabilność treningu
   - Checkpointy co 50 iteracji → `experiments/rl_logs/rl_iter_N.pt`
   - CSV log → `experiments/rl_logs/training_log.csv` (do wykresów w pracy)
   - Logowanie na konsolę: iter, W/B/D, avg length, loss, entropy, time

**Hiperparametry:**

| Parametr | Wartość | Uzasadnienie |
|----------|---------|--------------|
| Optimizer | Adam | Spójne z SL |
| Learning rate | 1e-5 | Mały — unikamy catastrophic forgetting |
| Iteracje | 500 | ~28 min na RTX 3050 |
| Gier/iterację | 32 | ~2560 stanów/batch |
| Temperatura | 1.0 | Balans eksploracja/eksploatacja |
| Max ruchów/grę | 200 | Limit → remis po przekroczeniu |
| Entropy coeff | 0.01 | Mały bonus za eksplorację |
| Gradient clipping | 1.0 | Stabilność |
| Checkpoint co | 50 iteracji | Porównanie pośrednich etapów RL |

**Kluczowe decyzje projektowe:**

| Decyzja | Uzasadnienie |
|---------|--------------|
| **REINFORCE** (nie PPO/A2C) | Najprostszy policy gradient, wystarczający do porównania SL vs SL+RL w pracy magisterskiej |
| **Sampling** zamiast argmax | Argmax = brak eksploracji → model nie odkryje lepszych ruchów. Multinomial sampling z softmax |
| **Nagroda +1/0/-1** z perspektywy gracza | Ruch białych × wynik białych, ruch czarnych × (-wynik białych). Model uczy się wygrywać obiema stronami |
| **Normalizacja nagród** | Odejmij średnią, podziel przez std — zmniejsza wariancję gradientów, przyspiesza zbieżność |
| **Entropy bonus** | Bez niego model szybko kolapsuje do deterministycznej polityki, tracąc zdolność eksploracji |
| **LR 1e-5** (100× mniejszy niż SL) | RL fine-tuning — zbyt duży LR niszczy wiedzę z SL (catastrophic forgetting) |
| **Ten sam model ChessCNN** | Bez zmian w architekturze — RL tylko aktualizuje wagi. Porównanie SL vs SL+RL fair |

**Reużyte narzędzia z etapów 1–3:**
- `board_to_tensor()`, `move_to_index()`, `index_to_move()`, `legal_move_mask()` z `data.py`
- `ChessCNN` z `chess_cnn.py` — ten sam model, ładujemy wagi SL jako punkt startowy

**Uruchomienie:**
```bash
# Trening RL na modelu SL 20k
make train-rl

# Trening RL na modelu SL 100k
make train-rl-100k

# Ręcznie z parametrami
python -m backend.training.train_rl \
    --checkpoint models/sl_model.pt \
    --output models/rl_model.pt \
    --iterations 500 --games-per-iter 32 --lr 1e-5
```

**Wynik:** `models/rl_model.pt` → automatycznie wykryty jako silnik "rl" w webapp (dropdown).

**Wczesne obserwacje z treningu RL (model 20k, po ~63 iteracjach):**
- Remisy dominują: 24–31 z 32 gier (limit 200 ruchów lub pat)
- Mało rozstrzygniętych partii: W+B = 3–6 z 32
- AvgLen stabilna ~160–178, Loss blisko zera (-0.03 do +0.07)
- Entropia lekko rośnie: 2.7 → 2.85–2.89 — model jeszcze eksploruje
- REINFORCE z sparse reward (tylko mat) bardzo wolny — sygnał z ~5 partii na 32

**Rozważania: limit ruchów i sparse reward:**
- `board.is_game_over(claim_draw=True)` już obsługuje 50-move rule, threefold repetition, stalemate, insufficient material
- 50-move rule resetuje się przy każdym biciu/ruchu pionem — semi-losowa gra ciągle resetuje counter
- Partie trwają ~165 ruchów nie bo limit jest za niski, ale bo model nie umie matować
- Podniesienie limitu (np. 500) → głównie dłuższe partie i remisy z 50-move rule, minimalny zysk w sygnale uczącym
- Prawdziwy problem: nagroda +1/-1 tylko przy macie, a mat jest trudny nawet przy nierównej grze

---

### Etap 5 — Ewaluacja i wyniki RL

**Utworzone pliki:**

1. **`backend/engine/stockfish_engine.py`** — wrapper Stockfish UCI (`BaseEngine`), konfigurowalne depth i time_limit
2. **`backend/training/evaluate.py`** — skrypt ewaluacyjny CLI: mecze między silnikami, CSV, PGN
3. **`Makefile`** — targety `evaluate` i `evaluate-full`

**Poprawki w trakcie:**
- **`NeuralEngine`** — dodano parametr `temperature` (default 0.0 = argmax dla API, 0.5 dla ewaluacji)
  - Przy argmax obie strony grały deterministycznie → 200 gier = 2 unikalne partie × 100 powtórzeń = same remisy
  - Temperature 0.5 daje zróżnicowane partie zachowując charakter gry modelu
- **`train_rl.py`** — mini-batching w `reinforce_update` (chunki po 1024 stanów, gradient accumulation)
  - CUDA OOM na iteracji 271 (RTX 3050 4GB) — cały batch ~5000 stanów naraz przekraczał VRAM
  - Fix: te same gradienty matematycznie, szczytowe VRAM ~5x niższe

**Wyniki ewaluacji SL:**

| Mecz | Gier | Engine 1 | Wygrane | Engine 2 | Wygrane | Remisy |
|------|------|----------|---------|----------|---------|--------|
| SL vs SL_100k | 200 | sl | 3 (1.5%) | sl_100k | 122 (61.0%) | 75 (37.5%) |
| SL_100k vs Stockfish d1 | 1000 | sl_100k | 2 (0.2%) | stockfish | 612 (61.2%) | 386 (38.6%) |

- SL_100k wyraźnie silniejszy od SL (5× więcej danych treningowych)
- Subiektywnie: SL_100k robi błędy taktyczne ale wymaga skupienia żeby wygrać
- 2 wygrane vs Stockfish = Stockfish depth 1 wpadł w mata w 1 ruch (brak lookahead)

**Trening RL_100k (2000 iteracji, nocny) — analiza logu:**

Zagregowane statystyki co 100 iteracji:

| Iteracje | W avg | B avg | D avg | AvgLen | Loss | Entropy |
|----------|-------|-------|-------|--------|------|---------|
| 1–100 | 5.8 | 4.0 | 22.2 | 150.8 | -0.06 | 2.42 |
| 101–200 | 5.5 | 3.5 | 23.0 | 151.5 | 0.01 | 2.86 |
| 201–300 | 5.7 | 3.0 | 23.3 | 151.6 | -0.01 | 3.40 |
| 301–400 | 6.0 | 2.3 | 23.8 | 154.4 | -0.14 | 3.87 |
| 401–500 | 6.5 | 1.8 | 23.7 | 154.0 | -0.11 | 4.37 |
| 501–600 | 6.8 | 1.4 | 23.9 | 154.4 | -0.34 | 4.76 |
| 601–700 | 8.2 | 1.0 | 22.8 | 150.2 | -0.47 | 4.96 |
| 701–800 | 9.6 | 0.6 | 21.8 | 145.6 | -0.42 | 5.13 |
| 801–900 | 11.0 | 0.4 | 20.6 | 141.3 | -0.57 | 5.20 |
| 901–1000 | 13.2 | 0.2 | 18.6 | 130.6 | -0.76 | 5.13 |
| 1001–1100 | 14.4 | 0.1 | 17.4 | 124.2 | -0.95 | 5.15 |
| 1101–1200 | 17.3 | 0.1 | 14.6 | 115.5 | -1.19 | 5.16 |
| 1201–1300 | 17.7 | 0.1 | 14.2 | 109.9 | -1.28 | 5.18 |
| 1301–1400 | 19.0 | 0.1 | 12.9 | 103.4 | -1.42 | 5.09 |
| 1401–1500 | 20.4 | 0.1 | 11.5 | 98.2 | -1.64 | 5.04 |
| 1501–1600 | 20.9 | 0.1 | 11.0 | 93.7 | -1.84 | 5.01 |
| 1601–1700 | 21.8 | 0.0 | 10.2 | 90.7 | -1.83 | 4.96 |
| 1701–1800 | 21.6 | 0.0 | 10.4 | 91.1 | -2.06 | 4.97 |
| 1801–1900 | 22.0 | 0.0 | 9.9 | 88.4 | -2.01 | 4.90 |
| 1901–2000 | 22.6 | 0.0 | 9.4 | 86.2 | -2.12 | 4.88 |

Przebieg degeneracji — fazy:
1. **Stagnacja (1–200):** W~6, B~4, D~23. Mało matów, loss~0. Model niewiele się uczy
2. **Asymetria (200–400):** B spada 4→2, W stabilne ~6. Czarne zaczynają przegrywać
3. **Czarne zanikają (400–700):** B: 1.8→1.0. Entropia rośnie 4.4→5.0
4. **White dominuje (700–1200):** W rośnie 10→17, B→0, AvgLen spada 145→115
5. **Pełna degeneracja (1200–2000):** W~22, B=0, AvgLen~86

Kluczowe obserwacje:
- Nie było "złotego punktu" — czarne nigdy nie wygrywały dużo, degeneracja stopniowa od początku
- Entropia monotonnie rośnie (2.4→5.2) i nigdy nie spada — czarne grają coraz bardziej losowo
- AvgLen spada liniowo (151→86) — nie bo model lepiej gra, a bo czarne szybciej "przegrywają"

**Wyniki RL_100k vs Stockfish:**

| Mecz | Gier | Engine 1 | Wygrane | Engine 2 | Wygrane | Remisy |
|------|------|----------|---------|----------|---------|--------|
| RL_100k vs Stockfish d1 | 100 | rl_100k | 0 (0%) | stockfish | 83 (83%) | 17 (17%) |

**RL_100k gorszy od SL_100k!** (83% przegranych vs 61.2%, 17% remisów vs 38.6%)

**Diagnoza: policy collapse (degeneracja self-play)**

Model nauczył się "kolaborować sam ze sobą" zamiast grać lepiej:
- **Białe:** agresywna gra → szybki mat
- **Czarne:** zdegenerowana polityka (skoczek oscyluje w tę i z powrotem) → "poddaje się"
- REINFORCE wzmacnia oba zachowania bo White +1 w każdej partii
- Dowody: 0 wygranych czarnych w treningu, gorszy od SL vs Stockfish, bezsensowne ruchy jako czarne

**Wartość dla pracy:**
- RL zmienił politykę (nie jest to brak efektu)
- Self-play jednego modelu bez zabezpieczeń → degeneracja — klasyczny problem w literaturze
- Sparse reward + self-play = perverse incentives

**Potencjalne rozwiązania:**

| Podejście | Opis | Trudność |
|-----------|------|----------|
| **Frozen opponent** | RL gra vs zamrożona kopia SL (nie aktualizowana) — zapobiega kolaboracji | Niska |
| **KL penalty** | Kara za oddalanie się od polityki SL: `loss += β * KL(π_RL \|\| π_SL)` | Średnia |
| **Early stopping** | Sprawdzić checkpointy co 50 iter — może wcześniejszy model był OK | Niska |
| **Opponent pool** | Pula przeciwników z różnych checkpointów (podejście AlphaStar) | Wysoka |

---

### Frozen opponent — implementacja

Rozwiązanie problemu policy collapse: model uczący gra vs zamrożona kopia SL (nigdy nie aktualizowana).

**Zmodyfikowane pliki:**

1. **`backend/training/self_play.py`** — nowa funkcja `play_one_game_vs_opponent(learner, opponent, device, learner_white, ...)`:
   - Learner gra jedną stroną, frozen opponent drugą
   - Tylko ruchy learnera zapisywane w trajektorii (nie trenujemy na ruchach przeciwnika)
   - Kolory alternowane co partię (`learner_white = i % 2 == 0`)

2. **`backend/training/train_rl.py`**:
   - Flaga `--frozen-opponent` — ładuje drugą kopię modelu SL z tych samych wag
   - Frozen opponent: `eval()`, nigdy nie aktualizowany
   - `collect_games()` — jeśli opponent podany, używa `play_one_game_vs_opponent` zamiast `play_one_game`

3. **`Makefile`** — targety `train-rl-frozen` i `train-rl-100k-frozen` (2000 iter)

**Dlaczego to powinno pomóc:**
- Przeciwnik nie "współpracuje" — zawsze gra na poziomie SL
- Model musi nauczyć się wygrywać z sensownym graczem, nie z samym sobą
- Nie ma pętli zwrotnej: lepsze białe ≠ gorsze czarne

**VRAM:** drugi model ~82 MB, nie tworzy grafów gradientów → znikomy koszt pamięci

**Uruchomienie:**
```bash
python -m backend.training.train_rl \
    --checkpoint models/sl_100k_model.pt \
    --output models/rl_100k_frozen_model.pt \
    --iterations 2000 --games-per-iter 32 --lr 1e-5 --frozen-opponent
```

### Wyniki frozen opponent (500 iteracji)

Zagregowane co 50 iteracji:

| Iteracje | W avg | B avg | D avg | AvgLen | Loss | Entropy |
|----------|-------|-------|-------|--------|------|---------|
| 1–50 | 6.6 | 6.2 | 19.2 | 70.0 | 0.37 | 2.25 |
| 51–100 | 6.8 | 5.6 | 19.6 | 70.9 | -0.16 | 2.34 |
| 101–150 | 7.3 | 6.1 | 18.6 | 70.3 | 0.31 | 2.36 |
| 151–200 | 6.5 | 6.4 | 19.1 | 69.5 | 0.96 | 2.48 |
| 201–250 | 7.0 | 5.7 | 19.3 | 70.2 | -4.38 | 2.54 |
| 251–300 | 6.7 | 6.3 | 19.0 | 70.0 | 1.18 | 2.51 |
| 301–350 | 6.6 | 6.2 | 19.2 | 69.9 | -0.52 | 2.66 |
| 351–400 | 6.8 | 6.3 | 18.9 | 70.7 | -1.55 | 2.67 |
| 401–450 | 7.2 | 5.8 | 19.1 | 70.0 | 0.28 | 2.64 |
| 451–500 | 6.5 | 6.1 | 19.4 | 69.6 | -0.51 | 2.77 |

Ewaluacja RL_100k_frozen vs Stockfish (depth 1, 100 gier):
- RL_100k_frozen: 0 wygranych, 69 przegranych, 31 remisów
- Gorszy niż SL_100k (69% przegranych vs 61.2%) ale lepszy niż zdegenerowany self-play (83%)

**Wnioski:**
- Frozen opponent rozwiązał problem degeneracji — W≈B symetryczne przez cały trening
- Ale model się nie uczy — W/B/D płaskie, brak trendu, loss bardzo szumny
- ~19/32 gier to remisy (reward=0) → za mało sygnału uczącego
- **Potwierdzenie:** problem leży w sparse reward, nie w self-play collapse
- Potrzebny reward shaping — nagroda nie tylko za mata, ale też za przewagę materialną

---

## Co dalej

- Zaimplementować reward shaping (nagroda za materiał) + frozen opponent
- Uruchomić trening na noc
- **Etap 6** — Analiza do pracy pisemnej

---

## Problemy do zapamiętania

1. **numpy/torch kompatybilność** — zawsze `numpy<2.3`, inaczej torch się nie zaimportuje
2. **Torch osobno** — nie w `requirements.txt`, bo wariant CPU/CUDA zależy od maszyny
3. **Promotion w move encoding** — `from*64+to` nie rozróżnia promocji. `index_to_move()` automatycznie dodaje `promotion=QUEEN` gdy pion dochodzi do końca. Wystarcza bo underpromocja jest marginalna
4. **PGN streaming** — `chess.pgn.read_game()` czyta jedną grę na raz. Nie parsować całego pliku!
5. **Maskowanie nielegalnych ruchów** — krytyczne w NeuralEngine. Bez tego model próbuje grać nielegalne ruchy i wygląda na niedziałający
6. **PowerShell na Windows** — `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` żeby aktywować venv. Backslash `\` nie działa jako kontynuacja linii — komendy w jednej linii
7. **CUDA na Windows** — `nvidia-smi` → sprawdzić wersję CUDA → `cu121` dla CUDA 12.x+
