.PHONY: setup install verify backend frontend dataset dataset-100k train train-100k train-rl train-rl-100k train-rl-frozen train-rl-100k-frozen train-rl-100k-shaped evaluate evaluate-full

# Full setup: create venv, install everything
setup: venv install verify

venv:
	python3 -m venv venv

install:
	. venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cpu
	. venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

verify:
	. venv/bin/activate && python verify_setup.py

backend:
	. venv/bin/activate && uvicorn backend.api.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dataset:
	. venv/bin/activate && python -m backend.training.pgn_dataset \
		--pgn datasets/lichess_elite_2023-12.pgn \
		--output datasets/processed \
		--max-games 20000 \
		--skip-plies 10

dataset-100k:
	. venv/bin/activate && python -m backend.training.pgn_dataset --pgn datasets/lichess_elite_2023-12.pgn --output datasets/processed_100k --max-games 100000 --skip-plies 10

train:
	. venv/bin/activate && python -m backend.training.train_sl \
		--data datasets/processed \
		--output models/sl_model.pt \
		--epochs 10 --batch-size 512 --lr 1e-3

train-100k:
	. venv/bin/activate && python -m backend.training.train_sl \
		--data datasets/processed_100k \
		--output models/sl_100k_model.pt \
		--epochs 10 --batch-size 512 --lr 1e-3

train-rl:
	. venv/bin/activate && python -m backend.training.train_rl \
		--checkpoint models/sl_model.pt \
		--output models/rl_model.pt \
		--iterations 500 --games-per-iter 32 --lr 1e-5

train-rl-100k:
	. venv/bin/activate && python -m backend.training.train_rl \
		--checkpoint models/sl_100k_model.pt \
		--output models/rl_100k_model.pt \
		--iterations 500 --games-per-iter 32 --lr 1e-5

train-rl-frozen:
	. venv/bin/activate && python -m backend.training.train_rl \
		--checkpoint models/sl_model.pt \
		--output models/rl_frozen_model.pt \
		--iterations 2000 --games-per-iter 32 --lr 1e-5 --frozen-opponent

train-rl-100k-frozen:
	. venv/bin/activate && python -m backend.training.train_rl \
		--checkpoint models/sl_100k_model.pt \
		--output models/rl_100k_frozen_model.pt \
		--iterations 2000 --games-per-iter 32 --lr 1e-5 --frozen-opponent

train-rl-100k-shaped:
	. venv/bin/activate && python -m backend.training.train_rl \
		--checkpoint models/sl_100k_model.pt \
		--output models/rl_100k_shaped_model.pt \
		--iterations 2000 --games-per-iter 32 --lr 1e-5 --frozen-opponent --reward-shaping

evaluate:
	. venv/bin/activate && python -m backend.training.evaluate \
		--engine1 sl --engine2 rl --games 500 --save-pgn

evaluate-full:
	. venv/bin/activate && python -m backend.training.evaluate \
		--engine1 sl --engine2 random --games 200 --save-pgn
	. venv/bin/activate && python -m backend.training.evaluate \
		--engine1 rl --engine2 random --games 200 --save-pgn
	. venv/bin/activate && python -m backend.training.evaluate \
		--engine1 sl --engine2 rl --games 500 --save-pgn
