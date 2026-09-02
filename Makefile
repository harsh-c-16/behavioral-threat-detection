PY := .venv/bin/python
# Max Transformer epochs; early stopping on the calibration period usually stops sooner.
EPOCHS ?= 20

.PHONY: setup data data-lanl train eval seeds test all

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip freeze > requirements.lock

# Synthetic LANL-shaped data: always works, no download.
data:
	$(PY) src/prepare_data.py --synthetic

# Real LANL slice. AUTH and REDTEAM are the links (or local paths) from
# https://csr.lanl.gov/data/cyber1/ ; see README for the split flags.
data-lanl:
	$(PY) src/download_data.py --auth "$(AUTH)" --redteam "$(REDTEAM)" --days 10
	$(PY) src/prepare_data.py

train:
	$(PY) src/train.py --model iforest
	$(PY) src/train.py --model transformer --epochs $(EPOCHS)

eval:
	$(PY) src/evaluate.py

# Seed 0 reuses the models from `make train`; seeds 1-2 retrain (LANL: ~35 min on CPU).
seeds:
	$(PY) src/seed_variance.py --seeds 0 1 2 --epochs $(EPOCHS)

test:
	$(PY) -m pytest -q tests

all: data train eval
