.PHONY: install install-ml test lint api ui download prepare audit train-fast train-accurate evaluate

PYTHON ?= python3
SOURCE_DATASET ?= data/bone-fracture/bone fracture detection.v4-v4.yolov8
DATA_ROOT ?= data/bone-fracture-detect-one-class
DATA_YAML ?= $(DATA_ROOT)/data.yaml

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

install-ml:
	$(PYTHON) -m pip install -r requirements-ml.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

api:
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(PYTHON) -m streamlit run frontend/app.py

download:
	$(PYTHON) scripts/download_dataset.py

prepare:
	$(PYTHON) scripts/prepare_detection_dataset.py \
		--source "$(SOURCE_DATASET)" \
		--output "$(DATA_ROOT)" \
		--single-class

audit:
	$(PYTHON) scripts/audit_dataset.py --data $(DATA_YAML) --output reports/data_audit.json

train-fast:
	$(PYTHON) scripts/train.py --data $(DATA_YAML) --profile fast

train-accurate:
	$(PYTHON) scripts/train.py --data $(DATA_YAML) --profile accurate

evaluate:
	$(PYTHON) scripts/evaluate.py --data $(DATA_YAML) --split test
