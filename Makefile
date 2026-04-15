PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON ?= $(VENV)/bin/python
VENV_PIP ?= $(VENV)/bin/pip
VENV_STREAMLIT ?= $(VENV)/bin/streamlit

# Training / model paths
DATA_YAML ?= data/coco_dataset/data.yaml
BASE_MODEL ?= yolov8n.pt
BEST_PT ?= models/checkpoints/best.pt
BEST_ONNX ?= models/checkpoints/best_quantized.onnx
PRUNED_PT ?= models/checkpoints/best_pruned.pt

# Training params
EPOCHS ?= 20
LR0 ?= 0.001
IMGSZ ?= 640
BATCH ?= 16
PATIENCE ?= 10

# Evaluation params
VAL_IMAGES ?= data/coco_dataset/images/val
VAL_LABELS ?= data/coco_dataset/labels/val

# Demo params
DEMO_VIDEO ?= data/raw_videos/demo.mp4
DEMO_BACKEND ?= onnx
DEMO_MAX_FRAMES ?= 300
DEMO_FPS ?= 18.5

.DEFAULT_GOAL := default

.PHONY: default help venv install run-app train export-onnx eval prune demo validate setup pipeline check-dirs

default: setup run-app

help:
	@echo "VisionTrack Make targets"
	@echo "  make                - Install deps (if needed) and run app"
	@echo "  make install        - Install Python dependencies"
	@echo "  make download       - Fetch COCO8 dataset and demo video"
	@echo "  make run-app        - Run Streamlit app"
	@echo "  make train          - Fine-tune YOLO model"
	@echo "  make export-onnx    - Export best.pt to ONNX"
	@echo "  make eval           - Evaluate precision/recall/F1"
	@echo "  make prune          - Optional model pruning"
	@echo "  make demo           - Generate demo PNG + MP4"
	@echo "  make validate       - Run audit validation checks"
	@echo "  make pipeline       - Train -> Export -> Eval -> Validate"
	@echo "  make setup          - Install + directory scaffolding"
	@echo ""
	@echo "Override variables example:"
	@echo "  make train DATA_YAML=data/coco_dataset/data.yaml EPOCHS=30 LR0=0.0005"
	@echo "  make demo DEMO_VIDEO=data/raw_videos/my_demo.mp4 DEMO_BACKEND=onnx"

check-dirs:
	@mkdir -p data/raw_videos data/raw_images data/coco_dataset
	@mkdir -p models/checkpoints utils reports/demo_results logs

setup: check-dirs install download

download: check-dirs
	$(VENV_PYTHON) download_coco8.py
	$(VENV_PYTHON) data/download_video.py

venv:
	@test -x "$(VENV_PYTHON)" || $(PYTHON) -m venv "$(VENV)"

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

run-app:
	$(VENV_STREAMLIT) run app.py

train: check-dirs
	$(VENV_PYTHON) models/train_yolo.py \
		--data $(DATA_YAML) \
		--model $(BASE_MODEL) \
		--epochs $(EPOCHS) \
		--imgsz $(IMGSZ) \
		--batch $(BATCH) \
		--lr0 $(LR0) \
		--patience $(PATIENCE)

export-onnx: check-dirs
	$(VENV_PYTHON) models/export_onnx.py \
		--weights $(BEST_PT) \
		--output $(BEST_ONNX)

eval: check-dirs
	$(VENV_PYTHON) models/evaluate_yolo.py \
		--weights $(BEST_PT) \
		--images-dir $(VAL_IMAGES) \
		--labels-dir $(VAL_LABELS)

prune: check-dirs
	$(VENV_PYTHON) models/prune_yolo.py \
		--weights $(BEST_PT) \
		--output $(PRUNED_PT) \
		--amount 0.2

demo: check-dirs
	$(VENV_PYTHON) generate_demos.py \
		--video $(DEMO_VIDEO) \
		--backend $(DEMO_BACKEND) \
		--max-frames $(DEMO_MAX_FRAMES) \
		--fps $(DEMO_FPS)

validate:
	$(VENV_PYTHON) validate_project.py

pipeline: train export-onnx eval validate

