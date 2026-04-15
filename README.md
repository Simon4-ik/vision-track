# VisionTrack

VisionTrack is an advanced computer vision proof-of-concept for **real-time person detection, tracking, and counting** across **multiple video streams**, exposed via an **interactive Streamlit web app**.

The goal is to simulate a smart city surveillance dashboard where operators can:
- Monitor crowd density across multiple public spaces (parks, stations, shopping areas)
- Track people in real time with unique IDs
- Count entries and exits through regions of interest (ROIs)
- Receive alerts when crowd thresholds are exceeded

## Key Features
- **YOLO-based person detection** — pre-trained on COCO, fine-tuned via transfer learning (YOLOv8n)
- **Supervision-powered tracking** — ByteTrack multi-object tracker with stable unique IDs
- **Multi-stream processing** — up to 4 simultaneous video feeds with independent controls
- **ROI-based entry/exit counting** — configurable polygon zones with grace-period logic
- **Streamlit dashboard** — per-stream detection/tracking/counting toggles, live FPS & latency metrics
- **ONNX Runtime backend** — quantized model export for optimized inference (~2-4x speedup)
- **Optional model pruning** — L1 unstructured pruning for 20-30% model size reduction
- **GPU acceleration** — automatic CUDA/MPS detection with safe CPU fallback
- **Comprehensive error handling** — graceful recovery from broken streams, logged to `logs/app_errors.log`

## Project Structure

```
vision-track/
├── data/
│   ├── raw_videos/          # Input video files for testing
│   ├── raw_images/          # Input image files for testing
│   └── coco_dataset/        # COCO8 dataset (downloaded via setup)
├── models/
│   ├── yolo_person_detection.py   # PyTorch & ONNX detector classes
│   ├── train_yolo.py              # Transfer learning training script
│   ├── export_onnx.py             # ONNX export with quantization
│   ├── prune_yolo.py              # Optional model pruning
│   ├── evaluate_yolo.py           # Precision/recall/F1 evaluation
│   ├── __init__.py
│   └── checkpoints/
│       ├── best.pt                # Fine-tuned YOLO weights
│       ├── best_quantized.onnx    # ONNX-exported model
│       └── config.yaml            # Training configuration
├── utils/
│   ├── data_loader.py                    # Video I/O utilities
│   ├── preprocessing.py                  # Resize, normalize, color conversion
│   ├── multi_stream_tracking_helpers.py  # ByteTrack tracker + FPS/latency stats
│   ├── counting_logic.py                # ROI entry/exit counting
│   ├── VisionTrack_Analysis.ipynb        # EDA and workflow notebook
│   └── __init__.py
├── reports/
│   ├── performance_metrics.json   # Detection precision, recall, F1, FPS, latency
│   └── demo_results/
│       ├── roi_counting_example.png
│       └── multi_stream_demo.mp4
├── logs/
│   └── app_errors.log             # Runtime error log
├── app.py                  # Streamlit dashboard entry point
├── generate_demos.py       # Generate demo PNG/MP4 artifacts
├── download_coco8.py       # Download COCO8 dataset
├── validate_project.py     # Audit validation script
├── Makefile                # Automation for train/export/eval/demo
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for GPU acceleration

### Step 1: Create Virtual Environment and Install Dependencies

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2: Download Dataset

```bash
python download_coco8.py
```

This downloads the COCO8 dataset to `data/coco_dataset/` with YOLO-format annotations.

### Step 3: Verify Installation

```bash
python -c "import torch, supervision, cv2, streamlit; print('All imports OK')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Usage

### Quick Start (Makefile)

```bash
make setup      # Install deps + download dataset
make pipeline   # Train -> export ONNX -> evaluate -> validate
make app        # Launch Streamlit dashboard
```

> **Windows users**: If `make` is unavailable, run the Python commands directly:
> `.\.venv\Scripts\python.exe download_coco8.py`, `.\.venv\Scripts\streamlit.exe run app.py`, etc.

### Training (Transfer Learning)

Fine-tune a pre-trained YOLOv8n model on the COCO8 dataset:

```bash
python models/train_yolo.py \
  --data data/coco_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 20 \
  --lr0 0.001
```

- Uses pre-trained COCO weights as starting point
- Small learning rate (0.001) preserves pre-trained features
- Minimum 10 epochs required; early stopping with patience=10
- Best weights saved to `models/checkpoints/best.pt`

### ONNX Export (Quantization)

```bash
python models/export_onnx.py \
  --weights models/checkpoints/best.pt \
  --output models/checkpoints/best_quantized.onnx
```

### Model Evaluation

```bash
python models/evaluate_yolo.py \
  --weights models/checkpoints/best.pt \
  --images-dir data/coco_dataset/images/val \
  --labels-dir data/coco_dataset/labels/val
```

Results saved to `reports/performance_metrics.json`. Minimum thresholds:
- Precision >= 0.85
- Recall >= 0.80
- F1-score >= 0.85
- Average FPS >= 15 (at 720p)

### Optional: Model Pruning

```bash
python models/prune_yolo.py \
  --weights models/checkpoints/best.pt \
  --output models/checkpoints/best_pruned.pt \
  --amount 0.2
```

### Generate Demo Artifacts

```bash
python generate_demos.py \
  --video data/raw_videos/demo.mp4 \
  --backend onnx
```

Produces `reports/demo_results/roi_counting_example.png` and `reports/demo_results/multi_stream_demo.mp4`.

### Launch the Streamlit App

```bash
streamlit run app.py
```

The dashboard allows you to:
- Select PyTorch or ONNX Runtime backend
- Configure confidence and IoU thresholds
- Set crowd alert thresholds
- Upload or connect up to 4 video streams
- Toggle detection, tracking, and ROI counting per stream
- Define custom ROI zones per stream
- View live FPS, latency, people count, and alert status

### Validation

```bash
python validate_project.py
```

Checks all required artifacts exist and metrics meet minimum thresholds.

## CUDA / GPU Support

VisionTrack automatically detects hardware:
- **CUDA GPU** — used when available (NVIDIA)
- **Apple MPS** — used on Apple Silicon Macs
- **CPU fallback** — always works, with lower FPS

```python
import torch
print("Using CUDA:", torch.cuda.is_available())
```

## Architecture Overview

1. **Detection**: YOLOv8n detects persons (COCO class 0) in each video frame
2. **Tracking**: ByteTrack (via supervision) assigns persistent IDs across frames
3. **Counting**: PolygonZone triggers count IN/OUT events with a 30-frame grace period
4. **Dashboard**: Streamlit renders annotated frames with live metrics per stream
5. **Optimization**: ONNX Runtime provides hardware-accelerated inference; optional pruning reduces model size
