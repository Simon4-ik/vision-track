# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionTrack is a real-time person detection, tracking, and counting system exposed as a Streamlit dashboard. It processes up to 4 simultaneous video streams using YOLOv8n for detection, ByteTrack for multi-object tracking, and polygon-zone ROI counting for entry/exit events.

## Common Commands

```bash
# Setup (venv + deps + dataset download)
make setup

# Full pipeline: train -> ONNX export -> evaluate -> validate
make pipeline

# Launch Streamlit dashboard
make run-app
# or directly:
streamlit run app.py

# Individual steps
make train                          # Fine-tune YOLO (min 10 epochs enforced)
make export-onnx                    # Export best.pt to quantized ONNX
make eval                           # Evaluate precision/recall/F1
make validate                       # Audit that all artifacts exist and meet thresholds
make demo                           # Generate demo PNG + MP4 artifacts

# Override training params
make train EPOCHS=30 LR0=0.0005 BATCH=8
```

On Windows without `make`, run Python commands directly using `.venv/Scripts/python.exe` and `.venv/Scripts/streamlit.exe`.

## Architecture

### Detection Pipeline (per frame)

1. **Detection** (`models/yolo_person_detection.py`): Two detector backends sharing the same interface (`predict(frame) -> list[dict]`):
   - `YoloPersonDetector` — Ultralytics PyTorch, filters to COCO class 0 (person) at the YOLO predict level
   - `OnnxPersonDetector` — ONNX Runtime with letterbox preprocessing, manual NMS via `cv2.dnn.NMSBoxes`, auto-selects execution provider (CUDA > CoreML > CPU)
   - Both return dicts with keys: `xyxy`, `confidence`, `class_id`

2. **Tracking** (`utils/multi_stream_tracking_helpers.py`): `MultiStreamTracker` maintains independent `sv.ByteTrack` instances per stream. Converts detection dicts into `sv.Detections` and returns tracked detections with `tracker_id`.

3. **Counting** (`utils/counting_logic.py`): `RoiCounter` uses `sv.PolygonZone` with a time-based grace period (2s default) to avoid false OUT counts from tracker flicker. Counts IN on first center-in-polygon, counts OUT after sustained absence.

4. **Dashboard** (`app.py`): Streamlit app with `@st.cache_resource` for detectors/tracker. Runs inference every 4th frame (`INFER_EVERY=4`) and displays cached results on intermediate frames. Resizes display frames to max 800px width to reduce WebSocket overhead.

### Key Design Decisions

- Detectors are cached via `@st.cache_resource` — changing model parameters in the sidebar requires a Streamlit rerun to take effect
- The app writes final metrics to `reports/performance_metrics.json` at run completion, merging detection metrics from any prior evaluation
- Training enforces minimum 10 epochs; best weights are copied from YOLO's run directory to `models/checkpoints/best.pt`
- ROI counting uses `time.monotonic()` for grace period timing, making it FPS-independent

### Minimum Quality Thresholds (enforced by `validate_project.py`)

- Precision >= 0.85, Recall >= 0.80, F1 >= 0.85
- Average FPS >= 15 at 720p

## Dependencies

Python 3.10+. Core: `torch`, `ultralytics`, `supervision`, `opencv-python`, `onnxruntime`, `streamlit`. GPU auto-detected (CUDA > MPS > CPU fallback).
