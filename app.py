"""VisionTrack — Smart City Multi-Stream Person Tracking Dashboard.

Streamlit application that displays real-time video feeds with overlaid
YOLO person detection, ByteTrack multi-object tracking, and ROI-based
entry/exit counting. Supports both PyTorch and ONNX Runtime backends,
with live FPS/latency metrics and configurable crowd alert thresholds.

Run with: streamlit run app.py
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from models.yolo_person_detection import OnnxPersonDetector, YoloPersonDetector
from utils.counting_logic import RoiCounter, save_performance_metrics
from utils.data_loader import open_video_stream
from utils.multi_stream_tracking_helpers import MultiStreamTracker

try:
    import torch
except ImportError:
    torch = None


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app_errors.log"
DEFAULT_STREAMS = 2


def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def get_device_info() -> str:
    if torch is None:
        return "PyTorch not installed - running on CPU."
    try:
        if torch.cuda.is_available():
            return f"Using CUDA GPU: {torch.cuda.get_device_name(0)}"
        return "CUDA not available - running on CPU."
    except Exception as exc:  # pragma: no cover
        logging.exception("Error while checking CUDA availability: %s", exc)
        return "Error checking CUDA - assuming CPU."


@st.cache_resource
def get_torch_detector(model_path: str, conf: float, iou: float, device: str) -> YoloPersonDetector:
    return YoloPersonDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        device=device,
    )


@st.cache_resource
def get_onnx_detector(onnx_path: str, conf: float, iou: float) -> OnnxPersonDetector:
    return OnnxPersonDetector(
        onnx_path=onnx_path,
        conf_threshold=conf,
        iou_threshold=iou,
    )


@st.cache_resource
def get_tracker() -> MultiStreamTracker:
    return MultiStreamTracker()


@st.cache_data
def get_video_dimensions(source: str) -> tuple[int, int]:
    """Return (width, height) of a video source. Falls back to 1280x720."""
    try:
        src: str | int = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1280, 720


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        conf = det["confidence"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            out,
            f"person {conf:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
        )
    return out


def main() -> None:
    setup_logging()
    st.set_page_config(page_title="VisionTrack", layout="wide")
    st.title("VisionTrack - Smart City Multi-Stream Dashboard")
    st.caption("Real-time person detection, tracking, and ROI counting.")
    st.info(get_device_info())

    with st.sidebar:
        st.header("Global Controls")
        backend = st.selectbox("Model backend", ["PyTorch", "ONNX Runtime"])
        model_path = st.text_input("PyTorch weights", value="models/checkpoints/best.pt")
        onnx_path = st.text_input(
            "ONNX model",
            value="models/checkpoints/best_quantized.onnx",
        )
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
        iou_threshold = st.slider("IoU threshold", 0.1, 0.95, 0.5, 0.05)
        crowd_threshold = st.number_input("Crowd alert threshold", 1, 200, 20, 1)
        max_frames = st.number_input("Frames per run", 1, 5000, 300, 10)
        run = st.button("Run Streams")

    num_streams = st.number_input("Number of streams", 1, 4, DEFAULT_STREAMS, 1)
    stream_configs: list[dict[str, Any]] = []
    tabs = st.tabs([f"Stream {i + 1}" for i in range(num_streams)])

    for idx, tab in enumerate(tabs):
        with tab:
            source = st.text_input(
                f"Source path / camera index (Stream {idx + 1})",
                value=f"data/raw_videos/stream_{idx + 1}.mp4",
                key=f"source_{idx}",
            )
            enable_detection = st.checkbox("Enable detection", True, key=f"det_{idx}")
            enable_tracking = st.checkbox("Enable tracking", True, key=f"trk_{idx}")
            enable_counting = st.checkbox("Enable ROI counting", True, key=f"cnt_{idx}")

            # --- Streamlit UI: Define the Rectangle Area ---
            # Defaults are 10%–90% of the actual video frame so the zone
            # covers most of the scene regardless of resolution.
            vid_w, vid_h = get_video_dimensions(source)
            def_x1 = int(vid_w * 0.10)
            def_y1 = int(vid_h * 0.10)
            def_x2 = int(vid_w * 0.90)
            def_y2 = int(vid_h * 0.90)
            st.caption(f"Video resolution: {vid_w}×{vid_h}")

            # Reset button forces the ROI inputs back to the proportional defaults
            # (Streamlit caches widget values in session state, so defaults only
            # apply on the very first load — this button overrides stale values)
            if st.button(f"Reset ROI to defaults (Stream {idx + 1})", key=f"reset_roi_{idx}"):
                st.session_state[f"rx1_{idx}"] = def_x1
                st.session_state[f"ry1_{idx}"] = def_y1
                st.session_state[f"rx2_{idx}"] = def_x2
                st.session_state[f"ry2_{idx}"] = def_y2

            roi_x1 = st.number_input(f"Zone Top-Left X ({idx + 1})", 0, vid_w, def_x1, 1, key=f"rx1_{idx}")
            roi_y1 = st.number_input(f"Zone Top-Left Y ({idx + 1})", 0, vid_h, def_y1, 1, key=f"ry1_{idx}")
            roi_x2 = st.number_input(f"Zone Bottom-Right X ({idx + 1})", 0, vid_w, def_x2, 1, key=f"rx2_{idx}")
            roi_y2 = st.number_input(f"Zone Bottom-Right Y ({idx + 1})", 0, vid_h, def_y2, 1, key=f"ry2_{idx}")

            # Convert the two corners into a full 4-point polygon (rectangle)
            polygon_array = np.array([
                [int(roi_x1), int(roi_y1)],  # Top-Left
                [int(roi_x2), int(roi_y1)],  # Top-Right
                [int(roi_x2), int(roi_y2)],  # Bottom-Right
                [int(roi_x1), int(roi_y2)]   # Bottom-Left
            ])

            stream_configs.append(
                {
                    "stream_id": f"stream_{idx + 1}",
                    "source": source,
                    "enable_detection": enable_detection,
                    "enable_tracking": enable_tracking,
                    "enable_counting": enable_counting,
                    "roi_polygon": polygon_array,
                }
            )

    st.subheader("Live Streams")
    top_cols = st.columns(2)
    bottom_cols = st.columns(2)
    grid = [top_cols[0], top_cols[1], bottom_cols[0], bottom_cols[1]]
    frame_slots: dict[str, Any] = {}
    metric_slots: dict[str, dict[str, Any]] = {}
    for i in range(num_streams):
        stream_id = stream_configs[i]["stream_id"]
        with grid[i]:
            st.markdown(f"#### {stream_id.replace('_', ' ').title()}")
            frame_slots[stream_id] = st.empty()
            c1, c2, c3, c4 = st.columns(4)
            metric_slots[stream_id] = {
                "fps": c1.empty(),
                "latency": c2.empty(),
                "count": c3.empty(),
                "alert": c4.empty(),
                "roi": st.empty(),
                "msg": st.empty(),
            }

    if not run:
        st.write("Configure stream controls above and click `Run Streams`.")
        return

    if backend == "PyTorch":
        device = YoloPersonDetector.resolve_device()
        try:
            detector = get_torch_detector(model_path, conf_threshold, iou_threshold, device)
        except Exception as exc:
            logging.exception("Failed loading model %s: %s", model_path, exc)
            st.error(
                "PyTorch model load failed. Verify weights path or use a valid Ultralytics checkpoint."
            )
            return
    else:
        try:
            detector = get_onnx_detector(onnx_path, conf_threshold, iou_threshold)
        except Exception as exc:
            logging.exception("Failed loading ONNX model %s: %s", onnx_path, exc)
            st.error(
                "ONNX model load failed. Verify ONNX path or re-export with models/export_onnx.py."
            )
            return
    tracker = get_tracker()
    counters: dict[str, RoiCounter] = {}
    captures: dict[str, cv2.VideoCapture] = {}
    active_streams: list[dict[str, Any]] = []

    for cfg in stream_configs:
        # Pass the newly created polygon array instead of the line
        counters[cfg["stream_id"]] = RoiCounter(roi_polygon=cfg["roi_polygon"])
        source_input: str | int = cfg["source"]
        if source_input.isdigit():
            source_input = int(source_input)
        try:
            captures[cfg["stream_id"]] = open_video_stream(source_input)
            active_streams.append(cfg)
        except Exception as exc:
            logging.exception("Failed opening source %s: %s", source_input, exc)
            metric_slots[cfg["stream_id"]]["msg"].error(
                f"Could not open source: {source_input}"
            )

    if not active_streams:
        st.error("No valid stream source could be opened. Check logs/app_errors.log.")
        return

    # Run inference every INFER_EVERY *displayed* frames; skip intermediate ones.
    INFER_EVERY = 3
    # Max width to resize display frames before sending to browser.
    DISPLAY_MAX_W = 640
    # Pre-resize large frames (e.g. 2560x1440) to this height BEFORE
    # detection — YOLO resizes to 640 internally anyway, and smaller
    # source frames speed up read/copy/letterbox dramatically.
    PROCESS_MAX_H = 720

    # Read each video's native FPS so we can skip frames for real-time playback
    video_native_fps: dict[str, float] = {}
    for cfg in active_streams:
        vfps = captures[cfg["stream_id"]].get(cv2.CAP_PROP_FPS)
        video_native_fps[cfg["stream_id"]] = vfps if vfps and vfps > 0 else 25.0

    run_start_time = time.perf_counter()
    run_frame_counts: dict[str, int] = {cfg["stream_id"]: 0 for cfg in active_streams}
    run_latency_totals: dict[str, float] = {cfg["stream_id"]: 0.0 for cfg in active_streams}
    # Track video-position frames (including skipped) for accurate FPS metric
    video_frames_advanced: dict[str, int] = {cfg["stream_id"]: 0 for cfg in active_streams}
    # Cache last inference results so non-inference frames still show detections
    last_detections: dict[str, list] = {cfg["stream_id"]: [] for cfg in active_streams}
    last_tracked: dict[str, Any] = {}

    try:
        for _ in range(int(max_frames)):
            elapsed = time.perf_counter() - run_start_time
            live_active = 0
            for cfg in active_streams:
                stream_id = cfg["stream_id"]
                cap = captures[stream_id]

                # --- FRAME SKIPPING: advance video to match real-time ---
                native_fps = video_native_fps[stream_id]
                target_pos = int(elapsed * native_fps)
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frames_behind = target_pos - current_pos - 1
                # grab() decodes header only (no pixel decode) — very fast
                for _ in range(max(0, min(frames_behind, 30))):
                    if not cap.grab():
                        break

                ret, frame = cap.read()
                if not ret:
                    metric_slots[stream_id]["msg"].warning(
                        "Stream ended or frame retrieval failed."
                    )
                    continue
                live_active += 1

                # --- PRE-RESIZE large frames to 720p before detection ---
                fh, fw = frame.shape[:2]
                if fh > PROCESS_MAX_H:
                    rs = PROCESS_MAX_H / fh
                    frame = cv2.resize(frame, (int(fw * rs), PROCESS_MAX_H))

                # Track video position for accurate playback-rate FPS
                video_frames_advanced[stream_id] = int(
                    cap.get(cv2.CAP_PROP_POS_FRAMES)
                )

                run_frame_counts[stream_id] += 1
                current_frame = run_frame_counts[stream_id]
                do_infer = (current_frame % INFER_EVERY == 0)

                display = frame.copy()
                inference_elapsed = 0.0

                if do_infer:
                    # --- INFERENCE PIPELINE ---
                    t_infer = time.perf_counter()

                    detections = []
                    if cfg["enable_detection"]:
                        detections = detector.predict(frame)
                    last_detections[stream_id] = detections

                    if cfg["enable_tracking"]:
                        xyxy = [tuple(d["xyxy"]) for d in detections] if detections else []
                        confs = [float(d["confidence"]) for d in detections] if detections else []
                        cids = [int(d["class_id"]) for d in detections] if detections else []
                        tracked = tracker.update_stream(stream_id, frame, xyxy, confs, cids)
                        last_tracked[stream_id] = tracked
                        if cfg["enable_counting"]:
                            counters[stream_id].update(tracked)

                    inference_elapsed = time.perf_counter() - t_infer
                    run_latency_totals[stream_id] += inference_elapsed

                # --- DRAW using cached results (every frame) ---
                detections = last_detections[stream_id]
                tracked = last_tracked.get(stream_id)

                display = draw_detections(display, detections)

                if tracked is not None and len(tracked) > 0:
                    for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.putText(
                            display,
                            f"ID {int(track_id)}",
                            (x1, min(display.shape[0] - 5, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

                if cfg["enable_counting"]:
                    display = counters[stream_id].annotate(display)

                # --- RESIZE for display (reduces WebSocket payload dramatically) ---
                h, w = display.shape[:2]
                if w > DISPLAY_MAX_W:
                    scale = DISPLAY_MAX_W / w
                    display = cv2.resize(display, (DISPLAY_MAX_W, int(h * scale)))

                # --- CALCULATE DISPLAY METRICS ---
                run_elapsed = time.perf_counter() - run_start_time
                # FPS = video frames advanced (incl. skipped) / wall time
                # This reflects the effective playback rate the user sees.
                vid_pos = video_frames_advanced[stream_id]
                fps = vid_pos / run_elapsed if run_elapsed > 0 else 0.0
                infer_frames = max(run_frame_counts[stream_id] // INFER_EVERY, 1)
                latency = (run_latency_totals[stream_id] / infer_frames) * 1000.0
                live_count = len(detections)
                alert = live_count >= int(crowd_threshold)

                # --- RENDER VIDEO every frame, metrics every INFER_EVERY frames ---
                frame_slots[stream_id].image(
                    cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                )
                if do_infer:
                    metric_slots[stream_id]["fps"].metric("FPS", f"{fps:.1f}")
                    metric_slots[stream_id]["latency"].metric("Latency (ms)", f"{latency:.1f}")
                    metric_slots[stream_id]["count"].metric("People", f"{live_count}")
                    metric_slots[stream_id]["alert"].metric("Alert", "ON" if alert else "OFF")
                    if alert:
                        metric_slots[stream_id]["msg"].error(
                            f"Threshold exceeded: {live_count} people detected."
                        )
                    else:
                        metric_slots[stream_id]["msg"].empty()
                    metric_slots[stream_id]["roi"].caption(
                        f"ROI IN: {counters[stream_id].counts_in} | OUT: {counters[stream_id].counts_out}"
                    )

            if live_active == 0:
                st.warning("All streams ended or failed.")
                break

        # Compute average playback FPS and inference latency across all active streams
        run_elapsed = time.perf_counter() - run_start_time
        fps_values = [
            vid_pos / run_elapsed
            for vid_pos in video_frames_advanced.values()
            if run_elapsed > 0 and vid_pos > 0
        ]
        latency_values = [
            (run_latency_totals[sid] / run_frame_counts[sid]) * 1000.0
            for sid in run_frame_counts
            if run_frame_counts[sid] > 0
        ]
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
        avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
        import json
        metrics_file = Path("reports/performance_metrics.json")
        precision, recall, f1 = 0.0, 0.0, 0.0
        if metrics_file.exists():
            try:
                with metrics_file.open("r", encoding="utf-8") as f:
                    ext = json.load(f)
                    precision = ext.get("detection_precision", 0.0)
                    recall = ext.get("detection_recall", 0.0)
                    f1 = ext.get("f1_score", 0.0)
            except Exception:
                pass

        save_performance_metrics(
            detection_precision=precision,
            detection_recall=recall,
            f1_score=f1,
            average_fps_per_stream=avg_fps,
            average_latency_ms=avg_latency,
        )
        st.success("Run complete. Metrics updated in reports/performance_metrics.json.")
    except Exception as exc:
        logging.exception("Runtime multi-stream error: %s", exc)
        st.error("A runtime error occurred. Check logs/app_errors.log.")
    finally:
        for cap in captures.values():
            cap.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logging.exception("Unhandled exception in Streamlit app: %s", exc)
        raise

