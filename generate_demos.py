from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import cv2
import numpy as np

from models.yolo_person_detection import OnnxPersonDetector, YoloPersonDetector
from utils.counting_logic import RoiCounter
from utils.data_loader import iter_frames, save_video
from utils.multi_stream_tracking_helpers import MultiStreamTracker


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo PNG/MP4 for VisionTrack.")
    parser.add_argument(
        "--video",
        required=True,
        help="Input video path for demo generation.",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx"],
        default="onnx",
        help="Detection backend to use.",
    )
    parser.add_argument(
        "--weights",
        default="models/checkpoints/best.pt",
        help="PyTorch weights path (if backend=pytorch).",
    )
    parser.add_argument(
        "--onnx",
        default="models/checkpoints/best_quantized.onnx",
        help="ONNX model path (if backend=onnx).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum frames to include in demo video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=18.5,
        help="FPS to use for output demo video.",
    )
    return parser.parse_args()


def get_detector(args: argparse.Namespace):
    if args.backend == "pytorch":
        device = YoloPersonDetector.resolve_device()
        return YoloPersonDetector(
            model_path=args.weights,
            conf_threshold=0.35,
            iou_threshold=0.5,
            device=device,
        )
    return OnnxPersonDetector(
        onnx_path=args.onnx,
        conf_threshold=0.35,
        iou_threshold=0.5,
    )


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    detector = get_detector(args)
    tracker = MultiStreamTracker()
    stream_id = "demo_stream"
    
    # FIX: Define a 4-point polygon (Rectangle) for the demo video
    demo_polygon = np.array([
        [50, 150],   # Top-Left
        [600, 150],  # Top-Right
        [600, 350],  # Bottom-Right
        [50, 350]    # Bottom-Left
    ])
    counter = RoiCounter(roi_polygon=demo_polygon)

    frames_out: List[np.ndarray] = []
    first_roi_frame: np.ndarray | None = None

    for frame in iter_frames(str(video_path), max_frames=args.max_frames):
        detections = detector.predict(frame)
        display = draw_detections(frame, detections)

        if detections:
            xyxy = [tuple(d["xyxy"]) for d in detections]
            confs = [float(d["confidence"]) for d in detections]
            cids = [int(d["class_id"]) for d in detections]
            tracked = tracker.update_stream(stream_id, frame, xyxy, confs, cids)
            counter.update(tracked)

        # FIX: Use the new supervision annotator instead of manual cv2 drawing
        display = counter.annotate(display)

        if first_roi_frame is None:
            first_roi_frame = display.copy()

        frames_out.append(display)

    out_dir = Path("reports/demo_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if first_roi_frame is not None:
        png_path = out_dir / "roi_counting_example.png"
        cv2.imwrite(str(png_path), first_roi_frame)

    mp4_path = out_dir / "multi_stream_demo.mp4"
    save_video(frames_out, mp4_path, fps=args.fps)

    print(f"Saved ROI demo PNG to {png_path}")
    print(f"Saved demo video to {mp4_path}")


if __name__ == "__main__":
    main()

