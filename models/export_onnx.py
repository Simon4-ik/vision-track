from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO checkpoint to ONNX and save quantized artifact name."
    )
    parser.add_argument(
        "--weights",
        default="models/checkpoints/best.pt",
        help="Path to trained YOLO checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="models/checkpoints/best_quantized.onnx",
        help="Path to exported ONNX file.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision during export when supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=args.imgsz, half=args.half)

    default_export = weights_path.with_suffix(".onnx")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if default_export.exists():
        out_path.write_bytes(default_export.read_bytes())
        if default_export.resolve() != out_path.resolve():
            default_export.unlink(missing_ok=True)
    else:
        raise FileNotFoundError(
            "ONNX export completed but output file was not found at expected path."
        )


if __name__ == "__main__":
    main()

