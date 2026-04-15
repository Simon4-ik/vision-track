from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for person detection.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model weights.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (>=10).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument(
        "--project",
        default="models/checkpoints",
        help="Output project directory for checkpoints.",
    )
    parser.add_argument("--name", default="visiontrack_person", help="Run name.")
    return parser.parse_args()


def write_training_config(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    payload = {
        "task": "person_detection_transfer_learning",
        "base_model": args.model,
        "dataset_yaml": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "patience": args.patience,
    }
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_training_log(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.epochs < 10:
        raise ValueError("Use at least 10 epochs to satisfy transfer-learning guidelines.")

    model = YOLO(args.model)
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )

    run_dir = Path(args.project) / args.name / "weights"
    best_weight = run_dir / "best.pt"
    final_best = Path(args.project) / "best.pt"
    final_best.parent.mkdir(parents=True, exist_ok=True)
    if best_weight.exists():
        final_best.write_bytes(best_weight.read_bytes())

    write_training_config(args, Path(args.project))
    write_training_log(
        metrics={
            "results_dir": str(train_results.save_dir),
            "best_weight": str(final_best),
        },
        output_dir=Path(args.project),
    )


if __name__ == "__main__":
    main()

