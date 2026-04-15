from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2

from models.yolo_person_detection import YoloPersonDetector
from utils.counting_logic import save_performance_metrics


Box = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO person detection.")
    parser.add_argument(
        "--weights",
        default="models/checkpoints/best.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing validation images.",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing YOLO-format labels matching image names.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP matching.",
    )
    return parser.parse_args()


def yolo_to_xyxy(
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[int, Box]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return class_id, (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def read_gt_person_boxes(label_path: Path, img_w: int, img_h: int) -> List[Box]:
    if not label_path.exists():
        return []
    gt: List[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != 0:
            continue
        cx, cy, w, h = map(float, parts[1:])
        _, box = yolo_to_xyxy(class_id, cx, cy, w, h, img_w, img_h)
        gt.append(box)
    return gt


def match_predictions(pred_boxes: List[Box], gt_boxes: List[Box], match_iou: float) -> tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx >= 0 and best_iou >= match_iou:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = max(0, len(gt_boxes) - len(matched_gt))
    return tp, fp, fn


def main() -> None:
    args = parse_args()
    device = YoloPersonDetector.resolve_device()
    detector = YoloPersonDetector(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=device,
    )

    image_dir = Path(args.images_dir)
    label_dir = Path(args.labels_dir)
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        gt_boxes = read_gt_person_boxes(label_path, w, h)
        preds = detector.predict(img)
        pred_boxes = [tuple(d["xyxy"]) for d in preds]

        tp, fp, fn = match_predictions(pred_boxes, gt_boxes, args.match_iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    save_performance_metrics(
        detection_precision=precision,
        detection_recall=recall,
        f1_score=f1,
        average_fps_per_stream=0.0,
        average_latency_ms=0.0,
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Updated reports/performance_metrics.json")


if __name__ == "__main__":
    main()

