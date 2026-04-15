from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    precision: float = 0.85
    recall: float = 0.80
    f1: float = 0.85
    avg_fps: float = 15.0


REQUIRED_FILES = [
    Path("models/checkpoints/best.pt"),
    Path("models/checkpoints/best_quantized.onnx"),
    Path("models/checkpoints/config.yaml"),
    Path("reports/performance_metrics.json"),
    Path("logs"),
]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    print("VisionTrack Validation Report")
    print("=" * 32)

    missing = [p for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print("\nRequired artifacts:")
        for p in REQUIRED_FILES:
            print(f"- {p}: {fmt_bool(p.exists())}")
        print("\nMissing required files/directories:")
        for p in missing:
            print(f"- {p}")
        return 2

    # CUDA check (informational only)
    cuda_available = None
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = None

    metrics_path = Path("reports/performance_metrics.json")
    metrics = load_metrics(metrics_path)

    precision = float(metrics.get("detection_precision", 0.0))
    recall = float(metrics.get("detection_recall", 0.0))
    f1 = float(metrics.get("f1_score", 0.0))
    avg_fps = float(metrics.get("average_fps_per_stream", 0.0))
    avg_latency = float(metrics.get("average_latency_ms", 0.0))

    t = Thresholds()
    checks = {
        "detection_precision": precision >= t.precision,
        "detection_recall": recall >= t.recall,
        "f1_score": f1 >= t.f1,
        "average_fps_per_stream": avg_fps >= t.avg_fps,
    }

    print("\nEnvironment:")
    if cuda_available is None:
        print("- CUDA available: unknown (torch not installed or import failed)")
    else:
        print(f"- CUDA available: {cuda_available}")

    print("\nArtifacts:")
    for p in REQUIRED_FILES:
        print(f"- {p}: {fmt_bool(p.exists())}")

    print("\nMetrics (from reports/performance_metrics.json):")
    print(f"- detection_precision: {precision:.4f} (min {t.precision:.2f}) -> {fmt_bool(checks['detection_precision'])}")
    print(f"- detection_recall: {recall:.4f} (min {t.recall:.2f}) -> {fmt_bool(checks['detection_recall'])}")
    print(f"- f1_score: {f1:.4f} (min {t.f1:.2f}) -> {fmt_bool(checks['f1_score'])}")
    print(f"- average_fps_per_stream: {avg_fps:.2f} (min {t.avg_fps:.1f}) -> {fmt_bool(checks['average_fps_per_stream'])}")
    print(f"- average_latency_ms: {avg_latency:.2f} (informational)")

    ok = all(checks.values())
    print("\nOverall:")
    print(f"- {fmt_bool(ok)}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

