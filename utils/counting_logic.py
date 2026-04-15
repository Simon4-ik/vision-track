"""ROI-based entry/exit counting logic using the supervision library.

Tracks individuals entering and exiting a user-defined polygon zone.
Uses a time-based grace-period mechanism to avoid premature OUT counts
caused by momentary tracking flickers, regardless of actual FPS.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import supervision as sv


@dataclass
class RoiCounter:
    """ROI-based entry/exit counter using supervision PolygonZone.

    A person is counted IN when their center first enters the polygon.
    A person is counted OUT only after being absent for `grace_period_s`
    seconds, preventing false exits from tracker flicker and remaining
    accurate regardless of the actual processing FPS.
    """

    roi_polygon: np.ndarray  # Accepts a 4-point rectangle array
    classes_to_count: Tuple[int, ...] = (0,)
    grace_period_s: float = 2.0  # seconds before an absent track fires OUT
    counts_in: int = 0
    counts_out: int = 0

    _zone: sv.PolygonZone = field(init=False)
    _annotator: sv.PolygonZoneAnnotator = field(init=False)
    # Maps track_id -> timestamp when it was last seen inside the zone
    _tracked_inside: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._zone = sv.PolygonZone(
            polygon=self.roi_polygon,
            triggering_anchors=(sv.Position.CENTER,)
        )
        self._annotator = sv.PolygonZoneAnnotator(
            zone=self._zone, thickness=2, text_thickness=1, text_scale=0.5
        )

    def update(self, detections: sv.Detections) -> None:
        now = time.monotonic()
        current_inside_ids = set()

        if len(detections) > 0 and detections.tracker_id is not None:
            mask = np.isin(detections.class_id, self.classes_to_count)
            filtered = detections[mask]

            if len(filtered) > 0 and filtered.tracker_id is not None:
                is_inside = self._zone.trigger(detections=filtered)
                for i, inside in enumerate(is_inside):
                    if inside:
                        current_inside_ids.add(filtered.tracker_id[i])

        # 1. Process INs and refresh last-seen timestamp for people in the zone
        for track_id in current_inside_ids:
            if track_id not in self._tracked_inside:
                self.counts_in += 1  # Brand new person walked in
            self._tracked_inside[track_id] = now  # Refresh last-seen time

        # 2. Process OUTs for people whose absence exceeds the grace period
        dead_ids = []
        for track_id, last_seen in self._tracked_inside.items():
            if track_id not in current_inside_ids:
                if (now - last_seen) >= self.grace_period_s:
                    self.counts_out += 1
                    dead_ids.append(track_id)

        # 3. Clean up confirmed exits
        for dead_id in dead_ids:
            del self._tracked_inside[dead_id]

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        return self._annotator.annotate(scene=frame.copy())


def save_performance_metrics(
    detection_precision: float,
    detection_recall: float,
    f1_score: float,
    average_fps_per_stream: float,
    average_latency_ms: float,
    output_path: str | Path = "reports/performance_metrics.json",
) -> None:
    """Persist metrics in the required JSON schema."""
    payload = {
        "detection_precision": float(detection_precision),
        "detection_recall": float(detection_recall),
        "f1_score": float(f1_score),
        "average_fps_per_stream": float(average_fps_per_stream),
        "average_latency_ms": float(average_latency_ms),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)