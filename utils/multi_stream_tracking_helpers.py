from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv


@dataclass
class StreamStats:
    """Per-stream performance statistics for tracking FPS and latency."""

    frame_count: int = 0
    total_time_s: float = 0.0  # Cumulative tracking-only time
    total_inference_s: float = 0.0  # Cumulative inference time (detection + tracking)
    start_time: float = field(default_factory=perf_counter)

    @property
    def fps(self) -> float:
        """Inference FPS: frames / total inference time (excludes UI overhead)."""
        if self.total_inference_s <= 0:
            return 0.0
        return self.frame_count / self.total_inference_s

    @property
    def wall_fps(self) -> float:
        """Wall-clock FPS including all overhead (Streamlit rendering, etc.)."""
        elapsed = perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.frame_count / elapsed

    @property
    def avg_latency_ms(self) -> float:
        """Average per-frame inference latency in milliseconds."""
        if self.frame_count == 0:
            return 0.0
        return (self.total_inference_s / self.frame_count) * 1000.0

    def record_inference_time(self, seconds: float) -> None:
        """Record the total inference time for one frame (detection + tracking)."""
        self.total_inference_s += seconds


@dataclass
class MultiStreamTracker:
    """Manage supervision trackers for multiple streams."""

    trackers: Dict[str, sv.ByteTrack] = field(default_factory=dict)
    stats: Dict[str, StreamStats] = field(default_factory=dict)

    def _get_tracker(self, stream_id: str) -> sv.ByteTrack:
        if stream_id not in self.trackers:
            self.trackers[stream_id] = sv.ByteTrack()
        if stream_id not in self.stats:
            self.stats[stream_id] = StreamStats()
        return self.trackers[stream_id]

    def update_stream(
    self, stream_id: str, frame: np.ndarray, detections_xyxy: List[Tuple[float, float, float, float]], confidences: List[float], class_ids: List[int],
    ) -> sv.Detections:
        start = perf_counter()
        tracker = self._get_tracker(stream_id)

        # FIX: Update ByteTrack with empty detections so it can age-out dead tracks
        if not detections_xyxy:
            tracked = tracker.update_with_detections(sv.Detections.empty())
            elapsed = perf_counter() - start
            st = self.stats[stream_id]
            st.frame_count += 1
            st.total_time_s += elapsed
            return tracked

        boxes = np.array(detections_xyxy, dtype=float)
        confs = np.array(confidences, dtype=float)
        classes = np.array(class_ids, dtype=int)

        det = sv.Detections(
            xyxy=boxes,
            confidence=confs,
            class_id=classes,
        )

        tracker = self._get_tracker(stream_id)
        tracked = tracker.update_with_detections(det)

        elapsed = perf_counter() - start
        st = self.stats[stream_id]
        st.frame_count += 1
        st.total_time_s += elapsed

        return tracked

