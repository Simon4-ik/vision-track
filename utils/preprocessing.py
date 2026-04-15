"""Image preprocessing utilities for YOLO-compatible input preparation.

Provides resizing with aspect-ratio preservation (letterbox) and
BGR-to-RGB conversion for model input normalization.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_with_aspect(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Resize frame keeping aspect ratio by padding."""
    h, w = frame.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((th, tw, 3), dtype=frame.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

