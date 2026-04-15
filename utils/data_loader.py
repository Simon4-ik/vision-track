"""Video I/O utilities for loading, iterating, and saving video streams.

Supports both file paths and camera indices as video sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable

import cv2
import numpy as np


def open_video_stream(source: str | int) -> cv2.VideoCapture:
    """Open a video stream from a file path or camera index."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def iter_frames(
    source: str | int,
    max_frames: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield frames from a video source.

    Used for offline evaluation / demo video generation.
    """
    cap = open_video_stream(source)
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def save_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: float,
) -> None:
    """Save a sequence of frames to a video file."""
    frames_iter = iter(frames)
    try:
        first = next(frames_iter)
    except StopIteration:
        return

    h, w = first.shape[:2]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        writer.write(first)
        for f in frames_iter:
            writer.write(f)
    finally:
        writer.release()

