"""Model package for VisionTrack — person detection with YOLO (PyTorch & ONNX)."""

from models.yolo_person_detection import OnnxPersonDetector, YoloPersonDetector

__all__ = ["YoloPersonDetector", "OnnxPersonDetector"]
