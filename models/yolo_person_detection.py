"""Person detection module using YOLO (PyTorch) and ONNX Runtime backends.

Provides two detector classes:
- YoloPersonDetector: Uses Ultralytics YOLO with PyTorch for inference.
- OnnxPersonDetector: Uses exported ONNX model with ONNX Runtime for optimized inference.

Both filter detections to the COCO 'person' class (class_id=0) and return
a list of dicts with keys: xyxy, confidence, class_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class YoloPersonDetector:
    """YOLO-based person detector using Ultralytics PyTorch backend.

    Loads a pre-trained or fine-tuned YOLO checkpoint and runs inference
    filtered to the person class only (COCO class 0).
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(model_path)

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        # OPTIMIZATION: Tell PyTorch to ONLY compute NMS for the Person class
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=[self.PERSON_CLASS_ID],  # <--- MASSIVE SPEEDUP HERE
            verbose=False,
        )
        
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        # Vectorized extraction (bypassing the slow Python if-statements)
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        return [
            {
                "xyxy": box.tolist(),
                "confidence": float(conf),
                "class_id": self.PERSON_CLASS_ID,
            }
            for box, conf in zip(xyxy, confs)
        ]

    @staticmethod
    def resolve_device() -> str:
        """Prefer CUDA or Apple Silicon (MPS), else CPU."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon hardware acceleration
                
            return "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def checkpoint_exists(checkpoint_path: str | Path) -> bool:
        return Path(checkpoint_path).exists()


class OnnxPersonDetector:
    """ONNX Runtime person detector using exported YOLO model.

    Provides a quantization-friendly inference path with automatic
    hardware provider selection (CUDA > CoreML > CPU).
    Preprocessing includes letterbox resize to 640x640 with BGR-to-RGB conversion.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        onnx_path: str = "models/checkpoints/best_quantized.onnx",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> None:
        import os
        import onnxruntime as ort  # lazy import

        self.onnx_path = onnx_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        # Auto-detect available CPU cores for optimal thread utilization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = os.cpu_count() or 4
        sess_options.intra_op_num_threads = max(1, cpu_count)

        # Only request providers that are actually available on this system
        available = set(ort.get_available_providers())
        preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers: List[str] = [p for p in preferred if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        # Standard YOLO letterbox-style resize matching the ONNX export resolution
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        size = self.session.get_inputs()[0].shape[2]  # Auto-detect from model
        scale = min(size / w0, size / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        img_resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top : top + nh, left : left + nw] = img_resized
        img_input = canvas.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[None, ...]  # NCHW
        return img_input, scale, left, top, w0, h0

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        img_input, scale, left, top, orig_w, orig_h = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: img_input})

        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]
            
        preds = preds.transpose()

        # --- FAST NUMPY VECTORIZATION ---
        # Extract geometry and confidence for the Person class instantly
        boxes_data = preds[:, :4]
        person_conf = preds[:, 4 + self.PERSON_CLASS_ID]
        
        # Create a boolean mask of only high-confidence detections
        mask = person_conf >= self.conf_threshold
        filtered_boxes = boxes_data[mask]
        filtered_conf = person_conf[mask]

        detections: list[dict[str, Any]] = []
        if len(filtered_boxes) == 0:
            return detections

        # Convert [center_x, center_y, width, height] to OpenCV format [x_min, y_min, w, h]
        x_min = filtered_boxes[:, 0] - (filtered_boxes[:, 2] / 2)
        y_min = filtered_boxes[:, 1] - (filtered_boxes[:, 3] / 2)
        
        bboxes_nms = np.stack([x_min, y_min, filtered_boxes[:, 2], filtered_boxes[:, 3]], axis=-1).tolist()
        scores_nms = filtered_conf.tolist()

        indices = cv2.dnn.NMSBoxes(
            bboxes=bboxes_nms, 
            scores=scores_nms, 
            score_threshold=self.conf_threshold, 
            nms_threshold=self.iou_threshold
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                x1 = x_min[i]
                y1 = y_min[i]
                w = filtered_boxes[i, 2]
                h = filtered_boxes[i, 3]
                conf = scores_nms[i]
                
                x2 = x1 + w
                y2 = y1 + h

                # Map back to original image
                x1 = float(np.clip((x1 - left) / scale, 0, orig_w - 1))
                y1 = float(np.clip((y1 - top) / scale, 0, orig_h - 1))
                x2 = float(np.clip((x2 - left) / scale, 0, orig_w - 1))
                y2 = float(np.clip((y2 - top) / scale, 0, orig_h - 1))

                detections.append(
                    {
                        "xyxy": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": self.PERSON_CLASS_ID,
                    }
                )
                
        return detections