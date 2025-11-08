"""Utilities for running YOLO-based object detection."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Thin wrapper around the Ultralytics YOLO models for dashcam analysis."""

    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        class_filter: Optional[List[str]] = None,
    ) -> None:
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = device or self._resolve_device()
        self.model_path = model_path or f"yolov8{self.model_size}.pt"
        self.model_name = Path(self.model_path).stem
        self.class_filter = {c.lower() for c in class_filter} if class_filter else None
        self.model = None
        self.class_names: Dict[int, str] = {}

        self._load_model()

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # noqa: WPS433 (import inside try for optional dependency)

            logger.info("Loading %s on %s", self.model_path, self.device)
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.class_names = self.model.names or {}
            logger.info("Model ready with %d classes", len(self.class_names))
        except Exception as exc:  # noqa: BLE001 (propagate useful error message)
            logger.error("Failed to load YOLO model: %s", exc)
            self.model = None

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], Dict[str, float]]:
        """Run a single-frame detection pass."""
        if image is None or image.size == 0:
            logger.warning("Detector received an empty frame")
            return [], {"num_raw_detections": 0, "inference_time": 0.0}

        if self.model is None:
            logger.error("YOLO model not loaded; attempting reload")
            self._load_model()
            if self.model is None:
                return [], {"num_raw_detections": 0, "inference_time": 0.0}

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time.perf_counter()

        try:
            results = self.model.predict(
                rgb_frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("YOLO inference failed: %s", exc)
            return [], {"num_raw_detections": 0, "inference_time": 0.0}

        inference_time = time.perf_counter() - start_time
        detections: List[Dict] = []

        if not results:
            metadata = {
                "num_raw_detections": 0,
                "num_filtered_detections": 0,
                "inference_time": inference_time,
                "model": self.model_name,
                "device": self.device,
            }
            return detections, metadata

        result = results[0]
        raw_count = int(result.boxes.shape[0]) if result.boxes is not None else 0

        for box in result.boxes:
            confidence = float(box.conf.item())
            cls_id = int(box.cls.item())
            class_name = self.class_names.get(cls_id, str(cls_id))

            if self.class_filter and class_name.lower() not in self.class_filter:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            detections.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "box_center": [float(center_x), float(center_y)],
                    "width": float(width),
                    "height": float(height),
                    "area": float(abs(width * height)),
                    "confidence": confidence,
                    "class_id": cls_id,
                    "class_name": class_name,
                }
            )

        metadata = {
            "num_raw_detections": raw_count,
            "num_filtered_detections": len(detections),
            "inference_time": inference_time,
            "model": self.model_name,
            "device": self.device,
        }

        return detections, metadata

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        batches: List[List[Dict]] = []
        for image in images:
            detections, _ = self.detect(image)
            batches.append(detections)
        return batches

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_scores: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det.get("box", [0, 0, 0, 0])]
            class_name = det.get("class_name", "object")
            confidence = det.get("confidence", 0.0)

            color_seed = sum(ord(char) for char in class_name)
            color = (
                (color_seed * 13) % 255,
                (color_seed * 37) % 255,
                (color_seed * 53) % 255,
            )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_parts: List[str] = []
            if show_labels:
                label_parts.append(class_name)
            if show_scores:
                label_parts.append(f"{confidence:.2f}")

            if label_parts:
                label = " ".join(label_parts)
                (label_width, label_height), _ = cv2.getTextSize(
                    label,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                )

                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - label_height - 6)),
                    (x1 + label_width + 4, y1),
                    color,
                    -1,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return annotated