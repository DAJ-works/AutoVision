"""Utilities for running YOLOP-based driving perception with YOLOv8 hybrid detection."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore[import-not-found]
except ImportError as import_exc:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "OpenCV (cv2) is not installed; YOLOP detector will fall back to numpy/PIL operations"
    )

try:
    from PIL import Image  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Pillow is part of requirements but guard anyway
    Image = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:
    YOLO = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "Ultralytics (YOLOv8) is not installed; hybrid detection will be disabled"
    )

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Wrapper around YOLOP + YOLOv8 hybrid detection for dashcam analysis."""

    DEFAULT_CLASS_NAMES: Dict[int, str] = {
        0: "vehicle",
        1: "pedestrian",
        2: "cyclist",
    }

    def __init__(
        self,
        model_size: str = "640",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        class_filter: Optional[List[str]] = None,
        use_hybrid: bool = True,
    ) -> None:
        self.input_size = int(model_size) if str(model_size).isdigit() else 640
        self.confidence_threshold = confidence_threshold
        self.device = device or self._resolve_device()
        self.model_path = model_path
        self.model_name = "yolop"
        self.class_filter = {c.lower() for c in class_filter} if class_filter else None
        self.model: Optional[torch.nn.Module] = None
        self.yolov8_model: Optional[Any] = None
        self.use_hybrid = use_hybrid and YOLO is not None
        self.class_names: Dict[int, str] = {}

        self._load_model()
        if self.use_hybrid:
            self._load_yolov8()

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        try:
            logger.info("Loading YOLOP model on %s", self.device)
            # The Torch Hub call downloads weights if missing.
            # Note: YOLOP entry point doesn't accept map_location parameter
            hub_kwargs: Dict[str, Any] = {
                "pretrained": True,
                "trust_repo": True,
            }
            self.model = torch.hub.load("hustvl/yolop", "yolop", **hub_kwargs)
            self.model = self.model.to(self.device)
            self.model.eval()

            # YOLOP models expose a class names mapping in some builds; fall back if absent.
            names: Any = getattr(self.model, "class_names", None) or getattr(
                self.model, "names", None
            )
            if isinstance(names, dict) and names:
                self.class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)) and names:
                self.class_names = {idx: str(name) for idx, name in enumerate(names)}
            else:
                # YOLOP typically only detects vehicles (class 0)
                self.class_names = {0: "car"}
                logger.warning("YOLOP model doesn't expose class names, using default: {0: 'car'}")

            # Fix YOLOP class names if they're just numeric strings (e.g., {0: '0'})
            # YOLOP detects vehicles, so map class 0 to "car"
            for class_id, class_name in list(self.class_names.items()):
                if class_name.isdigit():
                    self.class_names[class_id] = "car"
                    logger.info(f"Mapped YOLOP numeric class '{class_name}' to 'car'")

            logger.info("YOLOP ready with %d detection classes: %s", len(self.class_names), self.class_names)
        except Exception as exc:  # noqa: BLE001 - propagate meaningful diagnostics
            logger.error("Failed to load YOLOP model: %s", exc)
            self.model = None

    def _load_yolov8(self) -> None:
        """Load YOLOv8 model for hybrid detection (better close-range detection)."""
        try:
            logger.info("Loading YOLOv8 model for hybrid detection on %s", self.device)
            # Use YOLOv8n (nano) for speed, or YOLOv8m for better accuracy
            self.yolov8_model = YOLO('yolov8n.pt')
            logger.info("YOLOv8 loaded successfully for hybrid detection")
        except Exception as exc:
            logger.error("Failed to load YOLOv8 model: %s", exc)
            self.yolov8_model = None
            self.use_hybrid = False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def detect(
        self, image: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Run a single frame through YOLOP + YOLOv8 hybrid detection.

        Returns detections, numeric metadata, and auxiliary outputs (segmentation masks).
        """

        if image is None or image.size == 0:
            logger.warning("Detector received an empty frame")
            return [], self._base_metadata(0.0, note="empty-frame"), {}

        if not self._ensure_model_loaded():
            return [], self._base_metadata(0.0, note="model-unavailable"), {}

        original_height, original_width = image.shape[:2]
        
        # Run YOLOP detection (good for medium/far vehicles)
        yolop_detections, metadata, aux = self._detect_yolop(image)
        
        # Run YOLOv8 detection if hybrid mode enabled (good for close/large vehicles)
        if self.use_hybrid and self.yolov8_model is not None:
            yolov8_detections = self._detect_yolov8(image, original_height, original_width)
            # Merge detections, preferring YOLOv8 for large boxes and YOLOP for small/medium
            detections = self._merge_detections(yolop_detections, yolov8_detections, original_width, original_height)
            metadata['detection_sources'] = f"yolop={len(yolop_detections)}, yolov8={len(yolov8_detections)}, merged={len(detections)}"
        else:
            detections = yolop_detections
            metadata['detection_sources'] = f"yolop_only={len(detections)}"
        
        return detections, metadata, aux

    def _detect_yolop(
        self, image: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Run YOLOP detection on the image."""
        original_height, original_width = image.shape[:2]
        letterboxed, ratios, padding = self._letterbox(image, self.input_size)
        rgb_lb = self._to_rgb(letterboxed)
        base_tensor = torch.from_numpy(rgb_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        det_out = da_seg_out = ll_seg_out = None
        inference_time = 0.0
        last_error: Optional[Exception] = None

        inference_devices = self._inference_devices()

        for attempt, target_device in enumerate(inference_devices):
            try:
                tensor = base_tensor.to(target_device, non_blocking=False)
                if target_device != self.device:
                    logger.warning(
                        "Switching YOLOP inference device from %s to %s",
                        self.device,
                        target_device,
                    )
                    self.device = target_device
                    if self.model is not None:
                        self.model = self.model.to(self.device)

                start_time = time.perf_counter()
                with torch.inference_mode():
                    det_out, da_seg_out, ll_seg_out = self.model(tensor)
                inference_time = time.perf_counter() - start_time
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log_fn = logger.error if attempt == len(inference_devices) - 1 else logger.warning
                log_fn(
                    "YOLOP inference attempt %d failed on %s: %s",
                    attempt + 1,
                    target_device,
                    exc,
                )
                continue

        if last_error is not None:
            metadata = self._base_metadata(0.0, note="inference-error")
            metadata["error"] = str(last_error)
            return [], metadata, {}

        detections = self._parse_detections(
            det_out,
            ratios,
            padding,
            (original_height, original_width),
        )

        drivable_mask = self._parse_segmentation(da_seg_out, (original_height, original_width))
        lane_mask = self._parse_segmentation(ll_seg_out, (original_height, original_width))

        metadata = self._base_metadata(inference_time)
        metadata.update(
            {
                "num_raw_detections": len(detections),
                "num_filtered_detections": len(detections),
                "drivable_area_ratio": float(drivable_mask.mean()) if drivable_mask is not None else 0.0,
                "lane_pixel_coverage": float(lane_mask.mean()) if lane_mask is not None else 0.0,
            }
        )

        auxiliary = {
            "drivable_area_mask": drivable_mask,
            "lane_line_mask": lane_mask,
            "scale_ratios": ratios,
            "padding": padding,
        }

        return detections, metadata, auxiliary

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        batches: List[List[Dict[str, Any]]] = []
        for image in images:
            detections, _, _ = self.detect(image)
            batches.append(detections)
        return batches

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_scores: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for visualization utilities")

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _letterbox(
        self,
        image: np.ndarray,
        new_size: int,
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """Resize image to fit within `new_size` while preserving aspect ratio."""
        height, width = image.shape[:2]
        scale = min(new_size / height, new_size / width)
        new_height, new_width = int(round(height * scale)), int(round(width * scale))

        if cv2 is not None:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = self._resize_with_pillow(image, (new_width, new_height))

        pad_w = new_size - new_width
        pad_h = new_size - new_height
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if cv2 is not None:
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        else:
            padded = np.full((new_size, new_size, resized.shape[2]), color, dtype=resized.dtype)
            padded[top : top + new_height, left : left + new_width] = resized

        ratio = (scale, scale)
        pad = (left, top)
        return padded, ratio, pad

    def _parse_detections(
        self,
        det_out: Any,
        ratios: Tuple[float, float],
        padding: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Convert raw YOLOP detections to normalized dictionaries."""

        detections: List[Dict[str, Any]] = []

        if det_out is None:
            return detections

        # YOLOP returns a list where first element contains detections
        if isinstance(det_out, (list, tuple)):
            # YOLOP format: [detections_list, ...] where detections might be nested
            pred = det_out[0]
        else:
            pred = det_out

        # Handle nested list structure from YOLOP
        if isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]  # Get first batch

        if isinstance(pred, np.ndarray):
            preds = torch.from_numpy(pred)
        elif isinstance(pred, torch.Tensor):
            preds = pred
        else:
            logger.debug("Unexpected detection output type: %s", type(pred))
            return detections

        preds = preds.detach().cpu()
        
        # Handle different tensor shapes
        if preds.ndim == 3:
            # Batch dimension present: (1, N, 6)
            preds = preds[0]
        elif preds.ndim == 1:
            preds = preds.unsqueeze(0)

        if preds.numel() == 0 or preds.shape[0] == 0:
            return detections

        # Debug: log the actual shape
        logger.debug("YOLOP predictions shape: %s", preds.shape)
        
        # YOLOP/YOLOv5 format: each detection is [x1, y1, x2, y2, confidence, class_id]
        # But check we have the right dimensions
        if preds.ndim != 2 or preds.shape[1] < 6:
            logger.warning("Unexpected YOLOP output shape: %s (expected Nx6 or more)", preds.shape)
            return detections
        
        # YOLOP outputs raw predictions (25200 grid cells), need to apply NMS
        # First filter by confidence to reduce computation
        conf_mask = preds[:, 4] > self.confidence_threshold
        preds = preds[conf_mask]
        
        if preds.numel() == 0 or preds.shape[0] == 0:
            return detections
        
        # Log BEFORE conversion
        if preds.shape[0] > 0:
            logger.debug("BEFORE xywh→xyxy: [%.2f, %.2f, %.2f, %.2f]", 
                        preds[0, 0].item(), preds[0, 1].item(), preds[0, 2].item(), preds[0, 3].item())
        
        # YOLOP outputs boxes in [x_center, y_center, width, height] format
        # Convert to [x1, y1, x2, y2] format for NMS
        boxes_xywh = preds[:, :4].clone()
        boxes_xyxy = boxes_xywh.clone()
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2 = cy + h/2
        
        # Log AFTER conversion
        if boxes_xyxy.shape[0] > 0:
            logger.debug("AFTER xywh→xyxy: [%.2f, %.2f, %.2f, %.2f]", 
                        boxes_xyxy[0, 0].item(), boxes_xyxy[0, 1].item(), boxes_xyxy[0, 2].item(), boxes_xyxy[0, 3].item())
            
        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        boxes = boxes_xyxy
        scores = preds[:, 4].clone()
        class_ids = preds[:, 5].clone().long()
        
        # Apply NMS with IoU threshold of 0.45 (standard for YOLO)
        try:
            import torchvision
            keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            class_ids = class_ids[keep_indices]
            logger.debug(f"NMS reduced {conf_mask.sum()} detections to {len(keep_indices)}")
        except Exception as e:
            logger.warning(f"NMS failed: {e}, proceeding without NMS")

        # Log first detection for debugging
        if len(boxes) > 0:
            logger.debug("Raw YOLOP box sample (before scaling): %s, score: %.3f, class: %d", 
                        boxes[0].tolist(), scores[0].item(), class_ids[0].item())
            logger.debug("Ratios: %s, Padding: %s, Original shape: %s", ratios, padding, original_shape)

        boxes = self._scale_coords(boxes, ratios, padding, original_shape)
        
        # Log after scaling
        if len(boxes) > 0:
            logger.debug("Scaled YOLOP box sample: %s", boxes[0].tolist())

        for idx in range(boxes.shape[0]):
            confidence = float(scores[idx].item())
                
            if confidence < self.confidence_threshold:
                continue

            class_id = int(class_ids[idx].item())
            class_name = self.class_names.get(class_id, str(class_id))

            if self.class_filter and class_name.lower() not in self.class_filter:
                continue

            x1, y1, x2, y2 = boxes[idx].tolist()
            width = x2 - x1
            height = y2 - y1
            
            # Skip invalid boxes (zero or negative width/height from edge clamping)
            if width <= 1.0 or height <= 1.0:
                continue
            
            # Filter out detections in the upper 25% of frame (very top - likely sky/false positives)
            # Relaxed from 40% to allow horizon vehicles
            frame_height = original_shape[0]  # Use original frame height
            if y1 < frame_height * 0.25:
                continue
            
            # Filter out unreasonably small detections (noise) - relaxed threshold
            min_box_size = 10  # pixels (reduced from 20)
            if width < min_box_size or height < min_box_size:
                continue
            
            # Filter out unreasonably large detections (usually errors)
            # Relaxed to 70% to allow close vehicles
            if width > original_shape[1] * 0.7 or height > frame_height * 0.7:
                continue
            
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            detections.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "box_center": [float(center_x), float(center_y)],
                    "width": float(width),
                    "height": float(height),
                    "area": float(abs(width * height)),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

        return detections

    def _parse_segmentation(
        self,
        segmentation_out: Any,
        original_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if segmentation_out is None:
            return None

        if isinstance(segmentation_out, (list, tuple)):
            seg_tensor = segmentation_out[0]
        else:
            seg_tensor = segmentation_out

        if not isinstance(seg_tensor, torch.Tensor):
            logger.debug("Unexpected segmentation output type: %s", type(seg_tensor))
            return None

        seg_tensor = seg_tensor.detach().float()
        if seg_tensor.ndim == 4:
            seg_tensor = seg_tensor[0]

        if seg_tensor.ndim == 3:
            # Multi-class mask (C, H, W). Take the most likely class excluding background.
            if seg_tensor.shape[0] > 1:
                probabilities = F.softmax(seg_tensor, dim=0)
                mask = probabilities[1]  # class 1 is foreground in YOLOP heads
            else:
                mask = torch.sigmoid(seg_tensor[0])
        else:
            mask = torch.sigmoid(seg_tensor)

        mask_np = mask.cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        if cv2 is not None:
            resized = cv2.resize(
                binary_mask,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            resized = self._resize_mask_with_pillow(
                binary_mask,
                (original_shape[1], original_shape[0]),
            )

        return resized

    def _scale_coords(
        self,
        boxes: torch.Tensor,
        ratios: Tuple[float, float],
        padding: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Project letterboxed coordinates back onto the original frame."""

        gain_x, gain_y = ratios
        pad_x, pad_y = padding

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y

        boxes[:, [0, 2]] /= gain_x if gain_x else 1.0
        boxes[:, [1, 3]] /= gain_y if gain_y else 1.0

        boxes[:, 0].clamp_(0, original_shape[1])
        boxes[:, 2].clamp_(0, original_shape[1])
        boxes[:, 1].clamp_(0, original_shape[0])
        boxes[:, 3].clamp_(0, original_shape[0])

        return boxes

    def _ensure_model_loaded(self) -> bool:
        if self.model is None:
            logger.error("YOLOP model unavailable; attempting reload")
            self._load_model()
        return self.model is not None

    def _base_metadata(self, inference_time: float, note: Optional[str] = None) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "num_raw_detections": 0,
            "num_filtered_detections": 0,
            "inference_time": inference_time,
            "model": self.model_name,
            "device": self.device,
        }
        if note:
            metadata["note"] = note
        return metadata

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image[..., ::-1]

    def _resize_with_pillow(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if Image is None:
            raise RuntimeError("Pillow is required when OpenCV is unavailable")
        pil_image = Image.fromarray(image)
        resized = pil_image.resize(size, Image.BILINEAR)
        return np.asarray(resized)

    def _resize_mask_with_pillow(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if Image is None:
            raise RuntimeError("Pillow is required when OpenCV is unavailable")
        pil_image = Image.fromarray(mask.astype(np.uint8) * 255)
        resized = pil_image.resize(size, Image.NEAREST)
        return (np.asarray(resized) > 127).astype(np.uint8)

    def _inference_devices(self) -> List[str]:
        if self.device == "cpu":
            return ["cpu"]
        devices = [self.device]
        if self.device != "cpu":
            devices.append("cpu")
        return devices

    def _detect_yolov8(
        self, image: np.ndarray, original_height: int, original_width: int
    ) -> List[Dict[str, Any]]:
        """Run YOLOv8 detection on the image (better for close/large vehicles)."""
        try:
            # YOLOv8 vehicle classes: car=2, motorcycle=3, bus=5, truck=7
            vehicle_classes = [2, 3, 5, 7]
            
            # Run inference with lower confidence for better detection coverage
            results = self.yolov8_model.predict(
                image, 
                conf=0.10,  # Lower confidence threshold (was 0.15)
                classes=vehicle_classes,
                verbose=False,
                device=self.device
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                logger.info(f"YOLOv8 raw results: {len(boxes)} boxes detected before filtering")
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Map YOLOv8 class IDs to our class names
                    class_name_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                    class_name = class_name_map.get(class_id, 'vehicle')
                    
                    # Calculate box size as percentage of frame
                    width = x2 - x1
                    height = y2 - y1
                    box_area_ratio = (width * height) / (original_width * original_height)
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "class_id": 0,  # Use 0 for compatibility
                        "class_name": class_name,
                        "source": "yolov8",
                        "box_area_ratio": box_area_ratio,
                    })
            
            logger.info(f"YOLOv8 returning {len(detections)} detections")
            return detections
            
        except Exception as exc:
            logger.error("YOLOv8 detection failed: %s", exc)
            return []

    def _merge_detections(
        self,
        yolop_detections: List[Dict[str, Any]],
        yolov8_detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> List[Dict[str, Any]]:
        """Merge YOLOP and YOLOv8 detections, using YOLOv8 for large boxes and YOLOP for small/medium."""
        
        # Normalize YOLOP detections to have 'bbox' key and calculate box sizes
        for det in yolop_detections:
            # YOLOP uses 'box' key, convert to 'bbox' for consistency
            if 'box' in det and 'bbox' not in det:
                det['bbox'] = det['box']
            elif 'bbox' not in det:
                # Skip detections without box information
                logger.warning(f"Detection missing bbox/box key: {det.keys()}")
                continue
            
            bbox = det.get('bbox') or det.get('box')
            if not bbox:
                continue
                
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            det['box_area_ratio'] = (width * height) / (frame_width * frame_height)
            det['source'] = 'yolop'
        
        # Filter out detections without bbox
        yolop_detections = [d for d in yolop_detections if 'bbox' in d or 'box' in d]
        
        # Log raw detection counts before filtering
        logger.info(f"Raw detections before filtering - YOLOv8: {len(yolov8_detections)}, YOLOP: {len(yolop_detections)}")
        
        # Threshold: Use YOLOv8 for boxes > 2% of frame (DRAMATICALLY lowered from 10%)
        # Dashcam footage typically has vehicles occupying 0.5-5% of frame
        # This allows YOLOv8 to contribute for medium/close vehicles (>2%) while YOLOP handles distant (<2%)
        large_box_threshold = 0.02
        
        # Keep large YOLOv8 detections (better for close vehicles)
        large_yolov8 = [d for d in yolov8_detections if d.get('box_area_ratio', 0) > large_box_threshold]
        
        # Keep small/medium YOLOP detections (better for distant vehicles)
        small_medium_yolop = [d for d in yolop_detections if d.get('box_area_ratio', 0) <= large_box_threshold]
        
        # Log size distribution
        if yolov8_detections:
            yolov8_sizes = [d.get('box_area_ratio', 0) * 100 for d in yolov8_detections]
            logger.info(f"YOLOv8 box sizes (% of frame): {[f'{s:.1f}%' for s in yolov8_sizes[:5]]}")  # First 5
        
        # Convert YOLOv8 detections to YOLOP format (with 'box' key for compatibility)
        for det in large_yolov8:
            if 'bbox' in det:
                det['box'] = det['bbox']
                det['box_center'] = [(det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2]
                det['width'] = det['bbox'][2] - det['bbox'][0]
                det['height'] = det['bbox'][3] - det['bbox'][1]
                det['area'] = det['width'] * det['height']
        
        # Combine detections
        merged = large_yolov8 + small_medium_yolop
        
        # Remove duplicates using NMS-like approach
        merged = self._remove_duplicate_detections(merged)
        
        logger.info(f"Merged detections: {len(large_yolov8)} large (YOLOv8) + {len(small_medium_yolop)} small/medium (YOLOP) = {len(merged)} total")
        
        return merged

    def _remove_duplicate_detections(
        self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Remove duplicate detections using IoU threshold."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            
            # Get bbox from either 'bbox' or 'box' key
            best_bbox = best.get('bbox', best.get('box'))
            
            # Remove overlapping detections
            filtered_dets = []
            for det in sorted_dets:
                det_bbox = det.get('bbox', det.get('box'))
                if self._calculate_iou(best_bbox, det_bbox) < iou_threshold:
                    filtered_dets.append(det)
            sorted_dets = filtered_dets
        
        return keep

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0