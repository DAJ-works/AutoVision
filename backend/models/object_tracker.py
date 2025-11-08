"""Lightweight object tracking utilities for dashcam footage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    CONFIRMED = 3
    DELETED = 4


@dataclass
class Track:
    detection: Dict[str, Any]
    track_id: int
    frame_idx: Optional[int] = None
    timestamp: Optional[float] = None

    box: List[float] = field(init=False)
    box_center: List[float] = field(init=False)
    confidence: float = field(init=False)
    class_id: int = field(init=False)
    class_name: str = field(init=False)
    object_type: str = field(init=False)
    state: TrackState = field(default=TrackState.NEW, init=False)
    age: int = field(default=1, init=False)
    hits: int = field(default=1, init=False)
    time_since_update: int = field(default=0, init=False)
    velocity: Tuple[float, float] = field(default=(0.0, 0.0), init=False)
    boxes: List[List[float]] = field(default_factory=list, init=False)
    centers: List[List[float]] = field(default_factory=list, init=False)
    frames: List[int] = field(default_factory=list, init=False)
    timestamps: List[float] = field(default_factory=list, init=False)
    attributes: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.box = self.detection.get("box", [0.0, 0.0, 0.0, 0.0])
        self.box_center = self.detection.get("box_center") or self._compute_center(self.box)
        self.confidence = float(self.detection.get("confidence", 0.0))
        self.class_id = int(self.detection.get("class_id", -1))
        self.class_name = self.detection.get("class_name", "")
        self.object_type = self.detection.get("type", "normal")
        self._update_attributes(self.detection)
        self._record_observation(self.frame_idx, self.timestamp)

    def _compute_center(self, box: List[float]) -> List[float]:
        x1, y1, x2, y2 = box
        return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]

    def _update_attributes(self, detection: Dict[str, Any]) -> None:
        for key in ("color", "properties", "threat_level", "validation_score"):
            if key in detection:
                self.attributes[key] = detection[key]

    def _record_observation(self, frame_idx: Optional[int], timestamp: Optional[float]) -> None:
        self.boxes.append([float(coord) for coord in self.box])
        self.centers.append([float(coord) for coord in self.box_center])
        if frame_idx is not None:
            self.frames.append(int(frame_idx))
        if timestamp is not None:
            self.timestamps.append(float(timestamp))

    def update(self, detection: Dict[str, Any], frame_idx: Optional[int], timestamp: Optional[float]) -> None:
        prev_center = self.box_center
        prev_timestamp = self.timestamps[-1] if self.timestamps else None
        prev_frame = self.frames[-1] if self.frames else None

        self.box = detection.get("box", self.box)
        self.box_center = detection.get("box_center") or self._compute_center(self.box)
        self.confidence = float(detection.get("confidence", self.confidence))
        self.class_id = int(detection.get("class_id", self.class_id))
        self.class_name = detection.get("class_name", self.class_name)
        self.object_type = detection.get("type", self.object_type)
        self._update_attributes(detection)

        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED

        dt = None
        if timestamp is not None and prev_timestamp is not None and timestamp != prev_timestamp:
            dt = timestamp - prev_timestamp
        elif frame_idx is not None and prev_frame is not None and frame_idx != prev_frame:
            dt = frame_idx - prev_frame

        if dt:
            vx = (self.box_center[0] - prev_center[0]) / dt
            vy = (self.box_center[1] - prev_center[1]) / dt
            self.velocity = (float(vx), float(vy))

        self._record_observation(frame_idx, timestamp)

    def mark_missed(self) -> None:
        if self.state in {TrackState.TRACKED, TrackState.CONFIRMED}:
            self.state = TrackState.LOST
        self.time_since_update += 1

    def promote(self, min_hits: int) -> None:
        if self.state == TrackState.NEW and self.hits >= min_hits:
            self.state = TrackState.CONFIRMED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.track_id,
            "id": self.track_id,
            "box": self.box,
            "box_center": self.box_center,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "type": self.object_type,
            "state": self.state.name,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "velocity": self.velocity,
            "trajectory": self.centers,
            "boxes": self.boxes,
            "frames": self.frames,
            "timestamps": self.timestamps,
            "attributes": self.attributes,
        }


class ObjectTracker:
    def __init__(self, max_age: int = 60, min_hits: int = 1, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.track_id = 0

    def reset(self) -> None:
        self.tracks = []
        self.track_id = 0

    def _iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _cost_matrix(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        if not detections or not self.tracks:
            return np.zeros((len(detections), len(self.tracks)))

        cost = np.zeros((len(detections), len(self.tracks)), dtype=float)
        for det_idx, detection in enumerate(detections):
            det_box = np.asarray(detection.get("box", [0, 0, 0, 0]), dtype=float)
            det_class = detection.get("class_id", -1)
            for track_idx, track in enumerate(self.tracks):
                if track.class_id != det_class:
                    cost[det_idx, track_idx] = 1.0
                    continue
                track_box = np.asarray(track.box, dtype=float)
                cost[det_idx, track_idx] = 1.0 - self._iou(det_box, track_box)
        return cost

    def _assign(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        cost_matrix = self._cost_matrix(detections)
        if cost_matrix.size == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        remaining_dets = set(range(len(detections)))
        remaining_tracks = set(range(len(self.tracks)))
        matches: List[Tuple[int, int]] = []

        while remaining_dets and remaining_tracks:
            best_pair: Optional[Tuple[int, int]] = None
            best_cost = float("inf")

            for det_idx in remaining_dets:
                for track_idx in remaining_tracks:
                    cost = cost_matrix[det_idx, track_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (det_idx, track_idx)

            if best_pair is None or best_cost > (1.0 - self.iou_threshold):
                break

            matches.append(best_pair)
            remaining_dets.remove(best_pair[0])
            remaining_tracks.remove(best_pair[1])

        unmatched_dets = sorted(list(remaining_dets))
        unmatched_tracks = sorted(list(remaining_tracks))

        return matches, unmatched_dets, unmatched_tracks

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        for track in self.tracks:
            track.time_since_update += 1

        matches, unmatched_det_indices, unmatched_track_indices = self._assign(detections)

        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], frame_number, timestamp)

        for track_idx in unmatched_track_indices:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_det_indices:
            track = Track(detections[det_idx], self.track_id, frame_number, timestamp)
            track.promote(self.min_hits)
            self.tracks.append(track)
            self.track_id += 1

        active_tracks: List[Track] = []
        retained_tracks: List[Track] = []
        for track in self.tracks:
            if track.time_since_update > self.max_age:
                track.state = TrackState.DELETED
                continue

            if track.state in {TrackState.TRACKED, TrackState.CONFIRMED}:
                active_tracks.append(track)
            else:
                retained_tracks.append(track)

        self.tracks = active_tracks + retained_tracks
        for track in self.tracks:
            track.promote(self.min_hits)

        return [t.to_dict() for t in self.tracks if t.state in {TrackState.TRACKED, TrackState.CONFIRMED}]

    def get_tracks(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tracks if t.state in {TrackState.TRACKED, TrackState.CONFIRMED}]

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        return self.get_tracks()

    def get_trajectories(self) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []
        for track in self.tracks:
            if not track.centers:
                continue
            trajectories.append(
                {
                    "object_id": track.track_id,
                    "class_name": track.class_name,
                    "type": track.object_type,
                    "trajectory": track.centers,
                    "boxes": track.boxes,
                    "frames": track.frames,
                    "timestamps": track.timestamps,
                    "attributes": track.attributes,
                }
            )
        return trajectories