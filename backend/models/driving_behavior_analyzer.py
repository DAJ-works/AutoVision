"""Heuristic driving behavior analysis built on tracked vehicle data."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


class DrivingBehaviorAnalyzer:
    """Analyze tracked vehicle motion to flag unsafe driving patterns."""

    VEHICLE_CLASSES = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "van",
        "suv",
        "pickup",
        "vehicle",
        "ambulance",
        "fire truck",
        "police car",
        "taxi",
    }

    SIGN_CLASSES = {
        "stop sign",
        "traffic light",
        "traffic_sign",
        "stop_sign",
        "speed limit",
        "speed_limit",
        "yield sign",
        "yield",
        "traffic sign",
    }

    def __init__(
        self,
        high_speed_norm: float = 0.12,
        brake_delta_norm: float = 0.07,
        stop_speed_norm: float = 0.02,
        lane_shift_norm: float = 0.12,
        sign_distance_norm: float = 0.1,
    ) -> None:
        self.high_speed_norm = high_speed_norm
        self.brake_delta_norm = brake_delta_norm
        self.stop_speed_norm = stop_speed_norm
        self.lane_shift_norm = lane_shift_norm
        self.sign_distance_norm = sign_distance_norm

    def evaluate(
        self,
        frames: Mapping[int, Mapping[str, Any]],
        tracks: Mapping[Any, Mapping[str, Any]],
        video_metadata: Optional[Mapping[str, Any]] = None,
        class_counts: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Produce driving behavior insights using tracked objects and detections."""

        metadata = video_metadata or {}
        width = float(metadata.get("width") or metadata.get("frame_width") or 0)
        height = float(metadata.get("height") or metadata.get("frame_height") or 0)
        fps = float(metadata.get("fps") or 0) or 30.0
        frame_diag = math.hypot(width, height) if width and height else 1.0

        vehicle_tracks = {
            track_id: track
            for track_id, track in tracks.items()
            if track.get("class_name", "").lower() in self._normalized_vehicle_classes()
        }

        durations: List[float] = []
        for track in vehicle_tracks.values():
            duration = self._track_duration(track, fps)
            if duration is not None:
                durations.append(duration)

        vehicle_counts = self._vehicle_counts(class_counts or {})
        total_vehicle_detections = sum(vehicle_counts.values())

        prev_speeds: Dict[Any, float] = {}
        prev_centers: Dict[Any, Tuple[float, float]] = {}
        unsafe_events: List[Dict[str, Any]] = []
        sign_interactions: List[Dict[str, Any]] = []
        event_counts: defaultdict[str, int] = defaultdict(int)
        severity_counts: defaultdict[str, int] = defaultdict(int)
        already_reported_collisions: Set[Tuple[Any, Any]] = set()
        speed_samples: List[float] = []
        high_speed_events = 0
        sudden_brake_events = 0
        lane_shift_events = 0
        collision_events = 0

        normalized_frames = sorted(frames.items(), key=lambda item: item[0])
        for frame_number, frame_data in normalized_frames:
            timestamp = self._safe_float(frame_data.get("timestamp"))
            tracked_objects = frame_data.get("tracks") or []
            vehicle_objects = [
                obj for obj in tracked_objects if obj.get("class_name", "").lower() in self._normalized_vehicle_classes()
            ]

            # Speed and brake detection
            for obj in vehicle_objects:
                track_id = obj.get("object_id", obj.get("id"))
                if track_id is None:
                    continue

                speed_norm = self._normalized_speed(obj, frame_diag)
                if speed_norm is None:
                    continue

                speed_samples.append(speed_norm)

                if speed_norm >= self.high_speed_norm:
                    high_speed_events += 1
                    self._record_event(
                        unsafe_events,
                        event_counts,
                        severity_counts,
                        event_type="high_speed_movement",
                        severity="medium",
                        description="Vehicle exhibiting high relative speed",
                        frame=frame_number,
                        timestamp=timestamp,
                        track_ids=[track_id],
                        metrics={"normalized_speed": speed_norm},
                    )

                prev_speed = prev_speeds.get(track_id)
                if prev_speed is not None:
                    deceleration = prev_speed - speed_norm
                    if prev_speed >= self.high_speed_norm and deceleration >= self.brake_delta_norm and speed_norm <= self.stop_speed_norm:
                        sudden_brake_events += 1
                        self._record_event(
                            unsafe_events,
                            event_counts,
                            severity_counts,
                            event_type="sudden_brake",
                            severity="medium",
                            description="Abrupt deceleration detected",
                            frame=frame_number,
                            timestamp=timestamp,
                            track_ids=[track_id],
                            metrics={
                                "prior_speed": prev_speed,
                                "current_speed": speed_norm,
                                "delta": deceleration,
                            },
                        )

                prev_speeds[track_id] = speed_norm

                center = self._object_center(obj)
                if center:
                    prev_center = prev_centers.get(track_id)
                    if prev_center:
                        lateral_shift = abs(center[0] - prev_center[0]) / max(width, 1.0)
                        if (
                            lateral_shift >= self.lane_shift_norm
                            and speed_norm >= self.stop_speed_norm
                        ):
                            lane_shift_events += 1
                            self._record_event(
                                unsafe_events,
                                event_counts,
                                severity_counts,
                                event_type="aggressive_lane_change",
                                severity="low",
                                description="Significant lateral movement in short interval",
                                frame=frame_number,
                                timestamp=timestamp,
                                track_ids=[track_id],
                                metrics={
                                    "normalized_speed": speed_norm,
                                    "lateral_shift": lateral_shift,
                                },
                            )
                    prev_centers[track_id] = center

            # Collision heuristics
            for i, obj_a in enumerate(vehicle_objects):
                track_a = obj_a.get("object_id", obj_a.get("id"))
                box_a = obj_a.get("box")
                speed_a = self._normalized_speed(obj_a, frame_diag) or 0.0
                if box_a is None or track_a is None:
                    continue
                for obj_b in vehicle_objects[i + 1 :]:
                    track_b = obj_b.get("object_id", obj_b.get("id"))
                    box_b = obj_b.get("box")
                    speed_b = self._normalized_speed(obj_b, frame_diag) or 0.0
                    if box_b is None or track_b is None:
                        continue

                    pair_key = tuple(sorted((track_a, track_b)))
                    if pair_key in already_reported_collisions:
                        continue

                    iou = self._iou(box_a, box_b)
                    if iou >= 0.12 and (speed_a + speed_b) >= self.stop_speed_norm * 2:
                        collision_events += 1
                        already_reported_collisions.add(pair_key)
                        self._record_event(
                            unsafe_events,
                            event_counts,
                            severity_counts,
                            event_type="possible_collision",
                            severity="high",
                            description="Vehicles overlapping with significant motion",
                            frame=frame_number,
                            timestamp=timestamp,
                            track_ids=list(pair_key),
                            metrics={
                                "iou": iou,
                                "speed_sum": speed_a + speed_b,
                            },
                        )

            # Sign context
            sign_detections = [
                det for det in frame_data.get("detections", [])
                if det.get("class_name", "").lower() in self._normalized_sign_classes()
            ]

            if sign_detections:
                for sign in sign_detections:
                    sign_box = sign.get("box")
                    if not sign_box:
                        continue
                    sign_center = self._box_center(sign_box)
                    nearby_tracks: List[Dict[str, Any]] = []
                    for vehicle in vehicle_objects:
                        track_id = vehicle.get("object_id", vehicle.get("id"))
                        center = self._object_center(vehicle)
                        if not center or track_id is None:
                            continue
                        distance_norm = self._normalized_distance(center, sign_center, frame_diag)
                        if distance_norm <= self.sign_distance_norm:
                            speed_norm = self._normalized_speed(vehicle, frame_diag) or 0.0
                            nearby_tracks.append(
                                {
                                    "track_id": track_id,
                                    "normalized_speed": speed_norm,
                                    "distance_norm": distance_norm,
                                }
                            )

                            if sign.get("class_name", "").lower().startswith("stop") and speed_norm > self.stop_speed_norm:
                                self._record_event(
                                    unsafe_events,
                                    event_counts,
                                    severity_counts,
                                    event_type="possible_stop_violation",
                                    severity="medium",
                                    description="Vehicle near stop sign without evidence of stopping",
                                    frame=frame_number,
                                    timestamp=timestamp,
                                    track_ids=[track_id],
                                    metrics={
                                        "normalized_speed": speed_norm,
                                        "distance_norm": distance_norm,
                                    },
                                )

                    sign_interactions.append(
                        {
                            "sign_class": sign.get("class_name"),
                            "frame": frame_number,
                            "timestamp": timestamp,
                            "vehicles_in_vicinity": nearby_tracks,
                        }
                    )

        average_duration = mean(durations) if durations else 0.0
        median_duration = median(durations) if durations else 0.0
        average_speed_norm = mean(speed_samples) if speed_samples else 0.0
        max_speed_norm = max(speed_samples) if speed_samples else 0.0

        risk_score = min(
            100.0,
            severity_counts.get("high", 0) * 35
            + severity_counts.get("medium", 0) * 20
            + severity_counts.get("low", 0) * 10,
        )

        insights: List[str] = []
        if severity_counts.get("high", 0):
            insights.append(f"{severity_counts['high']} high severity unsafe driving events detected.")
        else:
            insights.append("No high severity unsafe driving events detected in the evaluated footage.")
        if high_speed_events:
            insights.append(f"High speed movement flagged {high_speed_events} time(s), review for speeding risk.")
        if sudden_brake_events:
            insights.append(f"Detected {sudden_brake_events} possible hard brake event(s).")
        if collision_events:
            insights.append("Collision risk detected; inspect overlapping vehicle tracks.")
        if not insights:
            insights.append("Driving behavior appears nominal under current heuristics.")

        vehicle_summary = {
            "total_vehicle_tracks": len(vehicle_tracks),
            "average_track_duration_sec": average_duration,
            "median_track_duration_sec": median_duration,
            "average_speed_norm": average_speed_norm,
            "max_speed_norm": max_speed_norm,
            "high_speed_events": high_speed_events,
            "sudden_brake_events": sudden_brake_events,
            "lane_shift_events": lane_shift_events,
            "possible_collisions": collision_events,
            "frames_evaluated": len(normalized_frames),
        }

        compliance_summary = {
            "events_by_type": dict(event_counts),
            "events_by_severity": dict(severity_counts),
            "risk_score": risk_score,
        }

        return {
            "vehicle_summary": vehicle_summary,
            "vehicle_counts": vehicle_counts,
            "total_vehicle_detections": total_vehicle_detections,
            "unsafe_events": unsafe_events,
            "sign_interactions": sign_interactions,
            "compliance_summary": compliance_summary,
            "insights": insights,
        }

    def _normalized_vehicle_classes(self) -> set[str]:
        return {cls.lower() for cls in self.VEHICLE_CLASSES}

    def vehicle_class_names(self) -> set[str]:
        return self._normalized_vehicle_classes()

    def _normalized_sign_classes(self) -> set[str]:
        return {cls.lower() for cls in self.SIGN_CLASSES}

    def _normalized_speed(self, obj: Mapping[str, Any], frame_diag: float) -> Optional[float]:
        velocity = obj.get("velocity")
        if not velocity:
            trajectory = obj.get("trajectory")
            timestamps = obj.get("timestamps")
            if trajectory and timestamps and len(trajectory) >= 2 and len(timestamps) >= 2:
                try:
                    dx = float(trajectory[-1][0]) - float(trajectory[-2][0])
                    dy = float(trajectory[-1][1]) - float(trajectory[-2][1])
                    dt = float(timestamps[-1]) - float(timestamps[-2])
                except (TypeError, ValueError):
                    return None
                if dt:
                    vx = dx / dt
                    vy = dy / dt
                    velocity = (vx, vy)
        if not velocity:
            return None
        try:
            vx, vy = float(velocity[0]), float(velocity[1])
        except (TypeError, ValueError, IndexError):
            return None
        speed = math.hypot(vx, vy)
        return speed / frame_diag if frame_diag else speed

    def _object_center(self, obj: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
        center = obj.get("box_center")
        if center:
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError, IndexError):
                pass
        box = obj.get("box")
        if not box or len(box) != 4:
            return None
        return self._box_center(box)

    def _box_center(self, box: Iterable[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(c) for c in box]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _normalized_distance(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        frame_diag: float,
    ) -> float:
        if frame_diag <= 0:
            return 0.0
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]) / frame_diag

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _iou(self, box_a: Iterable[float], box_b: Iterable[float]) -> float:
        try:
            ax1, ay1, ax2, ay2 = [float(c) for c in box_a]
            bx1, by1, bx2, by2 = [float(c) for c in box_b]
        except (TypeError, ValueError):
            return 0.0

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _record_event(
        self,
        events: List[Dict[str, Any]],
        event_counts: defaultdict[str, int],
        severity_counts: defaultdict[str, int],
        *,
        event_type: str,
        severity: str,
        description: str,
        frame: int,
        timestamp: Optional[float],
        track_ids: List[Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        event_counts[event_type] += 1
        severity_counts[severity] += 1
        events.append(
            {
                "type": event_type,
                "severity": severity,
                "description": description,
                "frame": frame,
                "timestamp": timestamp,
                "track_ids": track_ids,
                "metrics": metrics or {},
            }
        )

    def _track_duration(self, track: Mapping[str, Any], fps: float) -> Optional[float]:
        first_time = self._safe_float(track.get("first_time"))
        last_time = self._safe_float(track.get("last_time"))
        if first_time is not None and last_time is not None:
            return max(0.0, last_time - first_time)
        first_frame = track.get("first_frame")
        last_frame = track.get("last_frame")
        if first_frame is not None and last_frame is not None:
            try:
                frame_delta = int(last_frame) - int(first_frame)
            except (TypeError, ValueError):
                return None
            if fps:
                return max(0.0, frame_delta / fps)
        return None

    def _vehicle_counts(self, class_counts: Mapping[str, int]) -> Dict[str, int]:
        normalized_vehicle_classes = self._normalized_vehicle_classes()
        vehicle_counts: Dict[str, int] = {}
        for class_name, count in class_counts.items():
            if class_name.lower() in normalized_vehicle_classes:
                vehicle_counts[class_name] = int(count)
        return vehicle_counts
