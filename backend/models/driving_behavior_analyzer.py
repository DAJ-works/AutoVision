"""Driving behavior analysis with YOLOP-aware context and real-world metrics."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import mean, median
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class DrivingBehaviorAnalyzer:
    """Infer real-world vehicle behavior from tracked objects and YOLOP outputs."""

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

    VEHICLE_WIDTHS_METERS = {
        "car": 1.85,
        "truck": 2.60,
        "bus": 2.60,
        "motorcycle": 0.85,
        "bicycle": 0.65,
        "van": 1.95,
        "suv": 2.05,
        "pickup": 2.10,
    }

    DEFAULT_LANE_WIDTH_M = 3.7

    def __init__(
        self,
        high_speed_kmh: float = 90.0,
        brake_delta_kmh: float = 40.0,
        stop_speed_kmh: float = 5.0,
        lane_shift_m: float = 1.2,
        near_sign_distance_m: float = 18.0,
        collision_iou_threshold: float = 0.18,
        scale_history: int = 8,
    ) -> None:
        self.high_speed_kmh = high_speed_kmh
        self.brake_delta_kmh = brake_delta_kmh
        self.stop_speed_kmh = stop_speed_kmh
        self.lane_shift_m = lane_shift_m
        self.near_sign_distance_m = near_sign_distance_m
        self.collision_iou_threshold = collision_iou_threshold
        self.scale_history_size = max(1, scale_history)

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.scale_history: Deque[float] = deque(maxlen=self.scale_history_size)
        self.current_scale_m_per_px: Optional[float] = None
        self.scale_sources: Deque[str] = deque(maxlen=self.scale_history_size)
        self.speed_samples_kmh: List[float] = []
        self.per_frame_speed_kmh: List[float] = []
        self.events: List[Dict[str, Any]] = []
        self.event_counts: defaultdict[str, int] = defaultdict(int)
        self.severity_counts: defaultdict[str, int] = defaultdict(int)
        self.sign_interactions: List[Dict[str, Any]] = []
        self.road_coverage_history: List[float] = []
        self.lane_coverage_history: List[float] = []
        self.track_state: Dict[Any, Dict[str, Any]] = {}
        self.track_vehicle_counts: defaultdict[str, int] = defaultdict(int)
        self.high_speed_event_count = 0
        self.hard_brake_event_count = 0
        self.lane_shift_event_count = 0
        self.collision_event_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_frame(
        self,
        *,
        frame_number: int,
        timestamp: Optional[float],
        tracked_objects: Sequence[Mapping[str, Any]],
        detections: Sequence[Mapping[str, Any]],
        auxiliary_outputs: Mapping[str, Any],
        fps: float,
        frame_shape: Tuple[int, int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Update internal metrics using the current frame."""

        seconds = timestamp
        if seconds is None and fps > 0:
            seconds = frame_number / fps

        if seconds is None:
            seconds = 0.0

        lane_mask = auxiliary_outputs.get("lane_line_mask")
        drivable_mask = auxiliary_outputs.get("drivable_area_mask")

        frame_scale = self._update_scale_estimate(
            lane_mask=lane_mask,
            detections=detections,
            frame_shape=frame_shape,
        )
        if frame_scale is not None:
            self.scale_history.append(frame_scale)
            source = auxiliary_outputs.get("scale_source", "blend")
            self.scale_sources.append(source)
        if self.scale_history:
            self.current_scale_m_per_px = float(mean(self.scale_history))

        if drivable_mask is not None:
            drivable_ratio = float(np.count_nonzero(drivable_mask)) / drivable_mask.size
            self.road_coverage_history.append(drivable_ratio)
        if lane_mask is not None:
            lane_ratio = float(np.count_nonzero(lane_mask)) / lane_mask.size
            self.lane_coverage_history.append(lane_ratio)

        vehicle_objects = [
            obj
            for obj in tracked_objects
            if obj.get("class_name", "").lower() in self.vehicle_class_names()
        ]

        events_this_frame: List[Dict[str, Any]] = []

        for obj in vehicle_objects:
            track_id = self._track_id(obj)
            if track_id is None:
                continue

            class_name = obj.get("class_name", "unknown").lower()
            self.track_vehicle_counts[class_name] += 1

            center = self._object_center(obj)
            if center is None:
                continue

            state = self.track_state.setdefault(
                track_id,
                {
                    "class_name": class_name,
                    "first_seen_time": seconds,
                    "first_frame": frame_number,
                    "speed_samples": deque(maxlen=60),
                    "max_speed_kmh": 0.0,
                    "distance_m": 0.0,
                    "last_center": None,
                    "last_time": None,
                    "prev_speed_kmh": None,
                },
            )

            last_time = state.get("last_time")
            last_center = state.get("last_center")
            dt = seconds - last_time if last_time is not None else None

            if dt and dt > 0 and last_center is not None and self.current_scale_m_per_px:
                pixel_dx = center[0] - last_center[0]
                pixel_dy = center[1] - last_center[1]
                pixel_distance = math.hypot(pixel_dx, pixel_dy)
                distance_m = pixel_distance * self.current_scale_m_per_px
                speed_mps = distance_m / dt if dt > 0 else 0.0
                speed_kmh = speed_mps * 3.6

                state["distance_m"] += distance_m
                state["speed_samples"].append(speed_kmh)
                state["max_speed_kmh"] = max(state["max_speed_kmh"], speed_kmh)
                state["prev_speed_kmh"] = state.get("current_speed_kmh")
                state["current_speed_kmh"] = speed_kmh

                self.speed_samples_kmh.append(speed_kmh)
                self.per_frame_speed_kmh.append(speed_kmh)

                if speed_kmh >= self.high_speed_kmh:
                    self.high_speed_event_count += 1
                    events_this_frame.append(
                        self._record_event(
                            event_type="high_speed",
                            severity="medium",
                            description="Vehicle exceeds configured speed threshold",
                            frame=frame_number,
                            timestamp=seconds,
                            track_ids=[track_id],
                            metrics={"speed_kmh": speed_kmh},
                        )
                    )

                prev_speed = state.get("prev_speed_kmh")
                if (
                    prev_speed is not None
                    and prev_speed - speed_kmh >= self.brake_delta_kmh
                    and speed_kmh <= self.stop_speed_kmh
                ):
                    self.hard_brake_event_count += 1
                    events_this_frame.append(
                        self._record_event(
                            event_type="hard_brake",
                            severity="medium",
                            description="Rapid deceleration detected",
                            frame=frame_number,
                            timestamp=seconds,
                            track_ids=[track_id],
                            metrics={
                                "prior_speed_kmh": prev_speed,
                                "current_speed_kmh": speed_kmh,
                                "delta_kmh": prev_speed - speed_kmh,
                            },
                        )
                    )

                lateral_pixels = abs(pixel_dx)
                lateral_m = lateral_pixels * self.current_scale_m_per_px
                if lateral_m >= self.lane_shift_m and speed_kmh >= self.stop_speed_kmh:
                    self.lane_shift_event_count += 1
                    events_this_frame.append(
                        self._record_event(
                            event_type="lane_departure",
                            severity="low",
                            description="Significant lateral movement detected",
                            frame=frame_number,
                            timestamp=seconds,
                            track_ids=[track_id],
                            metrics={"lateral_shift_m": lateral_m, "speed_kmh": speed_kmh},
                        )
                    )

            state["last_center"] = center
            state["last_time"] = seconds

        if self.current_scale_m_per_px:
            self._detect_collisions(
                frame_number=frame_number,
                timestamp=seconds,
                vehicles=vehicle_objects,
            )

        self._detect_sign_interactions(
            frame_number=frame_number,
            timestamp=seconds,
            tracked_vehicles=vehicle_objects,
            detections=detections,
        )

        frame_metrics = {
            "meters_per_pixel": self.current_scale_m_per_px,
            "road_occupancy": self.road_coverage_history[-1] if self.road_coverage_history else None,
            "lane_line_density": self.lane_coverage_history[-1] if self.lane_coverage_history else None,
            "average_speed_kmh": mean(self.per_frame_speed_kmh[-len(vehicle_objects) :])
            if vehicle_objects and self.per_frame_speed_kmh
            else None,
        }

        return events_this_frame, frame_metrics

    def finalize(
        self,
        *,
        video_metadata: Mapping[str, Any],
        class_counts: Mapping[str, int],
        tracks: Mapping[Any, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Produce a final report summarizing collected driving metrics."""

        vehicle_counts = self._vehicle_counts(class_counts)
        total_vehicle_detections = int(sum(vehicle_counts.values()))

        vehicle_tracks = {
            track_id: track
            for track_id, track in tracks.items()
            if track.get("class_name", "").lower() in self.vehicle_class_names()
        }

        durations: List[float] = []
        fps = float(video_metadata.get("fps") or 0)
        for track in vehicle_tracks.values():
            duration = self._track_duration(track, fps)
            if duration is not None:
                durations.append(duration)

        avg_duration = mean(durations) if durations else 0.0
        med_duration = median(durations) if durations else 0.0

        avg_speed = mean(self.speed_samples_kmh) if self.speed_samples_kmh else 0.0
        max_speed = max(self.speed_samples_kmh) if self.speed_samples_kmh else 0.0

        vehicle_summary = {
            "total_vehicle_tracks": len(vehicle_tracks),
            "average_track_duration_sec": avg_duration,
            "median_track_duration_sec": med_duration,
            "average_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "estimated_scale_m_per_px": self.current_scale_m_per_px,
            "high_speed_events": self.high_speed_event_count,
            "hard_brake_events": self.hard_brake_event_count,
            "lane_shift_events": self.lane_shift_event_count,
            "possible_collisions": self.collision_event_count,
        }

        compliance_summary = {
            "events_by_type": dict(self.event_counts),
            "events_by_severity": dict(self.severity_counts),
            "risk_score": self._risk_score(),
            "avg_road_occupancy": mean(self.road_coverage_history) if self.road_coverage_history else None,
            "avg_lane_density": mean(self.lane_coverage_history) if self.lane_coverage_history else None,
        }

        insights = self._generate_insights(vehicle_summary, compliance_summary)

        return {
            "vehicle_summary": vehicle_summary,
            "vehicle_counts": dict(vehicle_counts),
            "total_vehicle_detections": total_vehicle_detections,
            "unsafe_events": self.events,
            "sign_interactions": self.sign_interactions,
            "compliance_summary": compliance_summary,
            "insights": insights,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def vehicle_class_names(self) -> set[str]:
        return {cls.lower() for cls in self.VEHICLE_CLASSES}

    def _track_id(self, obj: Mapping[str, Any]) -> Optional[Any]:
        return obj.get("object_id") or obj.get("id")

    def _object_center(self, obj: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
        center = obj.get("box_center")
        if center and len(center) == 2:
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError):
                return None
        box = obj.get("box")
        if box and len(box) == 4:
            return self._box_center(box)
        return None

    def _box_center(self, box: Iterable[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(coord) for coord in box]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _record_event(
        self,
        *,
        event_type: str,
        severity: str,
        description: str,
        frame: int,
        timestamp: float,
        track_ids: List[Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "type": event_type,
            "severity": severity,
            "description": description,
            "frame": frame,
            "timestamp": timestamp,
            "track_ids": track_ids,
            "metrics": metrics or {},
        }
        self.events.append(event)
        self.event_counts[event_type] += 1
        self.severity_counts[severity] += 1
        return event

    def _update_scale_estimate(
        self,
        *,
        lane_mask: Optional[np.ndarray],
        detections: Sequence[Mapping[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> Optional[float]:
        lane_scale = self._scale_from_lane_lines(lane_mask)
        vehicle_scale = self._scale_from_vehicles(detections, frame_shape)

        if lane_scale and vehicle_scale:
            return (lane_scale + vehicle_scale) / 2.0
        if lane_scale:
            return lane_scale
        if vehicle_scale:
            return vehicle_scale
        return None

    def _scale_from_lane_lines(self, lane_mask: Optional[np.ndarray]) -> Optional[float]:
        if lane_mask is None or lane_mask.size == 0:
            return None

        h, w = lane_mask.shape[:2]
        focus_region = lane_mask[int(h * 0.6) :, :]
        if focus_region.size == 0 or not np.any(focus_region):
            return None

        column_hist = focus_region.sum(axis=0)
        if not np.any(column_hist):
            return None

        threshold = column_hist.max() * 0.35
        active_columns = np.where(column_hist >= threshold)[0]
        if active_columns.size < 2:
            return None

        clusters = self._cluster_indices(active_columns)
        boundary_count = len(clusters)
        if boundary_count < 2:
            return None

        left_edge = clusters[0][0]
        right_edge = clusters[-1][-1]
        pixel_width = max(1, right_edge - left_edge)

        estimated_lanes = max(1, boundary_count - 1)
        road_width_m = self.DEFAULT_LANE_WIDTH_M * estimated_lanes
        return road_width_m / pixel_width

    def _scale_from_vehicles(
        self,
        detections: Sequence[Mapping[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> Optional[float]:
        if not detections:
            return None

        h, _ = frame_shape
        candidate_scales: List[float] = []
        for det in detections:
            class_name = det.get("class_name", "").lower()
            if class_name not in self.vehicle_class_names():
                continue

            box = det.get("box")
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = [float(coord) for coord in box]
            width_px = max(1.0, x2 - x1)
            center_y = (y1 + y2) / 2.0
            if center_y < h * 0.55:
                continue  # use objects close to the camera for scale

            nominal_width = self.VEHICLE_WIDTHS_METERS.get(class_name, 1.9)
            candidate_scales.append(nominal_width / width_px)

        if not candidate_scales:
            return None

        return float(median(candidate_scales))

    def _cluster_indices(self, indices: np.ndarray, gap: int = 6) -> List[np.ndarray]:
        clusters: List[np.ndarray] = []
        current: List[int] = []
        prev = None
        for idx in indices:
            if prev is None or idx - prev <= gap:
                current.append(idx)
            else:
                clusters.append(np.array(current))
                current = [idx]
            prev = idx
        if current:
            clusters.append(np.array(current))
        return clusters

    def _detect_collisions(
        self,
        *,
        frame_number: int,
        timestamp: float,
        vehicles: Sequence[Mapping[str, Any]],
    ) -> None:
        for i, vehicle_a in enumerate(vehicles):
            track_a = self._track_id(vehicle_a)
            box_a = vehicle_a.get("box")
            if track_a is None or box_a is None:
                continue

            state_a = self.track_state.get(track_a, {})
            speed_a = state_a.get("current_speed_kmh") or 0.0

            for vehicle_b in vehicles[i + 1 :]:
                track_b = self._track_id(vehicle_b)
                box_b = vehicle_b.get("box")
                if track_b is None or box_b is None:
                    continue

                state_b = self.track_state.get(track_b, {})
                speed_b = state_b.get("current_speed_kmh") or 0.0

                iou = self._iou(box_a, box_b)
                if iou < self.collision_iou_threshold:
                    continue

                relative_speed = abs(speed_a - speed_b)
                if relative_speed < 12.0:  # require noticeable speed mismatch
                    continue

                self.collision_event_count += 1
                self._record_event(
                    event_type="possible_collision",
                    severity="high",
                    description="Overlapping vehicles with high relative speed",
                    frame=frame_number,
                    timestamp=timestamp,
                    track_ids=[track_a, track_b],
                    metrics={
                        "iou": iou,
                        "speed_a_kmh": speed_a,
                        "speed_b_kmh": speed_b,
                        "relative_speed_kmh": relative_speed,
                    },
                )

    def _detect_sign_interactions(
        self,
        *,
        frame_number: int,
        timestamp: float,
        tracked_vehicles: Sequence[Mapping[str, Any]],
        detections: Sequence[Mapping[str, Any]],
    ) -> None:
        sign_detections = [
            det
            for det in detections
            if det.get("class_name", "").lower() in {cls.lower() for cls in self.SIGN_CLASSES}
        ]

        if not sign_detections:
            return

        for sign in sign_detections:
            sign_box = sign.get("box")
            if not sign_box:
                continue
            sign_center = self._box_center(sign_box)

            nearby: List[Dict[str, Any]] = []
            for vehicle in tracked_vehicles:
                track_id = self._track_id(vehicle)
                center = self._object_center(vehicle)
                state = self.track_state.get(track_id, {})
                speed_kmh = state.get("current_speed_kmh")

                if track_id is None or center is None or speed_kmh is None or self.current_scale_m_per_px is None:
                    continue

                distance_px = math.hypot(center[0] - sign_center[0], center[1] - sign_center[1])
                distance_m = distance_px * self.current_scale_m_per_px

                if distance_m <= self.near_sign_distance_m:
                    nearby.append(
                        {
                            "track_id": track_id,
                            "distance_m": distance_m,
                            "speed_kmh": speed_kmh,
                        }
                    )

                    if sign.get("class_name", "").lower().startswith("stop") and speed_kmh > self.stop_speed_kmh:
                        self._record_event(
                            event_type="stop_violation_risk",
                            severity="medium",
                            description="Vehicle near stop sign without slowing down",
                            frame=frame_number,
                            timestamp=timestamp,
                            track_ids=[track_id],
                            metrics={"speed_kmh": speed_kmh, "distance_m": distance_m},
                        )

            if nearby:
                self.sign_interactions.append(
                    {
                        "sign_class": sign.get("class_name"),
                        "frame": frame_number,
                        "timestamp": timestamp,
                        "vehicles": nearby,
                    }
                )

    def _iou_safe_coords(self, box: Iterable[float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(coord) for coord in box]
        return x1, y1, x2, y2

    def _iou(self, box_a: Iterable[float], box_b: Iterable[float]) -> float:
        ax1, ay1, ax2, ay2 = self._iou_safe_coords(box_a)
        bx1, by1, bx2, by2 = self._iou_safe_coords(box_b)

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

    def _vehicle_counts(self, class_counts: Mapping[str, int]) -> Dict[str, int]:
        normalized_vehicle_classes = self.vehicle_class_names()
        vehicle_counts: Dict[str, int] = {}
        for class_name, count in class_counts.items():
            if class_name.lower() in normalized_vehicle_classes:
                vehicle_counts[class_name] = int(count)
        if not vehicle_counts:
            for class_name, count in self.track_vehicle_counts.items():
                vehicle_counts[class_name] = int(count)
        return vehicle_counts

    def _track_duration(self, track: Mapping[str, Any], fps: float) -> Optional[float]:
        first_time = track.get("first_time")
        last_time = track.get("last_time")
        try:
            if first_time is not None and last_time is not None:
                return max(0.0, float(last_time) - float(first_time))
        except (TypeError, ValueError):
            pass

        first_frame = track.get("first_frame")
        last_frame = track.get("last_frame")
        if fps > 0 and first_frame is not None and last_frame is not None:
            try:
                return max(0.0, (int(last_frame) - int(first_frame)) / fps)
            except (TypeError, ValueError):
                return None
        return None

    def _risk_score(self) -> float:
        return min(
            100.0,
            self.severity_counts.get("high", 0) * 35
            + self.severity_counts.get("medium", 0) * 20
            + self.severity_counts.get("low", 0) * 10,
        )

    def _generate_insights(
        self,
        vehicle_summary: Mapping[str, Any],
        compliance_summary: Mapping[str, Any],
    ) -> List[str]:
        insights: List[str] = []
        if vehicle_summary.get("possible_collisions"):
            insights.append("Collision risks detected – review overlapping vehicle tracks with caution.")
        if vehicle_summary.get("high_speed_events"):
            insights.append(
                f"High-speed driving detected {vehicle_summary['high_speed_events']} time(s); investigate speeding behavior."
            )
        if vehicle_summary.get("hard_brake_events"):
            insights.append(
                f"Recorded {vehicle_summary['hard_brake_events']} severe braking event(s), indicating potential unsafe following distances."
            )
        if vehicle_summary.get("lane_shift_events"):
            insights.append(
                f"Detected {vehicle_summary['lane_shift_events']} abrupt lateral movements that may indicate unsafe lane changes."
            )
        if not insights:
            insights.append("No significant unsafe driving patterns detected under current heuristics.")
        return insights
