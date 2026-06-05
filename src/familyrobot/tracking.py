"""DeepSORT tracking wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch

from familyrobot.detection import BoundingBox, PersonDetection


@dataclass(frozen=True)
class TrackedPerson:
    """A tracked detection with a temporary track ID."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int = 0
    class_name: str = "person"


class DeepSORTLike(Protocol):
    """Minimal interface for a DeepSORT tracker."""

    def update_tracks(self, detections: Sequence[Any], frame: Any | None = None) -> Sequence[Any]: ...


TrackerLoader = Callable[[str | Path | None], DeepSORTLike]


class DeepSortTracker:
    """Thin wrapper around a DeepSORT tracker."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        tracker_loader: TrackerLoader | None = None,
    ) -> None:
        self._model_path = model_path
        self._tracker_loader = tracker_loader or self._load_deepsort_tracker
        self._tracker: DeepSORTLike | None = None

    def load(self) -> "DeepSortTracker":
        """Load the tracker if needed."""

        if self._tracker is None:
            self._tracker = self._tracker_loader(self._model_path)
        return self

    def track(self, detections: Sequence[PersonDetection], frame: Any | None = None) -> list[TrackedPerson]:
        """Update tracks from the current detections."""

        if self._tracker is None:
            self.load()

        assert self._tracker is not None
        prepared_detections = [self._to_raw_detection(detection) for detection in detections]
        tracks = self._tracker.update_tracks(prepared_detections, frame=frame)
        return self._parse_tracks(tracks)

    @staticmethod
    def _to_raw_detection(detection: PersonDetection) -> tuple[list[float], float, str]:
        bbox = detection.bbox
        ltwh = [
            float(bbox.x1),
            float(bbox.y1),
            float(bbox.x2 - bbox.x1),
            float(bbox.y2 - bbox.y1),
        ]
        return ltwh, float(detection.confidence), detection.class_name

    def _parse_tracks(self, tracks: Sequence[Any]) -> list[TrackedPerson]:
        parsed: list[TrackedPerson] = []
        for track in tracks:
            parsed_track = self._parse_track(track)
            if parsed_track is not None:
                parsed.append(parsed_track)
        return parsed

    def _parse_track(self, track: Any) -> TrackedPerson | None:
        if hasattr(track, "is_confirmed") and not track.is_confirmed():
            return None

        track_id = getattr(track, "track_id", None)
        if track_id is None:
            track_id = getattr(track, "trackid", None)
        if track_id is None:
            return None

        bbox = self._extract_bbox(track)
        if bbox is None:
            return None

        confidence = float(
            getattr(track, "det_conf", None)
            if getattr(track, "det_conf", None) is not None
            else getattr(track, "confidence", 0.0)
        )
        class_id = int(getattr(track, "class_id", 0))
        class_name = str(
            getattr(track, "det_class", None)
            if getattr(track, "det_class", None) is not None
            else getattr(track, "class_name", "person")
        )

        return TrackedPerson(
            track_id=int(track_id),
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
        )

    @staticmethod
    def _extract_bbox(track: Any) -> BoundingBox | None:
        bbox = getattr(track, "bbox", None)
        if bbox is not None:
            return DeepSortTracker._to_bbox(bbox)

        to_ltrb = getattr(track, "to_ltrb", None)
        if callable(to_ltrb):
            return DeepSortTracker._to_bbox(to_ltrb())

        to_tlbr = getattr(track, "to_tlbr", None)
        if callable(to_tlbr):
            return DeepSortTracker._to_bbox(to_tlbr())

        return None

    @staticmethod
    def _to_bbox(value: Any) -> BoundingBox:
        if hasattr(value, "tolist"):
            value = value.tolist()
        x1, y1, x2, y2 = value
        return BoundingBox(float(x1), float(y1), float(x2), float(y2))

    @staticmethod
    def _load_deepsort_tracker(model_path: str | Path | None) -> DeepSORTLike:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "deep-sort-realtime is required to load the DeepSORT tracker."
            ) from exc

        return DeepSort(embedder_gpu=torch.cuda.is_available(), half=torch.cuda.is_available())
