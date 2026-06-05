"""YOLO person detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch


Frame = Any


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class PersonDetection:
    """A single person detection."""

    bbox: BoundingBox
    confidence: float
    class_id: int = 0
    class_name: str = "person"


class YOLOModelLike(Protocol):
    """Minimal interface for a YOLO model."""

    def predict(self, source: Frame, **kwargs: Any) -> Sequence[Any]: ...


ModelLoader = Callable[[str | Path], YOLOModelLike]


class YoloPersonDetector:
    """Thin wrapper around a YOLO model for person detection."""

    def __init__(
        self,
        model_path: str | Path = "models/yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        model_loader: ModelLoader | None = None,
        person_class_ids: set[int] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._model_loader = model_loader or self._load_ultralytics_model
        self._person_class_ids = person_class_ids or {0}
        self._model: YOLOModelLike | None = None

    def load(self) -> "YoloPersonDetector":
        """Load the YOLO model if needed."""

        if self._model is None:
            self._model = self._model_loader(self._model_path)
        return self

    def detect(self, frame: Frame) -> list[PersonDetection]:
        """Detect people in a frame."""

        if self._model is None:
            self.load()

        assert self._model is not None
        results = self._model.predict(
            frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu",
        )
        return self._parse_results(results)

    def _parse_results(self, results: Sequence[Any]) -> list[PersonDetection]:
        detections: list[PersonDetection] = []
        for result in results:
            detections.extend(self._parse_result(result))
        return detections

    def _parse_result(self, result: Any) -> list[PersonDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = self._to_rows(getattr(boxes, "xyxy", None))
        confidences = self._to_values(getattr(boxes, "conf", None))
        class_ids = self._to_int_values(getattr(boxes, "cls", None))

        detections: list[PersonDetection] = []
        for index, coords in enumerate(xyxy):
            class_id = class_ids[index] if index < len(class_ids) else 0
            if class_id not in self._person_class_ids:
                continue

            confidence = confidences[index] if index < len(confidences) else 0.0
            if confidence < self._confidence_threshold:
                continue

            bbox = BoundingBox(*coords)
            detections.append(
                PersonDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                )
            )
        return detections

    @staticmethod
    def _to_rows(value: Any) -> list[list[float]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            rows = value.tolist()
        else:
            rows = list(value)
        return [[float(item) for item in row] for row in rows]

    @staticmethod
    def _to_values(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            values = value.tolist()
        else:
            values = list(value)
        return [float(item) for item in values]

    @staticmethod
    def _to_int_values(value: Any) -> list[int]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            values = value.tolist()
        else:
            values = list(value)
        return [int(item) for item in values]

    @staticmethod
    def _load_ultralytics_model(model_path: str | Path) -> YOLOModelLike:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Ultralytics is required to load the YOLO detector."
            ) from exc

        return YOLO(str(model_path))
