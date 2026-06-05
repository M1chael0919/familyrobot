from __future__ import annotations

from dataclasses import dataclass

from familyrobot.detection import BoundingBox, PersonDetection, YoloPersonDetector


@dataclass
class FakeBoxes:
    xyxy: list[list[float]]
    conf: list[float]
    cls: list[int]


@dataclass
class FakeResult:
    boxes: FakeBoxes


class FakeModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls = 0

    def predict(self, source, **kwargs):
        self.calls += 1
        self.last_source = source
        self.last_kwargs = kwargs
        return self.results


def test_detector_filters_people_and_confidence() -> None:
    model = FakeModel(
        [
            FakeResult(
                boxes=FakeBoxes(
                    xyxy=[[1, 2, 3, 4], [10, 20, 30, 40], [5, 6, 7, 8]],
                    conf=[0.9, 0.1, 0.8],
                    cls=[0, 0, 1],
                )
            )
        ]
    )

    detector = YoloPersonDetector(
        confidence_threshold=0.25,
        model_loader=lambda path: model,
    )

    detections = detector.detect("frame-1")

    assert model.calls == 1
    assert model.last_source == "frame-1"
    assert model.last_kwargs["conf"] == 0.25
    assert model.last_kwargs["iou"] == 0.45
    assert detections == [
        PersonDetection(
            bbox=BoundingBox(1.0, 2.0, 3.0, 4.0),
            confidence=0.9,
            class_id=0,
        )
    ]


def test_detector_returns_person_detections() -> None:
    model = FakeModel(
        [
            FakeResult(
                boxes=FakeBoxes(
                    xyxy=[[1, 2, 3, 4], [10, 20, 30, 40]],
                    conf=[0.9, 0.8],
                    cls=[0, 0],
                )
            )
        ]
    )

    detector = YoloPersonDetector(model_loader=lambda path: model)

    detections = detector.detect("frame")

    assert len(detections) == 2
    assert detections[0].bbox == BoundingBox(1.0, 2.0, 3.0, 4.0)
    assert detections[0].confidence == 0.9
    assert detections[1].bbox == BoundingBox(10.0, 20.0, 30.0, 40.0)
    assert detections[1].confidence == 0.8
