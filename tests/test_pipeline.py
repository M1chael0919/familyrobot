from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

from familyrobot.capture import CameraVideoInputAdapter
from familyrobot.detection import BoundingBox, PersonDetection
from familyrobot.pipeline import FamilyRobotPipeline
from familyrobot.tracking import TrackedPerson


class FakeCapture:
    def __init__(self) -> None:
        self.opened = True
        self.calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        self.calls += 1
        if self.calls == 1:
            return True, "frame-1"
        return False, None

    def release(self) -> None:
        self.opened = False


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        self.last_frame = frame
        return [
            PersonDetection(
                bbox=BoundingBox(1, 2, 3, 4),
                confidence=0.9,
            )
        ]


class FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def track(self, detections, frame=None):
        self.calls += 1
        self.last_detections = detections
        self.last_frame = frame
        return [
            TrackedPerson(
                track_id=7,
                bbox=BoundingBox(1, 2, 3, 4),
                confidence=0.9,
            )
        ]


def test_pipeline_runs_one_frame_and_writes_summary() -> None:
    output = StringIO()
    adapter = CameraVideoInputAdapter(capture_factory=lambda source, backend: FakeCapture())
    detector = FakeDetector()
    tracker = FakeTracker()

    pipeline = FamilyRobotPipeline(
        input_adapter=adapter,
        detector=detector,
        tracker=tracker,
        output=output,
    )

    result = pipeline.run(max_frames=1)

    assert len(result.frames) == 1
    assert detector.calls == 1
    assert tracker.calls == 1
    assert "frame=0 detections=1 tracks=1" in output.getvalue()
