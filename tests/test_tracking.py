from __future__ import annotations

from dataclasses import dataclass

from familyrobot.detection import BoundingBox, PersonDetection
from familyrobot.tracking import DeepSortTracker, TrackedPerson


@dataclass
class FakeTrack:
    track_id: int
    bbox: list[float]
    confidence: float = 0.88
    class_id: int = 0
    class_name: str = "person"
    confirmed: bool = True

    def is_confirmed(self) -> bool:
        return self.confirmed


class FakeTracker:
    def __init__(self, tracks: list[FakeTrack]) -> None:
        self.tracks = tracks
        self.calls = 0

    def update_tracks(self, detections, frame=None):
        self.calls += 1
        self.last_detections = detections
        self.last_frame = frame
        return self.tracks


def test_tracker_converts_detections_and_returns_tracked_people() -> None:
    tracker = FakeTracker(
        [
            FakeTrack(track_id=7, bbox=[1, 2, 3, 4]),
            FakeTrack(track_id=8, bbox=[10, 20, 30, 40], confirmed=False),
        ]
    )

    wrapper = DeepSortTracker(tracker_loader=lambda path: tracker)
    detections = [
        PersonDetection(
            bbox=BoundingBox(1, 2, 3, 4),
            confidence=0.9,
        )
    ]

    tracked = wrapper.track(detections, frame="frame-1")

    assert tracker.calls == 1
    assert tracker.last_frame == "frame-1"
    assert tracked == [
        TrackedPerson(
            track_id=7,
            bbox=BoundingBox(1.0, 2.0, 3.0, 4.0),
            confidence=0.88,
            class_id=0,
            class_name="person",
        )
    ]
    assert len(tracker.last_detections) == 1
    assert tracker.last_detections[0] == ([1.0, 2.0, 2.0, 2.0], 0.9, "person")


def test_tracker_skips_tracks_without_ids_or_boxes() -> None:
    class BrokenTrack:
        def is_confirmed(self) -> bool:
            return True

    tracker = FakeTracker([BrokenTrack()])
    wrapper = DeepSortTracker(tracker_loader=lambda path: tracker)

    tracked = wrapper.track([])

    assert tracked == []
