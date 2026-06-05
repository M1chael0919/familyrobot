from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from familyrobot.detection import BoundingBox, PersonDetection
from familyrobot.face_embedding import FaceEmbedding
from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity
from familyrobot.realtime_gui import RealtimeRecognitionSession
from familyrobot.tracking import TrackedPerson


@dataclass
class _FakeDetector:
    detections: list[PersonDetection]

    def detect(self, frame):  # noqa: ANN001
        return list(self.detections)


@dataclass
class _FakeTracker:
    tracks: list[TrackedPerson]

    def track(self, detections, frame=None):  # noqa: ANN001
        return list(self.tracks)


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image):  # noqa: ANN001
        self.calls += 1
        return FaceEmbedding(vector=np.array([1.0, 0.0], dtype=np.float32))


class _SequenceExtractor:
    def __init__(self, vectors: list[np.ndarray]) -> None:
        self._vectors = vectors
        self.calls = 0

    def extract(self, image):  # noqa: ANN001
        self.calls += 1
        index = min(self.calls - 1, len(self._vectors) - 1)
        return FaceEmbedding(vector=self._vectors[index])


class _FakeMatcher:
    def __init__(self, identity: PermanentIdentity) -> None:
        self._identity = identity
        self.vectors: list[np.ndarray] = []

    def match(self, vector):  # noqa: ANN001
        self.vectors.append(np.asarray(vector, dtype=np.float32))
        from familyrobot.identity import IdentityMatch

        return IdentityMatch(identity=self._identity, score=0.99, sample_count=2)


class _SequenceMatcher:
    def __init__(self, identities: list[PermanentIdentity]) -> None:
        self._identities = identities
        self.calls = 0

    def match(self, vector):  # noqa: ANN001
        from familyrobot.identity import IdentityMatch

        index = min(self.calls, len(self._identities) - 1)
        self.calls += 1
        return IdentityMatch(identity=self._identities[index], score=0.95, sample_count=2)


class _FakeInputAdapter:
    def open(self):
        return self

    def read_frame(self):
        return None

    def release(self):
        return None


def test_realtime_session_builds_identity_labels() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = PersonDetection(
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    track = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    identity = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection]),
        tracker=_FakeTracker([track]),
        matcher=_FakeMatcher(identity),
        extractor=_FakeExtractor(),
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
    )

    annotated, overlay = session.step(frame, 0)

    assert annotated.shape == frame.shape
    assert overlay.labels[7] == "轨迹 7 | 爸爸 / 父亲"
    assert overlay.identities[7] == identity


def test_realtime_session_caches_identity_for_same_track() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = PersonDetection(
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    track = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    identity = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")
    extractor = _FakeExtractor()

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection]),
        tracker=_FakeTracker([track]),
        matcher=_FakeMatcher(identity),
        extractor=extractor,
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
        recognition_refresh_interval=100,
    )

    session.step(frame, 0)
    session.step(frame, 1)

    assert extractor.calls == 1


def test_realtime_session_emits_greeting_once_per_track_window() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = PersonDetection(
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    track = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    identity = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection]),
        tracker=_FakeTracker([track]),
        matcher=_FakeMatcher(identity),
        extractor=_FakeExtractor(),
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
        greeting_interval=90,
    )

    session.step(frame, 0)
    first = session.maybe_greeting_for_track(7, 0)
    second = session.maybe_greeting_for_track(7, 1)

    assert first == "爸爸今天工作怎么样？"
    assert second is None


def test_realtime_session_returns_current_identity() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = PersonDetection(
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    track = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    identity = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection]),
        tracker=_FakeTracker([track]),
        matcher=_FakeMatcher(identity),
        extractor=_FakeExtractor(),
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
    )

    session.step(frame, 0)

    assert session.current_identity() == identity


def test_realtime_session_crops_upper_half_for_identity_lookup() -> None:
    frame = np.zeros((200, 120, 3), dtype=np.uint8)
    bbox = BoundingBox(10, 20, 70, 180)

    cropped = RealtimeRecognitionSession._crop(frame, bbox)

    assert cropped.shape == (80, 60, 3)


def test_realtime_session_averages_multiple_embeddings_before_matching() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = PersonDetection(
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    track = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(10, 20, 100, 180),
        confidence=0.91,
    )
    identity = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")
    extractor = _SequenceExtractor(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    matcher = _FakeMatcher(identity)

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection]),
        tracker=_FakeTracker([track]),
        matcher=matcher,
        extractor=extractor,
        greeting_service=GreetingService(),
        recognition_warmup_frames=2,
        recognition_refresh_interval=100,
    )

    session.step(frame, 0)
    session.step(frame, 1)

    assert extractor.calls == 2
    assert matcher.vectors
    assert np.allclose(matcher.vectors[0], np.array([0.5, 0.5], dtype=np.float32))


def test_realtime_session_mocks_multi_person_interaction_initiator() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection_a = PersonDetection(
        bbox=BoundingBox(62, 24, 146, 174),
        confidence=0.91,
    )
    detection_b = PersonDetection(
        bbox=BoundingBox(8, 44, 68, 154),
        confidence=0.88,
    )
    detection_c = PersonDetection(
        bbox=BoundingBox(150, 36, 194, 148),
        confidence=0.86,
    )
    track_a = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(62, 24, 146, 174),
        confidence=0.91,
    )
    track_b = TrackedPerson(
        track_id=8,
        bbox=BoundingBox(8, 44, 68, 154),
        confidence=0.88,
    )
    track_c = TrackedPerson(
        track_id=9,
        bbox=BoundingBox(150, 36, 194, 148),
        confidence=0.86,
    )
    father = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")
    mother = PermanentIdentity(identity_id="mother", display_name="妈妈", role="母亲")
    child = PermanentIdentity(identity_id="child", display_name="孩子", role="子女")
    extractor = _SequenceExtractor(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
        ]
    )

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection_a, detection_b, detection_c]),
        tracker=_FakeTracker([track_a, track_b, track_c]),
        matcher=_SequenceMatcher([father, mother, child]),
        extractor=extractor,
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
    )

    _, overlay = session.step(frame, 0)

    assert overlay.interaction_guess is not None
    assert overlay.interaction_guess.identity == father
    assert overlay.interaction_guess.is_mock is True
    assert overlay.interaction_guess.candidate_count == 3
    assert session.current_interaction_identity() == father


def test_realtime_session_mocks_two_person_scene() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection_a = PersonDetection(
        bbox=BoundingBox(62, 24, 146, 174),
        confidence=0.91,
    )
    detection_b = PersonDetection(
        bbox=BoundingBox(8, 44, 68, 154),
        confidence=0.88,
    )
    track_a = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(62, 24, 146, 174),
        confidence=0.91,
    )
    track_b = TrackedPerson(
        track_id=8,
        bbox=BoundingBox(8, 44, 68, 154),
        confidence=0.88,
    )
    father = PermanentIdentity(identity_id="father", display_name="爸爸", role="父亲")
    mother = PermanentIdentity(identity_id="mother", display_name="妈妈", role="母亲")

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection_a, detection_b]),
        tracker=_FakeTracker([track_a, track_b]),
        matcher=_SequenceMatcher([father, mother]),
        extractor=_SequenceExtractor(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        ),
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
    )

    _, overlay = session.step(frame, 0)

    assert overlay.interaction_guess is not None
    assert overlay.interaction_guess.identity == father
    assert overlay.interaction_guess.is_mock is True
    assert overlay.interaction_guess.candidate_count == 2


def test_realtime_session_does_not_mock_single_person_scene() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = PersonDetection(
        bbox=BoundingBox(62, 24, 146, 174),
        confidence=0.91,
    )
    track = TrackedPerson(
        track_id=7,
        bbox=BoundingBox(62, 24, 146, 174),
        confidence=0.91,
    )
    father = PermanentIdentity(identity_id="father", display_name="鐖哥埜", role="鐖朵翰")

    session = RealtimeRecognitionSession(
        input_adapter=_FakeInputAdapter(),
        detector=_FakeDetector([detection]),
        tracker=_FakeTracker([track]),
        matcher=_SequenceMatcher([father]),
        extractor=_SequenceExtractor([np.array([1.0, 0.0], dtype=np.float32)]),
        greeting_service=GreetingService(),
        recognition_warmup_frames=1,
    )

    _, overlay = session.step(frame, 0)

    assert overlay.interaction_guess is None
