"""Identity mapping model for track-to-person association."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np

from familyrobot.enrollment import EnrollmentManifest, EnrollmentRecord, EnrollmentStore
from familyrobot.face_embedding import FaceEmbedding, FaceEmbeddingExtractor, load_image


class IdentityLinkState(str, Enum):
    """Lifecycle state of a track-to-identity link."""

    UNKNOWN = "unknown"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class PermanentIdentity:
    """Stable identity for an enrolled family member."""

    identity_id: str
    display_name: str
    role: str | None = None
    enrolled_at: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackIdentityLink:
    """Temporary association between a DeepSORT track and a permanent identity."""

    track_id: int
    identity_id: str | None = None
    state: IdentityLinkState = IdentityLinkState.UNKNOWN
    confidence: float = 0.0
    first_seen_frame: int | None = None
    last_seen_frame: int | None = None
    confirmed_at_frame: int | None = None


@dataclass(frozen=True, slots=True)
class IdentityAssignment:
    """A resolved identity assignment for a tracked person."""

    track_id: int
    identity: PermanentIdentity | None = None
    link: TrackIdentityLink | None = None
    assigned_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class IdentityTemplate:
    """Normalized embedding template for one permanent identity."""

    identity: PermanentIdentity
    embedding: np.ndarray
    sample_count: int = 1


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    """Best identity match for an input embedding."""

    identity: PermanentIdentity
    score: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class UnknownIdentity:
    """Sentinel result used when a person cannot be matched confidently."""

    reason: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ReIDEvent:
    """Result of a re-identification update."""

    track_id: int
    identity: PermanentIdentity
    score: float
    restored_from_lost: bool
    frame_index: int


@dataclass(slots=True)
class IdentityTrackState:
    """Mutable state for a confirmed or lost identity."""

    identity: PermanentIdentity
    track_id: int | None = None
    last_seen_frame: int | None = None
    lost_since_frame: int | None = None

    def mark_seen(self, track_id: int, frame_index: int) -> None:
        self.track_id = track_id
        self.last_seen_frame = frame_index
        self.lost_since_frame = None

    def mark_lost(self, frame_index: int) -> None:
        self.track_id = None
        self.lost_since_frame = frame_index
        self.last_seen_frame = frame_index


class IdentityMatcher:
    """Cosine-similarity matcher over enrolled identity templates."""

    def __init__(self, templates: Sequence[IdentityTemplate]) -> None:
        self._templates = list(templates)

    @property
    def templates(self) -> list[IdentityTemplate]:
        return list(self._templates)

    @classmethod
    def from_enrollment(
        cls,
        store: EnrollmentStore,
        extractor: FaceEmbeddingExtractor,
    ) -> "IdentityMatcher":
        manifest = store.load()
        templates: list[IdentityTemplate] = []
        for record in manifest.records:
            template = cls._build_template(store.root, record, extractor)
            if template is not None:
                templates.append(template)
        return cls(templates)

    @staticmethod
    def _build_template(
        root: Path,
        record: EnrollmentRecord,
        extractor: FaceEmbeddingExtractor,
    ) -> IdentityTemplate | None:
        vectors: list[np.ndarray] = []
        for sample in record.sample_images:
            sample_path = root / sample
            embedding = IdentityMatcher._extract_embedding(extractor, sample_path)
            if embedding is not None:
                vectors.append(embedding.vector)

        if not vectors:
            return None

        stacked = np.stack(vectors, axis=0)
        averaged = np.mean(stacked, axis=0)
        identity = PermanentIdentity(
            identity_id=record.identity_id,
            display_name=record.display_name,
            role=record.role,
            enrolled_at=record.enrolled_at,
            metadata=record.metadata,
        )
        return IdentityTemplate(
            identity=identity,
            embedding=IdentityMatcher._normalize(averaged),
            sample_count=len(vectors),
        )

    def match(self, vector: np.ndarray) -> IdentityMatch | None:
        if not self._templates:
            return None

        query = self._normalize(vector)
        best_template: IdentityTemplate | None = None
        best_score = -1.0

        for template in self._templates:
            score = self._cosine_similarity(query, template.embedding)
            if score > best_score:
                best_template = template
                best_score = score

        if best_template is None:
            return None

        return IdentityMatch(
            identity=best_template.identity,
            score=best_score,
            sample_count=best_template.sample_count,
        )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            return values
        return values / norm

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_values = IdentityMatcher._normalize(left)
        right_values = IdentityMatcher._normalize(right)
        if left_values.size == 0 or right_values.size == 0:
            return 0.0
        return float(np.dot(left_values, right_values))

    @staticmethod
    def _extract_embedding(
        extractor: FaceEmbeddingExtractor,
        sample_path: Path,
    ) -> FaceEmbedding | None:
        if not sample_path.exists():
            return None

        try:
            image = load_image(sample_path)
        except Exception:
            return None
        return extractor.extract(image)


@dataclass(frozen=True, slots=True)
class UnknownPersonPolicy:
    """Decision policy for unknown people."""

    match_threshold: float = 0.65
    min_sample_count: int = 1


class UnknownPersonHandler:
    """Return an explicit unknown result when the match is not confident enough."""

    def __init__(self, policy: UnknownPersonPolicy | None = None) -> None:
        self._policy = policy or UnknownPersonPolicy()

    @property
    def policy(self) -> UnknownPersonPolicy:
        return self._policy

    def resolve(self, match: IdentityMatch | None) -> PermanentIdentity | UnknownIdentity:
        if match is None:
            return UnknownIdentity(reason="no_match")

        if match.score < self._policy.match_threshold:
            return UnknownIdentity(reason="low_confidence", score=match.score)

        if match.sample_count < self._policy.min_sample_count:
            return UnknownIdentity(reason="insufficient_gallery_samples", score=match.score)

        return match.identity


class IdentityReidentifier:
    """Restore a permanent identity after a track disappears and returns."""

    def __init__(
        self,
        matcher: IdentityMatcher,
        match_threshold: float = 0.65,
        disappearance_window: int = 45,
    ) -> None:
        self._matcher = matcher
        self._match_threshold = match_threshold
        self._disappearance_window = disappearance_window
        self._states: dict[str, IdentityTrackState] = {}
        self._track_to_identity: dict[int, str] = {}

    @property
    def states(self) -> dict[str, IdentityTrackState]:
        return dict(self._states)

    def confirm(self, track_id: int, vector: np.ndarray, frame_index: int) -> IdentityMatch | None:
        match = self._matcher.match(vector)
        if match is None or match.score < self._match_threshold:
            return None

        state = self._states.get(match.identity.identity_id)
        if state is None:
            state = IdentityTrackState(identity=match.identity)
            self._states[match.identity.identity_id] = state

        state.mark_seen(track_id, frame_index)
        self._track_to_identity[track_id] = match.identity.identity_id
        return match

    def mark_lost(self, track_id: int, frame_index: int) -> None:
        identity_id = self._track_to_identity.pop(track_id, None)
        if identity_id is None:
            return

        state = self._states.get(identity_id)
        if state is None:
            return
        state.mark_lost(frame_index)

    def reidentify(
        self,
        track_id: int,
        vector: np.ndarray,
        frame_index: int,
    ) -> ReIDEvent | None:
        match = self._matcher.match(vector)
        if match is None or match.score < self._match_threshold:
            return None

        state = self._states.get(match.identity.identity_id)
        restored_from_lost = False
        if state is not None and state.lost_since_frame is not None:
            restored_from_lost = (
                frame_index - state.lost_since_frame <= self._disappearance_window
            )
        elif state is None:
            state = IdentityTrackState(identity=match.identity)
            self._states[match.identity.identity_id] = state

        if state is None:
            return None

        state.mark_seen(track_id, frame_index)
        self._track_to_identity[track_id] = match.identity.identity_id
        return ReIDEvent(
            track_id=track_id,
            identity=match.identity,
            score=match.score,
            restored_from_lost=restored_from_lost,
            frame_index=frame_index,
        )
