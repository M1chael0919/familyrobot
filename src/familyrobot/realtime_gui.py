"""Realtime recognition orchestration for camera or video preview."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any

import numpy as np

from familyrobot.capture import CameraVideoInputAdapter
from familyrobot.detection import BoundingBox, PersonDetection, YoloPersonDetector
from familyrobot.enrollment import EnrollmentStore, default_enrollment_root
from familyrobot.face_embedding import FaceEmbeddingExtractor
from familyrobot.greetings import GreetingService
from familyrobot.identity import (
    IdentityMatcher,
    PermanentIdentity,
    UnknownIdentity,
    UnknownPersonHandler,
)
from familyrobot.sample_inputs import resolve_project_path
from familyrobot.speech import QueuedSpeechService
from familyrobot.voice import (
    MicrophoneRecorder,
    VoiceAction,
    VoiceCommand,
    VoiceCommandRouter,
    VoiceConfig,
    VoiceInteractionController,
    VoskTranscriber,
)
from familyrobot.tracking import DeepSortTracker, TrackedPerson


@dataclass(frozen=True, slots=True)
class RealtimeOverlay:
    frame_index: int
    detections: list[PersonDetection] = field(default_factory=list)
    tracks: list[TrackedPerson] = field(default_factory=list)
    identities: dict[int, PermanentIdentity | UnknownIdentity] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    interaction_guess: InteractionGuess | None = None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceTelemetryEvent:
    kind: str
    text: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class InteractionGuess:
    """Mocked speaker/initiator guess for multi-person wake scenarios."""

    identity: PermanentIdentity | None
    confidence: float
    reason: str
    candidate_count: int
    is_mock: bool = True


class VoiceTelemetrySink:
    def __init__(self) -> None:
        self._events: Queue[VoiceTelemetryEvent] = Queue()

    def record_asr(self, command: VoiceCommand, elapsed_seconds: float) -> None:
        self._events.put(
            VoiceTelemetryEvent(
                kind="asr",
                text=command.transcript or "<empty>",
                elapsed_ms=elapsed_seconds * 1000.0,
            )
        )
        print(f"[voice] ASR {elapsed_seconds * 1000.0:.1f} ms | {command.transcript}")

    def record_tts(self, text: str, elapsed_seconds: float) -> None:
        self._events.put(
            VoiceTelemetryEvent(
                kind="tts",
                text=text,
                elapsed_ms=elapsed_seconds * 1000.0,
            )
        )
        print(f"[voice] TTS {elapsed_seconds * 1000.0:.1f} ms | {text}")

    def drain(self) -> list[VoiceTelemetryEvent]:
        events: list[VoiceTelemetryEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                break
        return events


class VoiceListener(Thread):
    def __init__(self, controller: VoiceInteractionController, telemetry: VoiceTelemetrySink) -> None:
        super().__init__(daemon=True, name="familyrobot-voice-listener")
        self._controller = controller
        self._telemetry = telemetry
        self._stop = Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            started = perf_counter()
            command = self._controller.listen_once()
            self._telemetry.record_asr(command, perf_counter() - started)
            if command.action is VoiceAction.EXIT:
                self._stop.set()
                break


@dataclass(slots=True)
class TrackRecognitionState:
    """Cached recognition state for one DeepSORT track."""

    identity: PermanentIdentity | UnknownIdentity | None = None
    last_seen_frame: int = -1
    recognized_at_frame: int = -1
    recognition_streak: int = 0
    last_greeting_frame: int = -1
    last_bbox: BoundingBox | None = None
    last_track_confidence: float = 0.0
    embedding_samples: list[np.ndarray] = field(default_factory=list)


class RealtimeRecognitionSession:
    """Run detection, tracking, and cached identity lookup."""

    def __init__(
        self,
        input_adapter: CameraVideoInputAdapter,
        detector: YoloPersonDetector,
        tracker: DeepSortTracker,
        matcher: IdentityMatcher,
        extractor: FaceEmbeddingExtractor,
        greeting_service: GreetingService | None = None,
        unknown_handler: UnknownPersonHandler | None = None,
        recognition_warmup_frames: int = 3,
        recognition_refresh_interval: int = 30,
        greeting_interval: int = 90,
        interaction_guess_min_identities: int = 2,
    ) -> None:
        self._input_adapter = input_adapter
        self._detector = detector
        self._tracker = tracker
        self._matcher = matcher
        self._extractor = extractor
        self._greeting_service = greeting_service or GreetingService()
        self._unknown_handler = unknown_handler or UnknownPersonHandler()
        self._recognition_warmup_frames = recognition_warmup_frames
        self._recognition_refresh_interval = recognition_refresh_interval
        self._greeting_interval = greeting_interval
        self._interaction_guess_min_identities = interaction_guess_min_identities
        self._track_states: dict[int, TrackRecognitionState] = {}
        self._last_interaction_guess: InteractionGuess | None = None

    @property
    def input_adapter(self) -> CameraVideoInputAdapter:
        return self._input_adapter

    def step(self, frame: Any, frame_index: int) -> tuple[Any, RealtimeOverlay]:
        started_total = perf_counter()
        started_detect = perf_counter()
        detections = self._detector.detect(frame)
        yolo_ms = (perf_counter() - started_detect) * 1000.0

        started_track = perf_counter()
        tracks = self._tracker.track(detections, frame=frame)
        deepsort_ms = (perf_counter() - started_track) * 1000.0
        identities: dict[int, PermanentIdentity | UnknownIdentity] = {}
        labels: dict[int, str] = {}
        identity_ms = 0.0

        active_track_ids = {track.track_id for track in tracks}
        self._expire_missing_tracks(active_track_ids)

        for track in tracks:
            state = self._track_states.setdefault(track.track_id, TrackRecognitionState())
            state.last_seen_frame = frame_index
            state.last_bbox = track.bbox
            state.last_track_confidence = track.confidence

            crop = self._crop(frame, track.bbox)
            should_collect = (
                state.identity is None
                or isinstance(state.identity, UnknownIdentity)
                or (
                    isinstance(state.identity, PermanentIdentity)
                    and state.recognized_at_frame >= 0
                    and (frame_index - state.recognized_at_frame) >= self._recognition_refresh_interval
                )
            )
            if should_collect:
                embedding = self._extractor.extract(crop)
                if embedding is not None:
                    state.embedding_samples.append(embedding.vector)
                    max_samples = max(1, self._recognition_warmup_frames)
                    if len(state.embedding_samples) > max_samples:
                        state.embedding_samples = state.embedding_samples[-max_samples:]

            if self._should_refresh_identity(state, frame_index, len(state.embedding_samples)):
                started_identity = perf_counter()
                resolved = self._resolve_identity_from_samples(state.embedding_samples)
                if isinstance(resolved, UnknownIdentity) and isinstance(state.identity, PermanentIdentity):
                    resolved = state.identity

                previous_identity = state.identity
                state.identity = resolved
                identity_ms += (perf_counter() - started_identity) * 1000.0

                if isinstance(resolved, PermanentIdentity):
                    state.recognized_at_frame = frame_index
                    if previous_identity == resolved:
                        state.recognition_streak += 1
                    else:
                        state.recognition_streak = 1
                else:
                    if previous_identity is None or isinstance(previous_identity, UnknownIdentity):
                        state.recognition_streak = 0
                state.embedding_samples.clear()

            identity = state.identity or UnknownIdentity(reason="pending")
            identities[track.track_id] = identity
            labels[track.track_id] = self._format_label(track, identity)

        interaction_guess = self._guess_interaction_initiator(frame, frame_index)
        self._last_interaction_guess = interaction_guess
        annotated = frame.copy()
        self._annotate_frame(annotated, detections, tracks, labels, interaction_guess)
        return annotated, RealtimeOverlay(
            frame_index=frame_index,
            detections=list(detections),
            tracks=list(tracks),
            identities=identities,
            labels=labels,
            interaction_guess=interaction_guess,
            timings={
                "yolo_ms": yolo_ms,
                "deepsort_ms": deepsort_ms,
                "identity_ms": identity_ms,
                "total_ms": (perf_counter() - started_total) * 1000.0,
            },
        )

    def greeting_for(self, identity: PermanentIdentity | UnknownIdentity | None) -> str | None:
        return self._greeting_service.greeting_for(identity)

    def current_identity(self) -> PermanentIdentity | None:
        candidates: list[tuple[int, int, int, PermanentIdentity]] = []
        for state in self._track_states.values():
            if not isinstance(state.identity, PermanentIdentity):
                continue
            if state.last_seen_frame < 0:
                continue
            candidates.append(
                (
                    state.recognition_streak,
                    state.last_seen_frame,
                    state.recognized_at_frame,
                    state.identity,
                )
            )
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][3]

    def current_interaction_guess(self) -> InteractionGuess | None:
        return self._last_interaction_guess

    def current_interaction_identity(self) -> PermanentIdentity | None:
        guess = self._last_interaction_guess
        if guess is not None and guess.identity is not None:
            return guess.identity
        return self.current_identity()

    def current_interaction_label(self) -> str | None:
        guess = self._last_interaction_guess
        if guess is None:
            return None
        if guess.identity is None:
            return f"发起者推测：未能确定（mock, {guess.confidence:.2f}）"
        return f"发起者推测：{guess.identity.display_name}（mock, {guess.confidence:.2f}）"

    def maybe_greeting_for_track(self, track_id: int, frame_index: int) -> str | None:
        state = self._track_states.get(track_id)
        if state is None or state.identity is None:
            return None
        if isinstance(state.identity, UnknownIdentity):
            return None
        if state.last_greeting_frame >= 0 and frame_index - state.last_greeting_frame < self._greeting_interval:
            return None
        greeting = self._greeting_service.greeting_for(state.identity)
        if greeting is None:
            return None
        state.last_greeting_frame = frame_index
        return greeting

    def _should_refresh_identity(
        self,
        state: TrackRecognitionState,
        frame_index: int,
        sample_count: int,
    ) -> bool:
        if sample_count == 0:
            return False
        if state.identity is None or isinstance(state.identity, UnknownIdentity):
            return sample_count >= self._recognition_warmup_frames
        if state.recognized_at_frame < 0:
            return sample_count >= self._recognition_warmup_frames
        return (frame_index - state.recognized_at_frame) >= self._recognition_refresh_interval

    def _resolve_identity_from_samples(
        self,
        samples: list[np.ndarray],
    ) -> PermanentIdentity | UnknownIdentity:
        if not samples:
            return UnknownIdentity(reason="no_face")

        sample_count = min(len(samples), max(1, self._recognition_warmup_frames))
        stacked = np.stack(samples[-sample_count:], axis=0)
        averaged = np.mean(stacked, axis=0)
        match = self._matcher.match(averaged)
        return self._unknown_handler.resolve(match)

    def _guess_interaction_initiator(self, frame: Any, frame_index: int) -> InteractionGuess | None:
        candidates: list[tuple[float, TrackRecognitionState, int]] = []
        frame_height = 0.0
        frame_width = 0.0
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            frame_height = float(frame.shape[0])
            frame_width = float(frame.shape[1])

        for track_id, state in self._track_states.items():
            if not isinstance(state.identity, PermanentIdentity):
                continue
            if state.last_seen_frame < 0 or state.last_bbox is None:
                continue
            score = self._score_interaction_candidate(state, frame_index, frame_width, frame_height)
            candidates.append((score, state, track_id))

        # 只统计已经解析出长期身份的轨迹，未知轨迹不参与多人 mock。
        if len(candidates) < self._interaction_guess_min_identities:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_state, best_track_id = candidates[0]
        if best_score < 0.35:
            return None

        if len(candidates) == 1:
            reason = "single_person_scene"
        elif best_score - candidates[1][0] < 0.08:
            reason = "multi_person_heuristic"
        else:
            reason = "stable_visual_pick"

        return InteractionGuess(
            identity=best_state.identity if isinstance(best_state.identity, PermanentIdentity) else None,
            confidence=round(min(0.99, best_score), 2),
            reason=f"{reason}:track={best_track_id}",
            candidate_count=len(candidates),
            is_mock=True,
        )

    @staticmethod
    def _score_interaction_candidate(
        state: TrackRecognitionState,
        frame_index: int,
        frame_width: float,
        frame_height: float,
    ) -> float:
        age = max(0, frame_index - state.last_seen_frame)
        recency = max(0.0, 1.0 - min(age, 30) / 30.0)
        stability = min(1.0, state.recognition_streak / 5.0)
        track_quality = max(0.0, min(1.0, state.last_track_confidence))

        bbox = state.last_bbox
        if bbox is None or frame_width <= 0.0 or frame_height <= 0.0:
            center_score = 0.5
            area_score = 0.5
        else:
            width = max(1.0, float(bbox.x2 - bbox.x1))
            height = max(1.0, float(bbox.y2 - bbox.y1))
            area_ratio = (width * height) / max(1.0, frame_width * frame_height)
            area_score = min(1.0, area_ratio / 0.18)

            bbox_center_x = (float(bbox.x1) + float(bbox.x2)) / 2.0
            bbox_center_y = (float(bbox.y1) + float(bbox.y2)) / 2.0
            distance = ((bbox_center_x - frame_width / 2.0) ** 2 + (bbox_center_y - frame_height / 2.0) ** 2) ** 0.5
            max_distance = ((frame_width / 2.0) ** 2 + (frame_height / 2.0) ** 2) ** 0.5
            center_score = 1.0 - min(1.0, distance / max_distance)

        score = (
            0.35 * stability
            + 0.20 * recency
            + 0.20 * track_quality
            + 0.15 * area_score
            + 0.10 * center_score
        )
        return max(0.0, min(1.0, score))

    def _expire_missing_tracks(self, active_track_ids: set[int]) -> None:
        missing_ids = set(self._track_states) - active_track_ids
        for track_id in missing_ids:
            del self._track_states[track_id]

    @staticmethod
    def _crop(frame: Any, bbox: Any) -> Any:
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = max(x1 + 1, int(bbox.x2))
        mid_y = y1 + max(1, int((float(bbox.y2) - float(bbox.y1)) / 2.0))
        y2 = max(y1 + 1, min(int(bbox.y2), mid_y))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _format_label(track: TrackedPerson, identity: PermanentIdentity | UnknownIdentity) -> str:
        if isinstance(identity, UnknownIdentity):
            return f"轨迹 {track.track_id} | 未识别"
        role = f" / {identity.role}" if identity.role else ""
        return f"轨迹 {track.track_id} | {identity.display_name}{role}"

    @staticmethod
    def _annotate_frame(
        frame: Any,
        detections: list[PersonDetection],
        tracks: list[TrackedPerson],
        labels: dict[int, str],
        interaction_guess: InteractionGuess | None,
    ) -> None:
        import cv2

        unicode_labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []

        for detection in detections:
            RealtimeRecognitionSession._draw_box_outline(
                frame,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
                color=(0, 180, 0),
            )
            start = (int(detection.bbox.x1), int(detection.bbox.y1))
            label = f"检测 {detection.confidence:.2f}"
            text_width = max(120, 10 * len(label))
            cv2.rectangle(frame, (start[0], max(0, start[1] - 26)), (start[0] + text_width, start[1]), (0, 180, 0), -1)
            unicode_labels.append((label, (start[0] + 6, max(2, start[1] - 24)), (255, 255, 255)))

        for track in tracks:
            RealtimeRecognitionSession._draw_box_outline(
                frame,
                track.bbox.x1,
                track.bbox.y1,
                track.bbox.x2,
                track.bbox.y2,
                color=(0, 96, 255),
            )
            start = (int(track.bbox.x1), int(track.bbox.y1))
            label = labels.get(track.track_id, f"轨迹 {track.track_id}")
            text_width = max(120, 10 * len(label))
            cv2.rectangle(frame, (start[0], max(0, start[1] - 26)), (start[0] + text_width, start[1]), (0, 96, 255), -1)
            unicode_labels.append((label, (start[0] + 6, max(2, start[1] - 24)), (255, 255, 255)))

        if interaction_guess is not None:
            if interaction_guess.identity is not None:
                guess_text = f"发起者推测：{interaction_guess.identity.display_name}（mock, {interaction_guess.confidence:.2f}）"
            else:
                guess_text = f"发起者推测：未能确定（mock, {interaction_guess.confidence:.2f}）"
            unicode_labels.append((guess_text, (16, 72), (31, 41, 55)))

        if unicode_labels:
            RealtimeRecognitionSession._draw_unicode_texts(frame, unicode_labels)

        RealtimeRecognitionSession._draw_unicode_texts(
            frame,
            [("按 Q 或 Esc 退出", (16, 36), (240, 240, 240))],
        )

    @staticmethod
    def _draw_box_outline(

        frame: Any,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[int, int, int],
    ) -> None:
        import cv2

        start = (int(x1), int(y1))
        end = (int(x2), int(y2))
        cv2.rectangle(frame, start, end, color, 2)

    @staticmethod
    def _draw_unicode_texts(
        frame: Any,
        texts: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
    ) -> None:
        from PIL import Image, ImageDraw
        import cv2

        font = RealtimeRecognitionSession._load_unicode_font(24)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for text, origin, color in texts:
            draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
        frame[:, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_unicode_font(size: int):
        from PIL import ImageFont

        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/arialuni.ttf"),
        ]
        for path in candidates:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    @staticmethod
    def _draw_box(
        frame: Any,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        import cv2

        start = (int(x1), int(y1))
        end = (int(x2), int(y2))
        text_width = max(120, 10 * len(label))
        cv2.rectangle(frame, start, end, color, 2)
        cv2.rectangle(frame, (start[0], max(0, start[1] - 26)), (start[0] + text_width, start[1]), color, -1)
        RealtimeRecognitionSession._draw_unicode_text(
            frame,
            label,
            (start[0] + 6, max(2, start[1] - 24)),
            (255, 255, 255),
        )

    @staticmethod
    def _draw_unicode_text(frame: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        from PIL import Image, ImageDraw, ImageFont
        import cv2

        font = RealtimeRecognitionSession._load_unicode_font(24)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
        frame[:, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_unicode_font(size: int):
        from PIL import ImageFont

        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/arialuni.ttf"),
        ]
        for path in candidates:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()


def build_realtime_session(
    source: int | str | Path = 0,
    model_path: str | Path = "models/yolov8n.pt",
    enrollment_root: str | Path | None = None,
) -> RealtimeRecognitionSession:
    resolved_root = default_enrollment_root() if enrollment_root is None else resolve_project_path(enrollment_root)
    store = EnrollmentStore(resolved_root)
    extractor = FaceEmbeddingExtractor()
    matcher = IdentityMatcher.from_enrollment(store, extractor)
    return RealtimeRecognitionSession(
        input_adapter=CameraVideoInputAdapter(source=source),
        detector=YoloPersonDetector(model_path=resolve_project_path(model_path)),
        tracker=DeepSortTracker(),
        matcher=matcher,
        extractor=extractor,
    )
