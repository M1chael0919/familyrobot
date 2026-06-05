"""Video audio extraction and frame-time alignment helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from familyrobot.identity import PermanentIdentity
from familyrobot.voice import VoiceCommand, VoiceCommandRouter, VoiceTranscriptSegment, VoskTranscriber


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic video timing metadata."""

    video_path: Path
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float | None:
        if self.fps <= 0 or self.frame_count <= 0:
            return None
        return self.frame_count / self.fps

    def timestamp_for_frame(self, frame_index: int) -> float:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.fps <= 0:
            return float(frame_index)
        return frame_index / self.fps

    def frame_for_timestamp(self, timestamp_seconds: float) -> int:
        if timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if self.fps <= 0:
            return int(timestamp_seconds)
        return int(timestamp_seconds * self.fps)

    def frame_window_for_timestamp(self, timestamp_seconds: float) -> tuple[int, int]:
        frame_index = self.frame_for_timestamp(timestamp_seconds)
        return frame_index, min(frame_index + 1, max(self.frame_count, frame_index + 1))


@dataclass(frozen=True, slots=True)
class RecordedVideoAlignment:
    """Audio and frame alignment for a recorded video file."""

    metadata: VideoMetadata
    audio_path: Path

    def timestamp_for_frame(self, frame_index: int) -> float:
        return self.metadata.timestamp_for_frame(frame_index)

    def frame_for_timestamp(self, timestamp_seconds: float) -> int:
        return self.metadata.frame_for_timestamp(timestamp_seconds)

    def frame_window_for_timestamp(self, timestamp_seconds: float) -> tuple[int, int]:
        return self.metadata.frame_window_for_timestamp(timestamp_seconds)


@dataclass(frozen=True, slots=True)
class VideoWakeEvent:
    """A wake-word event aligned to a video timestamp."""

    transcript: str
    timestamp_seconds: float
    frame_index: int
    segment: VoiceTranscriptSegment
    command: VoiceCommand | None = None


class VideoAudioExtractor:
    """Extract a mono PCM WAV audio track from a local video file."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        output_dir: str | Path | None = None,
        ffmpeg_executable: str | Path | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._ffmpeg_executable = ffmpeg_executable

    def extract(self, video_path: str | Path, output_dir: str | Path | None = None) -> Path:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        destination_dir = Path(output_dir) if output_dir is not None else self._output_dir
        if destination_dir is None:
            destination_dir = video_path.parent / f"{video_path.stem}.audio"
        destination_dir.mkdir(parents=True, exist_ok=True)

        audio_path = destination_dir / f"{video_path.stem}.wav"
        command = [
            self._resolve_ffmpeg_executable(),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(self._channels),
            "-ar",
            str(self._sample_rate),
            "-acodec",
            "pcm_s16le",
            str(audio_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path

    def _resolve_ffmpeg_executable(self) -> str:
        if self._ffmpeg_executable is not None:
            return str(self._ffmpeg_executable)

        try:
            import imageio_ffmpeg
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "imageio-ffmpeg is required to extract audio from video files."
            ) from exc

        return imageio_ffmpeg.get_ffmpeg_exe()


class WavAudioPlayer:
    """Play a WAV file asynchronously through the local audio device."""

    def __init__(self) -> None:
        self._thread = None
        self._stop_requested = False

    def start(self, wav_path: str | Path) -> None:
        if self._thread is not None:
            return

        from threading import Thread

        self._stop_requested = False
        self._thread = Thread(target=self._run, args=(Path(wav_path),), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested = True
        try:
            import sounddevice as sd
        except ImportError:  # pragma: no cover - depends on environment
            sd = None
        if sd is not None:
            sd.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self, wav_path: Path) -> None:
        from scipy.io import wavfile

        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on environment
            print("[video-audio] playback failed: sounddevice is required")
            raise RuntimeError("sounddevice is required to play extracted video audio.") from exc

        try:
            sample_rate, samples = wavfile.read(str(wav_path))
            array = np.asarray(samples)
            if array.ndim > 1:
                array = array.mean(axis=1)
            if array.dtype != np.float32:
                if np.issubdtype(array.dtype, np.integer):
                    info = np.iinfo(array.dtype)
                    max_abs = max(abs(info.min), abs(info.max)) or 1
                    array = array.astype(np.float32) / float(max_abs)
                else:
                    array = array.astype(np.float32)
            print(f"[video-audio] playing {wav_path} @ {sample_rate}Hz")
            sd.play(array, sample_rate)
            sd.wait()
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            print(f"[video-audio] playback failed: {exc}")
            raise
        finally:
            self._thread = None


class VideoWakeEventFinder:
    """Find wake-word transcript segments in extracted video audio."""

    def __init__(
        self,
        *,
        transcriber: VoskTranscriber,
        wake_phrase: str,
    ) -> None:
        self._transcriber = transcriber
        self._wake_phrase = self._normalize(wake_phrase)

    def find_events(self, audio_path: str | Path, alignment: RecordedVideoAlignment) -> list[VideoWakeEvent]:
        events: list[VideoWakeEvent] = []
        for segment in self._transcriber.transcribe_wav_segments(audio_path):
            normalized = self._normalize(segment.text)
            if self._wake_phrase not in normalized:
                continue
            timestamp_seconds = segment.start_seconds
            events.append(
                VideoWakeEvent(
                    transcript=segment.text,
                    timestamp_seconds=timestamp_seconds,
                    frame_index=alignment.frame_for_timestamp(timestamp_seconds),
                    segment=segment,
                )
            )
        return events

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(text.split()).lower()


class VideoWakeEventRouter:
    """Route aligned video wake events through the normal voice engine."""

    def __init__(
        self,
        *,
        voice_controller,
        identity_provider: Callable[[], PermanentIdentity | None] | None = None,
    ) -> None:
        self._voice_controller = voice_controller
        self._identity_provider = identity_provider

    def dispatch(self, event: VideoWakeEvent) -> VoiceCommand:
        identity_context = self._identity_provider() if self._identity_provider is not None else None
        command = self._voice_controller.process_transcript(
            event.transcript,
            identity_context=identity_context,
        )
        return VoiceCommand(
            transcript=command.transcript,
            activated=command.activated,
            action=command.action,
            response_text=command.response_text,
            target_identity_id=command.target_identity_id,
            target_display_name=command.target_display_name,
        )


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read frame timing metadata from a local video file."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required to inspect video metadata.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoMetadata(video_path=video_path, fps=fps, frame_count=frame_count)
    finally:
        capture.release()


def build_recorded_video_alignment(
    video_path: str | Path,
    *,
    extractor: VideoAudioExtractor | None = None,
    output_dir: str | Path | None = None,
) -> RecordedVideoAlignment:
    """Extract audio and return frame-time alignment for a recorded video."""

    metadata = read_video_metadata(video_path)
    audio_extractor = extractor or VideoAudioExtractor()
    audio_path = audio_extractor.extract(video_path, output_dir=output_dir)
    return RecordedVideoAlignment(metadata=metadata, audio_path=audio_path)


def iter_video_wake_events(
    video_path: str | Path,
    *,
    wake_phrase: str,
    transcriber: VoskTranscriber,
    extractor: VideoAudioExtractor | None = None,
    output_dir: str | Path | None = None,
) -> tuple[RecordedVideoAlignment, list[VideoWakeEvent]]:
    """Extract audio and find wake events for a recorded video."""

    alignment = build_recorded_video_alignment(video_path, extractor=extractor, output_dir=output_dir)
    finder = VideoWakeEventFinder(transcriber=transcriber, wake_phrase=wake_phrase)
    events = finder.find_events(alignment.audio_path, alignment)
    return alignment, events
