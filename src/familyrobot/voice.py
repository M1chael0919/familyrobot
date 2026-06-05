"""Voice interaction helpers for wake phrase, ASR, and command routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity
from familyrobot.speech import QueuedSpeechService


class RecognizerLike(Protocol):
    """Minimal interface for a Vosk recognizer."""

    def AcceptWaveform(self, data: bytes) -> bool: ...  # noqa: N802

    def Result(self) -> str: ...  # noqa: N802

    def FinalResult(self) -> str: ...  # noqa: N802


ModelLoader = Callable[[str | Path], object]
RecognizerFactory = Callable[[object, int], RecognizerLike]


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Runtime configuration for voice interaction."""

    model_path: Path = Path("models/vosk-model-small-cn-0.22")
    sample_rate: int = 16000
    record_seconds: float = 4.0
    wake_phrase: str = "你好"
    silence_threshold: float = 0.01
    language_hint: str = "zh-CN"


@dataclass(frozen=True, slots=True)
class VoiceTranscript:
    """Transcribed speech text."""

    text: str
    raw_json: str
    is_final: bool = True


@dataclass(frozen=True, slots=True)
class VoiceTranscriptSegment:
    """Timestamped transcript segment from streaming ASR."""

    text: str
    raw_json: str
    start_seconds: float
    end_seconds: float
    is_final: bool = True


class VoiceAction(str, Enum):
    """High-level action selected from a spoken command."""

    IGNORE = "ignore"
    GREET = "greet"
    HELP = "help"
    STATUS = "status"
    EXIT = "exit"
    ECHO = "echo"


@dataclass(frozen=True, slots=True)
class VoiceCommand:
    """Parsed voice command after wake-phrase filtering."""

    transcript: str
    activated: bool
    action: VoiceAction = VoiceAction.IGNORE
    response_text: str | None = None
    target_identity_id: str | None = None
    target_display_name: str | None = None


class VoskTranscriber:
    """ASR backend powered by Vosk."""

    def __init__(
        self,
        config: VoiceConfig | None = None,
        model_loader: ModelLoader | None = None,
        recognizer_factory: RecognizerFactory | None = None,
    ) -> None:
        self._config = config or VoiceConfig()
        self._model_loader = model_loader or self._load_vosk_model
        self._recognizer_factory = recognizer_factory or self._create_recognizer
        self._model: object | None = None

    @property
    def config(self) -> VoiceConfig:
        return self._config

    def transcribe_samples(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> VoiceTranscript:
        rate = sample_rate or self._config.sample_rate
        normalized = self._normalize_audio(samples, rate)
        recognizer = self._make_recognizer(self._config.sample_rate)
        recognizer.AcceptWaveform(normalized.tobytes())
        raw = recognizer.FinalResult()
        return VoiceTranscript(text=self._extract_text(raw), raw_json=raw, is_final=True)

    def transcribe_wav_file(self, wav_path: str | Path) -> VoiceTranscript:
        sample_rate, samples = wavfile.read(str(wav_path))
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sample_rate != self._config.sample_rate:
            samples = self._resample(samples, sample_rate, self._config.sample_rate)
            sample_rate = self._config.sample_rate
        return self.transcribe_samples(np.asarray(samples), sample_rate=sample_rate)

    def transcribe_wav_segments(
        self,
        wav_path: str | Path,
        *,
        chunk_seconds: float = 0.75,
    ) -> list[VoiceTranscriptSegment]:
        sample_rate, samples = wavfile.read(str(wav_path))
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sample_rate != self._config.sample_rate:
            samples = self._resample(samples, sample_rate, self._config.sample_rate)
            sample_rate = self._config.sample_rate

        normalized = self._normalize_audio(np.asarray(samples), sample_rate)
        recognizer = self._make_recognizer(self._config.sample_rate)
        chunk_frames = max(1, int(self._config.sample_rate * chunk_seconds))
        segments: list[VoiceTranscriptSegment] = []
        offset = 0

        while offset < len(normalized):
            chunk = normalized[offset : offset + chunk_frames]
            if not len(chunk):
                break
            raw = None
            if recognizer.AcceptWaveform(chunk.tobytes()):
                raw = recognizer.Result()
            if raw:
                segment = self._segment_from_raw(
                    raw,
                    start_seconds=offset / self._config.sample_rate,
                    end_seconds=(offset + len(chunk)) / self._config.sample_rate,
                )
                if segment is not None:
                    segments.append(segment)
            offset += chunk_frames

        final_raw = recognizer.FinalResult()
        final_segment = self._segment_from_raw(
            final_raw,
            start_seconds=max(0.0, (offset - chunk_frames) / self._config.sample_rate),
            end_seconds=len(normalized) / self._config.sample_rate,
        )
        if final_segment is not None and not self._is_duplicate_segment(segments, final_segment):
            segments.append(final_segment)
        return segments

    def _make_recognizer(self, sample_rate: int) -> RecognizerLike:
        model = self._ensure_model()
        return self._recognizer_factory(model, sample_rate)

    def _ensure_model(self) -> object:
        if self._model is None:
            self._model = self._model_loader(self._config.model_path)
        return self._model

    @staticmethod
    def _load_vosk_model(model_path: str | Path) -> object:
        try:
            from vosk import Model
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("vosk is required for ASR.") from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found: {model_path}. "
                "Download a Chinese model from https://alphacephei.com/vosk/models "
                "and unpack it here."
            )
        return Model(str(model_path))

    @staticmethod
    def _create_recognizer(model: object, sample_rate: int) -> RecognizerLike:
        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("vosk is required for ASR.") from exc

        recognizer = KaldiRecognizer(model, sample_rate)
        return recognizer

    @staticmethod
    def _extract_text(raw_json: str) -> str:
        data = json.loads(raw_json)
        return str(data.get("text", "")).strip()

    @staticmethod
    def _segment_from_raw(raw_json: str, start_seconds: float, end_seconds: float) -> VoiceTranscriptSegment | None:
        data = json.loads(raw_json)
        text = str(data.get("text", "")).strip()
        if not text:
            return None

        words = data.get("result")
        if isinstance(words, list) and words:
            starts = [float(word.get("start", start_seconds)) for word in words if isinstance(word, dict)]
            ends = [float(word.get("end", end_seconds)) for word in words if isinstance(word, dict)]
            if starts and ends:
                start_seconds = min(starts)
                end_seconds = max(ends)

        return VoiceTranscriptSegment(
            text=text,
            raw_json=raw_json,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            is_final=True,
        )

    @staticmethod
    def _is_duplicate_segment(
        segments: list[VoiceTranscriptSegment],
        candidate: VoiceTranscriptSegment,
    ) -> bool:
        if not segments:
            return False
        last = segments[-1]
        return (
            last.text == candidate.text
            and abs(last.start_seconds - candidate.start_seconds) < 1e-6
            and abs(last.end_seconds - candidate.end_seconds) < 1e-6
        )

    def _normalize_audio(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        array = np.asarray(samples)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if array.dtype != np.int16:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, -1.0, 1.0)
                array = (array * 32767.0).astype(np.int16)
            else:
                array = array.astype(np.int16)
        if sample_rate != self._config.sample_rate:
            array = self._resample(array, sample_rate, self._config.sample_rate)
        return array.astype(np.int16)

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return np.asarray(samples)
        gcd = np.gcd(source_rate, target_rate)
        up = target_rate // gcd
        down = source_rate // gcd
        return resample_poly(np.asarray(samples), up, down)


class MicrophoneRecorder:
    """Blocking microphone recorder built on sounddevice."""

    def __init__(self, sample_rate: int = 16000, device: int | None = None) -> None:
        self._sample_rate = sample_rate
        self._device = device

    def record(self, seconds: float) -> np.ndarray:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("sounddevice is required for microphone capture.") from exc

        frames = max(1, int(self._sample_rate * seconds))
        audio = sd.rec(
            frames,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
        )
        sd.wait()
        return np.asarray(audio).reshape(-1)


class VoiceCommandRouter:
    """Map transcripts to simple voice actions."""

    def __init__(
        self,
        wake_phrase: str = "你好",
        greeting_service: GreetingService | None = None,
        identities: dict[str, PermanentIdentity] | None = None,
    ) -> None:
        self._wake_phrase = self._normalize(wake_phrase)
        self._greeting_service = greeting_service or GreetingService()
        self._identities = identities or {}

    @property
    def wake_phrase(self) -> str:
        return self._wake_phrase

    def route(
        self,
        transcript: str,
        identity_context: PermanentIdentity | None = None,
    ) -> VoiceCommand:
        normalized = self._normalize(transcript)
        if not normalized:
            return VoiceCommand(transcript="", activated=False)

        if not normalized.startswith(self._wake_phrase):
            return VoiceCommand(transcript=transcript, activated=False)

        remainder = normalized.removeprefix(self._wake_phrase).strip()
        if not remainder:
            if identity_context is not None:
                response = self._greeting_service.greeting_for(identity_context)
                if response is None:
                    response = f"你好，{identity_context.display_name}。"
                return VoiceCommand(
                    transcript=transcript,
                    activated=True,
                    action=VoiceAction.GREET,
                    response_text=response,
                    target_identity_id=identity_context.identity_id,
                    target_display_name=identity_context.display_name,
                )
            return VoiceCommand(
                transcript=transcript,
                activated=True,
                action=VoiceAction.ECHO,
                response_text="我在。",
            )

        if any(keyword in remainder for keyword in ("退出", "停止", "结束")):
            return VoiceCommand(
                transcript=transcript,
                activated=True,
                action=VoiceAction.EXIT,
                response_text="好的，语音模式已关闭。",
            )

        if any(keyword in remainder for keyword in ("帮助", "你能做什么", "能做什么")):
            return VoiceCommand(
                transcript=transcript,
                activated=True,
                action=VoiceAction.HELP,
                response_text="你可以说：问候爸爸、问候妈妈、查看状态、退出语音模式。",
            )

        if any(keyword in remainder for keyword in ("状态", "看看状态")):
            return VoiceCommand(
                transcript=transcript,
                activated=True,
                action=VoiceAction.STATUS,
                response_text="语音模式正常。",
            )

        target = identity_context or self._find_identity(remainder)
        if target is not None:
            response = self._greeting_service.greeting_for(target) or f"你好，{target.display_name}。"
            return VoiceCommand(
                transcript=transcript,
                activated=True,
                action=VoiceAction.GREET,
                response_text=response,
                target_identity_id=target.identity_id,
                target_display_name=target.display_name,
            )

        return VoiceCommand(
            transcript=transcript,
            activated=True,
            action=VoiceAction.ECHO,
            response_text=f"我听到了：{remainder}",
        )

    def _find_identity(self, text: str) -> PermanentIdentity | None:
        for identity in self._identities.values():
            if identity.identity_id and self._normalize(identity.identity_id) in text:
                return identity
            if identity.display_name and self._normalize(identity.display_name) in text:
                return identity
            if identity.role and self._normalize(identity.role) in text:
                return identity
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(text.split()).lower()


@dataclass(slots=True)
class VoiceInteractionController:
    """Glue recorder, ASR, router, and speech together."""

    transcriber: VoskTranscriber
    recorder: MicrophoneRecorder
    router: VoiceCommandRouter
    speech_service: QueuedSpeechService
    identity_context_provider: Callable[[], PermanentIdentity | None] | None = None

    def process_audio(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        identity_context: PermanentIdentity | None = None,
    ) -> VoiceCommand:
        transcript = self.transcriber.transcribe_samples(samples, sample_rate=sample_rate)
        return self.process_transcript(transcript.text, identity_context=identity_context)

    def process_wav(
        self,
        wav_path: str | Path,
        identity_context: PermanentIdentity | None = None,
    ) -> VoiceCommand:
        transcript = self.transcriber.transcribe_wav_file(wav_path)
        return self.process_transcript(transcript.text, identity_context=identity_context)

    def process_transcript(
        self,
        transcript: str,
        identity_context: PermanentIdentity | None = None,
    ) -> VoiceCommand:
        resolved_identity = identity_context
        if resolved_identity is None and self.identity_context_provider is not None:
            resolved_identity = self.identity_context_provider()
        command = self.router.route(transcript, identity_context=resolved_identity)
        if command.response_text:
            self.speech_service.speak(command.response_text)
        return command

    def listen_once(self) -> VoiceCommand:
        audio = self.recorder.record(self.transcriber.config.record_seconds)
        return self.process_audio(audio, sample_rate=self.transcriber.config.sample_rate)
