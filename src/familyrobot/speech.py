"""Text-to-speech helpers for the family robot demo."""

from __future__ import annotations

import shutil
import subprocess
import sys
from time import perf_counter
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, Protocol
from pathlib import Path


class SpeechBackend(Protocol):
    """Minimal interface for a speech synthesis backend."""

    def say(self, text: str) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    """Runtime configuration for speech output."""

    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    pyttsx3_rate: int = 185
    pyttsx3_volume: float = 1.0
    preferred_voice_keywords: tuple[str, ...] = ("zh", "chinese", "mandarin", "sapi")


class EdgePlaybackBackend:
    """Natural-sounding online TTS backend powered by edge-tts."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self._config = config or SpeechConfig()
        self._command = self._resolve_command()

    def say(self, text: str) -> None:
        command = self._command
        if command is None:
            raise RuntimeError("edge-playback is not available.")

        cmd = [
            command,
            "--voice",
            self._config.edge_voice,
            "--text",
            text,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        return None

    def _resolve_command(self) -> str | None:
        command = shutil.which("edge-playback")
        if command is not None:
            return command

        scripts_dir = Path(sys.executable).resolve().with_name("Scripts")
        candidate = scripts_dir / "edge-playback.exe"
        if candidate.exists():
            return str(candidate)
        return None


class Pyttsx3Backend:
    """Offline TTS backend powered by pyttsx3."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self._config = config or SpeechConfig()
        self._engine: object | None = None

    def say(self, text: str) -> None:
        engine = self._ensure_engine()
        engine.say(text)
        engine.runAndWait()

    def stop(self) -> None:
        if self._engine is None:
            return
        self._engine.stop()

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine

        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("pyttsx3 is required for local speech output.") from exc

        engine = pyttsx3.init()
        self._configure_engine(engine)
        self._engine = engine
        return engine

    def _configure_engine(self, engine: object) -> None:
        voices = list(getattr(engine, "getProperty")("voices") or [])
        selected_voice = self._select_voice(voices)
        if selected_voice is not None:
            getattr(engine, "setProperty")("voice", selected_voice)
        getattr(engine, "setProperty")("rate", self._config.pyttsx3_rate)
        getattr(engine, "setProperty")("volume", self._config.pyttsx3_volume)

    def _select_voice(self, voices: list[object]) -> str | None:
        for voice in voices:
            voice_id = str(getattr(voice, "id", ""))
            voice_name = str(getattr(voice, "name", ""))
            language_blob = str(getattr(voice, "languages", ""))
            haystack = f"{voice_id} {voice_name} {language_blob}".lower()
            if any(keyword in haystack for keyword in self._config.preferred_voice_keywords):
                return voice_id
        return None


class FallbackSpeechBackend:
    """Try a primary backend first and fall back to a secondary backend."""

    def __init__(self, primary: SpeechBackend, fallback: SpeechBackend) -> None:
        self._primary = primary
        self._fallback = fallback

    def say(self, text: str) -> None:
        try:
            self._primary.say(text)
        except Exception:
            self._fallback.say(text)

    def stop(self) -> None:
        self._primary.stop()
        self._fallback.stop()


@dataclass(slots=True)
class QueuedSpeechService:
    """Queue-backed speech service that keeps UI threads responsive."""

    backend: SpeechBackend = field(
        default_factory=lambda: FallbackSpeechBackend(
            primary=EdgePlaybackBackend(),
            fallback=Pyttsx3Backend(),
        )
    )
    timing_callback: Callable[[str, float], None] | None = None
    _queue: Queue[str | None] = field(default_factory=Queue, init=False)
    _thread: Thread | None = field(default=None, init=False)
    _running: Event = field(default_factory=Event, init=False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = Thread(target=self._run, name="familyrobot-tts", daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if self._thread is None:
            self.start()
        self._queue.put(text)

    def close(self) -> None:
        if self._thread is None:
            return
        self._running.clear()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        self.backend.stop()
        self._thread = None

    def _run(self) -> None:
        while self._running.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue

            if item is None:
                break

            started = perf_counter()
            self.backend.say(item)
            if self.timing_callback is not None:
                self.timing_callback(item, perf_counter() - started)
