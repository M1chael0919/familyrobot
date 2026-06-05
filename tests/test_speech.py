from __future__ import annotations

from familyrobot.speech import EdgePlaybackBackend, FallbackSpeechBackend, QueuedSpeechService


class _FakeSpeechBackend:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stopped = False

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        self.stopped = True


def test_queued_speech_service_speaks_enqueued_text() -> None:
    backend = _FakeSpeechBackend()
    service = QueuedSpeechService(backend=backend)

    service.start()
    service.speak("\u4f60\u597d\uff0c\u7238\u7238\u3002")
    service.close()

    assert backend.spoken == ["\u4f60\u597d\uff0c\u7238\u7238\u3002"]
    assert backend.stopped is True


def test_fallback_backend_uses_secondary_on_primary_error() -> None:
    spoken: list[str] = []

    class Primary:
        def say(self, text: str) -> None:
            raise RuntimeError("boom")

        def stop(self) -> None:
            return None

    class Secondary:
        def say(self, text: str) -> None:
            spoken.append(text)

        def stop(self) -> None:
            return None

    backend = FallbackSpeechBackend(primary=Primary(), fallback=Secondary())
    backend.say("\u4f60\u597d\uff0c\u5988\u5988\u3002")

    assert spoken == ["\u4f60\u597d\uff0c\u5988\u5988\u3002"]


def test_edge_backend_builds_edge_playback_command(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "familyrobot.speech.shutil.which",
        lambda name: "edge-playback" if name == "edge-playback" else None,
    )
    monkeypatch.setattr(
        "familyrobot.speech.subprocess.run",
        lambda cmd, check, stdout, stderr: calls.append(cmd),
    )

    backend = EdgePlaybackBackend()
    backend.say("\u4f60\u597d\uff0c\u5bb6\u5ead\u673a\u5668\u4eba\u3002")

    assert calls and calls[0][0] == "edge-playback"
    assert "--voice" in calls[0]


def test_queued_speech_service_reports_timing() -> None:
    backend = _FakeSpeechBackend()
    timings: list[tuple[str, float]] = []

    service = QueuedSpeechService(
        backend=backend,
        timing_callback=lambda text, elapsed: timings.append((text, elapsed)),
    )

    service.start()
    service.speak("你好，家庭机器人。")
    service.close()

    assert backend.spoken == ["你好，家庭机器人。"]
    assert timings and timings[0][0] == "你好，家庭机器人。"
    assert timings[0][1] >= 0.0
