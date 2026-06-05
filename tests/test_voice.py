from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity
from familyrobot.voice import (
    VoiceAction,
    VoiceCommandRouter,
    VoiceConfig,
    VoiceInteractionController,
    VoskTranscriber,
)


class _FakeRecognizer:
    def __init__(self, raw_result: str) -> None:
        self.raw_result = raw_result
        self.accepted_payloads: list[bytes] = []

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        self.accepted_payloads.append(data)
        return True

    def Result(self) -> str:  # noqa: N802
        return self.raw_result

    def FinalResult(self) -> str:  # noqa: N802
        return self.raw_result


class _FakeSpeechService:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class _FakeRecorder:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.calls: list[float] = []

    def record(self, seconds: float) -> np.ndarray:
        self.calls.append(seconds)
        return self.audio


def test_vosk_transcriber_uses_config_sample_rate_and_caches_model(tmp_path: Path) -> None:
    loader_calls: list[Path] = []
    recognizer_calls: list[int] = []
    recognizers: list[_FakeRecognizer] = []

    def model_loader(model_path: str | Path) -> object:
        loader_calls.append(Path(model_path))
        return object()

    def recognizer_factory(model: object, sample_rate: int) -> _FakeRecognizer:
        recognizer_calls.append(sample_rate)
        recognizer = _FakeRecognizer('{"text": "你好小家 爸爸"}')
        recognizers.append(recognizer)
        return recognizer

    transcriber = VoskTranscriber(
        config=VoiceConfig(model_path=tmp_path / "vosk-model", sample_rate=16000),
        model_loader=model_loader,
        recognizer_factory=recognizer_factory,
    )

    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    transcript = transcriber.transcribe_samples(samples, sample_rate=8000)
    transcript_again = transcriber.transcribe_samples(samples, sample_rate=8000)

    assert transcript.text == "你好小家 爸爸"
    assert transcript_again.text == "你好小家 爸爸"
    assert loader_calls == [tmp_path / "vosk-model"]
    assert recognizer_calls == [16000, 16000]
    assert len(recognizers) == 2
    assert recognizers[0].accepted_payloads
    assert len(recognizers[0].accepted_payloads[0]) > 0


def test_vosk_transcriber_reads_wav_files(tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    wavfile.write(wav_path, 8000, np.array([0, 1000, -1000, 500], dtype=np.int16))

    loader_calls: list[Path] = []

    def model_loader(model_path: str | Path) -> object:
        loader_calls.append(Path(model_path))
        return object()

    transcriber = VoskTranscriber(
        config=VoiceConfig(model_path=tmp_path / "vosk-model", sample_rate=16000),
        model_loader=model_loader,
        recognizer_factory=lambda model, sample_rate: _FakeRecognizer('{"text": "打开语音"}'),
    )

    transcript = transcriber.transcribe_wav_file(wav_path)

    assert transcript.text == "打开语音"
    assert loader_calls == [tmp_path / "vosk-model"]


def test_voice_command_router_routes_identity_and_fallback_text() -> None:
    service = GreetingService()
    service.register_template("father", "爸爸，欢迎回家。", display_name="爸爸", role="父亲")
    identities = {
        "father": PermanentIdentity(
            identity_id="father",
            display_name="爸爸",
            role="父亲",
        )
    }
    router = VoiceCommandRouter(
        wake_phrase="你好小家",
        greeting_service=service,
        identities=identities,
    )

    greeting = router.route("你好小家 爸爸")
    echo = router.route("你好小家 今天天气怎么样")
    ignored = router.route("普通聊天内容")

    assert greeting.activated is True
    assert greeting.action is VoiceAction.GREET
    assert greeting.response_text == "爸爸，欢迎回家。"
    assert greeting.target_identity_id == "father"
    assert greeting.target_display_name == "爸爸"

    assert echo.activated is True
    assert echo.action is VoiceAction.ECHO
    assert echo.response_text == "我听到了：今天天气怎么样"

    assert ignored.activated is False
    assert ignored.action is VoiceAction.IGNORE


def test_voice_controller_speaks_routed_response() -> None:
    service = GreetingService()
    service.register_template("mother", "妈妈，欢迎回家。", display_name="妈妈", role="母亲")
    router = VoiceCommandRouter(
        wake_phrase="你好小家",
        greeting_service=service,
        identities={
            "mother": PermanentIdentity(
                identity_id="mother",
                display_name="妈妈",
                role="母亲",
            )
        },
    )

    controller = VoiceInteractionController(
        transcriber=VoskTranscriber(
            config=VoiceConfig(),
            model_loader=lambda model_path: object(),
            recognizer_factory=lambda model, sample_rate: _FakeRecognizer('{"text": "你好小家 妈妈"}'),
        ),
        recorder=_FakeRecorder(np.zeros(4, dtype=np.float32)),
        router=router,
        speech_service=_FakeSpeechService(),
    )

    command = controller.process_transcript("你好小家 妈妈")

    assert command.action is VoiceAction.GREET
    assert command.response_text == "妈妈，欢迎回家。"
    assert controller.speech_service.spoken == ["妈妈，欢迎回家。"]


def test_voice_command_router_prefers_visual_identity_context() -> None:
    service = GreetingService()
    service.register_template("father", "爸爸，今天工作怎么样？", display_name="爸爸", role="父亲")
    father = PermanentIdentity(
        identity_id="father",
        display_name="爸爸",
        role="父亲",
    )
    router = VoiceCommandRouter(
        wake_phrase="你好",
        greeting_service=service,
        identities={},
    )

    command = router.route("你好", identity_context=father)

    assert command.activated is True
    assert command.action is VoiceAction.GREET
    assert command.response_text == "爸爸，今天工作怎么样？"
    assert command.target_identity_id == "father"
    assert command.target_display_name == "爸爸"


def test_voice_controller_uses_identity_context_provider() -> None:
    service = GreetingService()
    service.register_template("father", "爸爸，今天工作怎么样？", display_name="爸爸", role="父亲")
    father = PermanentIdentity(
        identity_id="father",
        display_name="爸爸",
        role="父亲",
    )
    router = VoiceCommandRouter(
        wake_phrase="你好",
        greeting_service=service,
        identities={},
    )

    controller = VoiceInteractionController(
        transcriber=VoskTranscriber(
            config=VoiceConfig(),
            model_loader=lambda model_path: object(),
            recognizer_factory=lambda model, sample_rate: _FakeRecognizer('{"text": "你好"}'),
        ),
        recorder=_FakeRecorder(np.zeros(4, dtype=np.float32)),
        router=router,
        speech_service=_FakeSpeechService(),
        identity_context_provider=lambda: father,
    )

    command = controller.process_transcript("你好")

    assert command.action is VoiceAction.GREET
    assert command.response_text == "爸爸，今天工作怎么样？"
    assert controller.speech_service.spoken == ["爸爸，今天工作怎么样？"]
