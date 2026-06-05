from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from familyrobot.enrollment import EnrollmentManifest, EnrollmentRecord, EnrollmentStore
from familyrobot.voice import VoiceAction, VoiceCommand
import voice_main


class _FakeSpeechService:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeRouter:
    wake_phrase = "你好小家"


class _FakeController:
    def __init__(self) -> None:
        self.speech_service = _FakeSpeechService()
        self.router = _FakeRouter()
        self.processed_paths: list[Path] = []
        self.listen_calls = 0

    def process_wav(self, wav_path: str | Path) -> VoiceCommand:
        self.processed_paths.append(Path(wav_path))
        return VoiceCommand(
            transcript="你好小家 爸爸",
            activated=True,
            action=VoiceAction.GREET,
            response_text="爸爸，欢迎回家。",
            target_identity_id="father",
            target_display_name="爸爸",
        )

    def listen_once(self) -> VoiceCommand:
        self.listen_calls += 1
        return VoiceCommand(
            transcript="你好小家 退出",
            activated=True,
            action=VoiceAction.EXIT,
            response_text="好的，语音模式已关闭。",
        )


def test_build_voice_controller_loads_identity_from_enrollment(tmp_path: Path) -> None:
    store = EnrollmentStore(tmp_path / "enrollment")
    store.save(
        EnrollmentManifest(
            records=[
                EnrollmentRecord(
                    identity_id="father",
                    display_name="爸爸",
                    role="父亲",
                )
            ]
        )
    )

    controller = voice_main.build_voice_controller(
        model_path=tmp_path / "vosk-model",
        enrollment_root=store.root,
        wake_phrase="你好小家",
        record_seconds=2.0,
        device=None,
    )

    command = controller.router.route("你好小家 爸爸")

    assert command.activated is True
    assert command.action is VoiceAction.GREET
    assert command.target_identity_id == "father"
    assert command.target_display_name == "爸爸"
    assert command.response_text == "爸爸今天工作怎么样？"


def test_run_voice_mode_processes_wav_once(tmp_path: Path) -> None:
    controller = _FakeController()
    wav_path = tmp_path / "test.wav"

    exit_code = voice_main.run_voice_mode(controller, wav_path)

    assert exit_code == 0
    assert controller.processed_paths == [wav_path]
    assert controller.listen_calls == 0


def test_main_closes_speech_service(monkeypatch, tmp_path: Path) -> None:
    controller = _FakeController()
    captured_args = {}

    def fake_build_voice_controller(**kwargs):
        captured_args.update(kwargs)
        return controller

    monkeypatch.setattr(voice_main, "build_voice_controller", fake_build_voice_controller)
    monkeypatch.setattr(
        voice_main,
        "parse_args",
        lambda: Namespace(
            audio=str(tmp_path / "test.wav"),
            model=str(tmp_path / "model"),
            enrollment_root=tmp_path / "enrollment",
            wake_phrase="你好小家",
            record_seconds=4.0,
            device=None,
        ),
    )

    exit_code = voice_main.main()

    assert exit_code == 0
    assert controller.processed_paths == [tmp_path / "test.wav"]
    assert controller.speech_service.closed is True
    assert captured_args["wake_phrase"] == "你好小家"
