"""Standalone entrypoint for wake-word voice interaction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from familyrobot.enrollment import EnrollmentStore, default_enrollment_root
from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family Robot voice mode")
    parser.add_argument(
        "--audio",
        default="mic",
        help="Use 'mic' for live listening or provide a .wav file path for one-shot testing.",
    )
    parser.add_argument(
        "--model",
        default="models/vosk-model-small-cn-0.22",
        help="Path to the local Vosk model directory.",
    )
    parser.add_argument(
        "--enrollment-root",
        default=default_enrollment_root(),
        type=Path,
        help="Local enrollment directory root.",
    )
    parser.add_argument(
        "--wake-phrase",
        default="你好",
        help="Wake phrase used to activate the voice mode.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=4.0,
        help="Microphone capture duration for each listening cycle.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index.",
    )
    return parser.parse_args()


def _load_identities(enrollment_root: Path) -> dict[str, PermanentIdentity]:
    store = EnrollmentStore(enrollment_root)
    manifest = store.load()
    identities: dict[str, PermanentIdentity] = {}
    for record in manifest.records:
        identities[record.identity_id] = PermanentIdentity(
            identity_id=record.identity_id,
            display_name=record.display_name,
            role=record.role,
            enrolled_at=record.enrolled_at,
            metadata=record.metadata,
        )
    return identities


def build_voice_controller(
    *,
    model_path: Path,
    enrollment_root: Path,
    wake_phrase: str,
    record_seconds: float,
    device: int | None,
) -> VoiceInteractionController:
    config = VoiceConfig(
        model_path=model_path,
        record_seconds=record_seconds,
        wake_phrase=wake_phrase,
    )
    transcriber = VoskTranscriber(config=config)
    recorder = MicrophoneRecorder(sample_rate=config.sample_rate, device=device)
    router = VoiceCommandRouter(
        wake_phrase=wake_phrase,
        greeting_service=GreetingService(),
        identities=_load_identities(enrollment_root),
    )
    speech_service = QueuedSpeechService()
    return VoiceInteractionController(
        transcriber=transcriber,
        recorder=recorder,
        router=router,
        speech_service=speech_service,
    )


def _print_command(command: VoiceCommand) -> None:
    if not command.transcript:
        print("未识别到有效语音。")
        return

    if not command.activated:
        print(f"未唤醒：{command.transcript}")
        return

    action = command.action.value
    print(f"已唤醒：{action} | {command.transcript}")
    if command.response_text:
        print(command.response_text)


def run_voice_mode(controller: VoiceInteractionController, audio_source: str | Path) -> int:
    if isinstance(audio_source, Path):
        command = controller.process_wav(audio_source)
        _print_command(command)
        return 0

    if audio_source.strip().lower() != "mic":
        command = controller.process_wav(Path(audio_source))
        _print_command(command)
        return 0

    print(f"语音模式已启动，请说出唤醒词：{controller.router.wake_phrase}")
    while True:
        command = controller.listen_once()
        _print_command(command)
        if command.action is VoiceAction.EXIT:
            return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    controller = build_voice_controller(
        model_path=Path(args.model),
        enrollment_root=args.enrollment_root,
        wake_phrase=args.wake_phrase,
        record_seconds=args.record_seconds,
        device=args.device,
    )

    try:
        return run_voice_mode(controller, args.audio)
    finally:
        controller.speech_service.close()


if __name__ == "__main__":
    raise SystemExit(main())
