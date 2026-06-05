"""Replay recorded video with aligned wake-word voice interactions."""

from __future__ import annotations

import argparse
from pathlib import Path

from familyrobot.enrollment import EnrollmentStore, default_enrollment_root
from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity
from familyrobot.realtime_gui import build_realtime_session
from familyrobot.sample_inputs import resolve_project_path
from familyrobot.speech import QueuedSpeechService
from familyrobot.voice import MicrophoneRecorder, VoiceCommandRouter, VoiceConfig, VoiceInteractionController, VoskTranscriber
from familyrobot.video_voice import VideoAudioExtractor, iter_video_wake_events


class _SilentSpeechService:
    def speak(self, text: str) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family Robot recorded video voice replay")
    parser.add_argument("source", help="Path to a local video file.")
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
        help="Path to a YOLO model file.",
    )
    parser.add_argument(
        "--enrollment-root",
        default=default_enrollment_root(),
        type=Path,
        help="Local enrollment directory root.",
    )
    parser.add_argument(
        "--model-path",
        default="models/vosk-model-small-cn-0.22",
        help="Path to the local Vosk model directory.",
    )
    parser.add_argument(
        "--wake-phrase",
        default="你好",
        help="Wake phrase used to activate recorded video interactions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used to store extracted audio.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the video playback window while processing.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak routed responses through local TTS.",
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


def _build_voice_controller(
    *,
    wake_phrase: str,
    model_path: Path,
    enrollment_root: Path,
    speak: bool,
) -> VoiceInteractionController:
    config = VoiceConfig(model_path=model_path, wake_phrase=wake_phrase)
    return VoiceInteractionController(
        transcriber=VoskTranscriber(config=config),
        recorder=MicrophoneRecorder(sample_rate=config.sample_rate),
        router=VoiceCommandRouter(
            wake_phrase=wake_phrase,
            greeting_service=GreetingService(),
            identities=_load_identities(enrollment_root),
        ),
        speech_service=QueuedSpeechService() if speak else _SilentSpeechService(),
    )


def _show_frame(window_name: str, frame) -> bool:
    import cv2

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def main() -> int:
    args = parse_args()
    video_path = resolve_project_path(args.source)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else None
    session = build_realtime_session(
        source=video_path,
        model_path=resolve_project_path(args.model),
        enrollment_root=args.enrollment_root,
    )
    voice_controller = _build_voice_controller(
        wake_phrase=args.wake_phrase,
        model_path=Path(args.model_path),
        enrollment_root=args.enrollment_root,
        speak=args.speak,
    )
    alignment, wake_events = iter_video_wake_events(
        video_path,
        wake_phrase=args.wake_phrase,
        transcriber=voice_controller.transcriber,
        extractor=VideoAudioExtractor(),
        output_dir=output_dir,
    )

    print(f"video={alignment.metadata.video_path}")
    print(f"audio={alignment.audio_path}")
    if wake_events:
        print(f"wake_events={len(wake_events)}")
    else:
        print("wake_events=0")

    next_event_index = 0
    window_name = "Family Robot Video Voice Replay"

    with session.input_adapter:
        frame_index = 0
        while True:
            frame = session.input_adapter.read_frame()
            if frame is None:
                break

            annotated, overlay = session.step(frame, frame_index)
            current_timestamp = alignment.timestamp_for_frame(frame_index)

            while next_event_index < len(wake_events) and wake_events[next_event_index].timestamp_seconds <= current_timestamp:
                event = wake_events[next_event_index]
                identity_context = session.current_interaction_identity()
                command = voice_controller.process_transcript(
                    event.transcript,
                    identity_context=identity_context,
                )
                identity_name = identity_context.display_name if identity_context is not None else "unknown"
                print(
                    f"[video-voice] frame={frame_index} "
                    f"ts={event.timestamp_seconds:.2f}s "
                    f"identity={identity_name} "
                    f"transcript={event.transcript} "
                    f"response={command.response_text or ''}"
                )
                next_event_index += 1

            if args.display and not _show_frame(window_name, annotated):
                break

            frame_index += 1

    if args.display:
        import cv2

        cv2.destroyWindow(window_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
