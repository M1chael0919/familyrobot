"""Standalone entrypoint for the realtime recognition GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from familyrobot.realtime_window import build_realtime_window
from familyrobot.enrollment import default_enrollment_root
from familyrobot.sample_inputs import bundled_sample_input
from familyrobot.sample_inputs import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family Robot realtime GUI")
    parser.add_argument("--source", default=0, help="Camera index, path, or sample alias.")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Path to a YOLO model file.")
    parser.add_argument(
        "--enrollment-root",
        default=default_enrollment_root(),
        type=Path,
        help="Local enrollment directory root.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak recognized greetings through local TTS.",
    )
    parser.add_argument(
        "--video-voice",
        action="store_true",
        help="When the source is a local video file, extract and play the original audio while routing wake-word events.",
    )
    return parser.parse_args()


def _coerce_source(value: str) -> int | str | Path:
    normalized = value.strip().lower()
    if normalized in {"sample", "test-video"}:
        return bundled_sample_input(normalized)
    try:
        return int(value)
    except ValueError:
        return Path(value)


def main() -> int:
    from PySide6.QtWidgets import QApplication

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    app = QApplication([])
    window = build_realtime_window(
        source=_coerce_source(args.source),
        model_path=resolve_project_path(args.model),
        enrollment_root=args.enrollment_root,
        speak=args.speak,
        video_voice=args.video_voice,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

