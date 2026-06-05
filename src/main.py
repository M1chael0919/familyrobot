"""Demo entrypoint for the family robot pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from familyrobot.pipeline import build_pipeline
from familyrobot.sample_inputs import bundled_sample_input
from familyrobot.sample_inputs import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family Robot demo runner")
    parser.add_argument(
        "--source",
        default=0,
        help="Camera index or path to a local video file.",
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
        help="Path to a YOLO model file.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for smoke testing.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show an OpenCV window with detection and track overlays.",
    )
    parser.add_argument(
        "--window-name",
        default="Family Robot",
        help="Window title used in display mode.",
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
    args = parse_args()
    pipeline = build_pipeline(
        source=_coerce_source(args.source),
        model_path=resolve_project_path(args.model),
        display=args.display,
        window_name=args.window_name,
    )
    pipeline.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
