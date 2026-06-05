"""Inspect recorded video audio extraction and frame-time alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from familyrobot.sample_inputs import resolve_project_path
from familyrobot.video_voice import build_recorded_video_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family Robot video alignment helper")
    parser.add_argument("source", help="Path to a local video file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where extracted audio should be written.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to inspect.",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Optional timestamp in seconds to map back to a frame index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = resolve_project_path(args.source)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else None
    alignment = build_recorded_video_alignment(video_path, output_dir=output_dir)

    metadata = alignment.metadata
    print(f"video={metadata.video_path}")
    print(f"fps={metadata.fps:.3f}")
    print(f"frame_count={metadata.frame_count}")
    if metadata.duration_seconds is not None:
        print(f"duration_seconds={metadata.duration_seconds:.3f}")
    print(f"audio_path={alignment.audio_path}")
    print(f"frame[{args.frame_index}]_timestamp={alignment.timestamp_for_frame(args.frame_index):.3f}")

    if args.timestamp is not None:
        frame_index = alignment.frame_for_timestamp(args.timestamp)
        window_start, window_end = alignment.frame_window_for_timestamp(args.timestamp)
        print(f"timestamp[{args.timestamp:.3f}]_frame={frame_index}")
        print(f"timestamp[{args.timestamp:.3f}]_frame_window={window_start}-{window_end}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
