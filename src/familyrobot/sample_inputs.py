"""Resolve bundled sample input assets for local smoke tests."""

from __future__ import annotations

from pathlib import Path


SAMPLE_INPUT_NAMES = {
    "sample": "1780297629910.mp4",
    "test-video": "豆包.mp4",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def bundled_sample_input(name: str) -> Path:
    try:
        file_name = SAMPLE_INPUT_NAMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown sample input: {name}") from exc

    path = resolve_project_path(file_name)
    if not path.exists():
        raise FileNotFoundError(path)
    return path
