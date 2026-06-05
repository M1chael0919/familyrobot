from __future__ import annotations

from pathlib import Path

import pytest

from familyrobot.capture import CameraVideoInputAdapter, InputSource


class FakeCapture:
    def __init__(self, frames: list[tuple[bool, object]], opened: bool = True) -> None:
        self.frames = frames
        self.opened = opened
        self.read_calls = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object]:
        self.read_calls += 1
        if self.read_calls <= len(self.frames):
            return self.frames[self.read_calls - 1]
        return False, None

    def release(self) -> None:
        self.released = True
        self.opened = False


def test_input_source_normalizes_path() -> None:
    source = InputSource(source=Path("video.mp4"), backend=None)

    assert source.normalized() == "video.mp4"


def test_adapter_opens_reads_and_releases() -> None:
    created: list[tuple[object, object]] = []

    def factory(source: int | str, backend: int | None) -> FakeCapture:
        created.append((source, backend))
        return FakeCapture([(True, "frame-1"), (False, None)])

    adapter = CameraVideoInputAdapter(
        source=Path("sample.mp4"),
        backend=123,
        capture_factory=factory,
    )

    with adapter as opened:
        assert opened.is_open()
        assert created == [("sample.mp4", 123)]
        assert opened.read_frame() == "frame-1"
        assert opened.read_frame() is None

    assert adapter.is_open() is False


def test_adapter_raises_when_source_cannot_open() -> None:
    def factory(source: int | str, backend: int | None) -> FakeCapture:
        return FakeCapture([], opened=False)

    adapter = CameraVideoInputAdapter(capture_factory=factory)

    with pytest.raises(RuntimeError, match="Unable to open camera or video source"):
        adapter.open()
