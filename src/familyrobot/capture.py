"""Camera and video input adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


Frame = Any


class CaptureLike(Protocol):
    """Minimal interface for a video capture object."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Frame]: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class InputSource:
    """Normalized input source for the adapter."""

    source: int | str | Path = 0
    backend: int | None = None

    def normalized(self) -> int | str:
        if isinstance(self.source, Path):
            return str(self.source)
        return self.source


CaptureFactory = Callable[[int | str, int | None], CaptureLike]


class CameraVideoInputAdapter:
    """Thin wrapper around a camera or local video file."""

    def __init__(
        self,
        source: int | str | Path = 0,
        backend: int | None = None,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self._input = InputSource(source=source, backend=backend)
        self._capture_factory = capture_factory or self._create_opencv_capture
        self._capture: CaptureLike | None = None

    def open(self) -> "CameraVideoInputAdapter":
        """Open the configured source."""

        if self._capture is not None:
            return self

        capture = self._capture_factory(
            self._input.normalized(),
            self._input.backend,
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open camera or video source.")

        self._capture = capture
        return self

    def is_open(self) -> bool:
        """Return whether the source is currently open."""

        return self._capture is not None and self._capture.isOpened()

    def read(self) -> tuple[bool, Frame]:
        """Read a single frame from the source."""

        if self._capture is None:
            raise RuntimeError("The input source is not open.")
        return self._capture.read()

    def read_frame(self) -> Frame | None:
        """Read a frame and return ``None`` when the source is exhausted."""

        success, frame = self.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        """Release the underlying capture object."""

        if self._capture is None:
            return
        self._capture.release()
        self._capture = None

    def __enter__(self) -> "CameraVideoInputAdapter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @staticmethod
    def _create_opencv_capture(source: int | str, backend: int | None) -> CaptureLike:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "OpenCV is required to open a real camera or video source."
            ) from exc

        if backend is None:
            return cv2.VideoCapture(source)
        return cv2.VideoCapture(source, backend)
