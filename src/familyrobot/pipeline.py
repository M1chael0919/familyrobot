"""Pipeline orchestration for the family robot demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

from familyrobot.capture import CameraVideoInputAdapter
from familyrobot.detection import PersonDetection, YoloPersonDetector
from familyrobot.tracking import DeepSortTracker, TrackedPerson
from familyrobot.sample_inputs import resolve_project_path


@dataclass(frozen=True)
class FrameSummary:
    """Summary of one processed frame."""

    frame_index: int
    detections: list[PersonDetection] = field(default_factory=list)
    tracks: list[TrackedPerson] = field(default_factory=list)


@dataclass(frozen=True)
class DemoResult:
    """Collected results from a demo run."""

    frames: list[FrameSummary] = field(default_factory=list)


class FamilyRobotPipeline:
    """Minimal pipeline for vision-stage smoke testing."""

    def __init__(
        self,
        input_adapter: CameraVideoInputAdapter,
        detector: YoloPersonDetector,
        tracker: DeepSortTracker,
        output: TextIO,
        display: bool = False,
        window_name: str = "Family Robot",
    ) -> None:
        self._input_adapter = input_adapter
        self._detector = detector
        self._tracker = tracker
        self._output = output
        self._display = display
        self._window_name = window_name

    def run(self, max_frames: int | None = None) -> DemoResult:
        """Run the vision pipeline and print frame summaries."""

        frame_summaries: list[FrameSummary] = []
        processed = 0

        with self._input_adapter as input_adapter:
            while True:
                if max_frames is not None and processed >= max_frames:
                    break

                frame = input_adapter.read_frame()
                if frame is None:
                    break

                detections = self._detector.detect(frame)
                tracks = self._tracker.track(detections, frame=frame)
                summary = FrameSummary(
                    frame_index=processed,
                    detections=list(detections),
                    tracks=list(tracks),
                )
                frame_summaries.append(summary)
                self._write_summary(summary)
                if self._display:
                    if not self._show_frame(frame, summary):
                        break
                processed += 1

        if self._display:
            self._close_display()

        return DemoResult(frames=frame_summaries)

    def _write_summary(self, summary: FrameSummary) -> None:
        self._output.write(
            f"frame={summary.frame_index} "
            f"detections={len(summary.detections)} "
            f"tracks={len(summary.tracks)}\n"
        )
        for detection in summary.detections:
            self._output.write(
                "  detection "
                f"bbox=({detection.bbox.x1:.1f},{detection.bbox.y1:.1f},"
                f"{detection.bbox.x2:.1f},{detection.bbox.y2:.1f}) "
                f"confidence={detection.confidence:.3f}\n"
            )
        for track in summary.tracks:
            self._output.write(
                "  track "
                f"id={track.track_id} "
                f"bbox=({track.bbox.x1:.1f},{track.bbox.y1:.1f},"
                f"{track.bbox.x2:.1f},{track.bbox.y2:.1f}) "
                f"confidence={track.confidence:.3f}\n"
            )

    def _show_frame(self, frame: Any, summary: FrameSummary) -> bool:
        """Draw detections and tracks on a frame and show them in a window."""

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("OpenCV is required for display mode.") from exc

        annotated = frame.copy()

        for detection in summary.detections:
            self._draw_box(
                annotated,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
                color=(0, 255, 0),
                label=f"person {detection.confidence:.2f}",
            )

        for track in summary.tracks:
            self._draw_box(
                annotated,
                track.bbox.x1,
                track.bbox.y1,
                track.bbox.x2,
                track.bbox.y2,
                color=(0, 0, 255),
                label=f"id {track.track_id}",
            )

        cv2.imshow(self._window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        return True

    def _close_display(self) -> None:
        try:
            import cv2
        except ImportError:  # pragma: no cover - depends on environment
            return

        cv2.destroyWindow(self._window_name)

    @staticmethod
    def _draw_box(
        frame: Any,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("OpenCV is required for drawing overlays.") from exc

        start = (int(x1), int(y1))
        end = (int(x2), int(y2))
        cv2.rectangle(frame, start, end, color, 2)
        text_origin = (start[0], max(0, start[1] - 8))
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def build_pipeline(
    source: int | str | Path = 0,
    model_path: str | Path = "models/yolov8n.pt",
    output: TextIO | None = None,
    display: bool = False,
    window_name: str = "Family Robot",
) -> FamilyRobotPipeline:
    """Build the default demo pipeline."""

    if output is None:
        import sys

        output = sys.stdout

    return FamilyRobotPipeline(
        input_adapter=CameraVideoInputAdapter(source=source),
        detector=YoloPersonDetector(model_path=resolve_project_path(model_path)),
        tracker=DeepSortTracker(),
        output=output,
        display=display,
        window_name=window_name,
    )
