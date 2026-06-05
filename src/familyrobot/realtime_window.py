"""PySide6 realtime recognition window."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from familyrobot.enrollment import EnrollmentStore, default_enrollment_root
from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity
from familyrobot.realtime_gui import (
    VoiceListener,
    VoiceTelemetrySink,
    build_realtime_session,
)
from familyrobot.sample_inputs import resolve_project_path
from familyrobot.speech import QueuedSpeechService
from familyrobot.video_voice import (
    RecordedVideoAlignment,
    VideoAudioExtractor,
    VideoWakeEvent,
    WavAudioPlayer,
    iter_video_wake_events,
)
from familyrobot.voice import (
    MicrophoneRecorder,
    VoiceCommandRouter,
    VoiceConfig,
    VoiceInteractionController,
    VoskTranscriber,
)


def _load_qt_widgets() -> dict[str, Any]:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    return {
        "QApplication": QApplication,
        "QHBoxLayout": QHBoxLayout,
        "QImage": QImage,
        "QLabel": QLabel,
        "QPixmap": QPixmap,
        "QSizePolicy": QSizePolicy,
        "QTextEdit": QTextEdit,
        "QTimer": QTimer,
        "Qt": Qt,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
    }


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
    enrollment_root: Path,
    wake_phrase: str,
    record_seconds: float,
    device: int | None,
    telemetry: VoiceTelemetrySink,
    identity_context_provider: Callable[[], PermanentIdentity | None] | None = None,
) -> VoiceInteractionController:
    config = VoiceConfig(
        model_path=Path("models/vosk-model-small-cn-0.22"),
        record_seconds=record_seconds,
        wake_phrase=wake_phrase,
    )
    return VoiceInteractionController(
        transcriber=VoskTranscriber(config=config),
        recorder=MicrophoneRecorder(sample_rate=config.sample_rate, device=device),
        router=VoiceCommandRouter(
            wake_phrase=wake_phrase,
            greeting_service=GreetingService(),
            identities=_load_identities(enrollment_root),
        ),
        speech_service=QueuedSpeechService(timing_callback=telemetry.record_tts),
        identity_context_provider=identity_context_provider,
    )


def _is_recorded_video_source(source: int | str | Path) -> bool:
    return not isinstance(source, int)


class RealtimeRecognitionWindow(_load_qt_widgets()["QWidget"]):
    def __init__(
        self,
        session: Any,
        speech_service: QueuedSpeechService | None = None,
        voice_controller: VoiceInteractionController | None = None,
        telemetry: VoiceTelemetrySink | None = None,
        video_alignment: RecordedVideoAlignment | None = None,
        video_wake_events: list[VideoWakeEvent] | None = None,
        video_audio_path: Path | None = None,
    ) -> None:
        widgets = _load_qt_widgets()
        super().__init__()
        self._session = session
        self._speech_service = speech_service
        self._voice_controller = voice_controller
        self._telemetry = telemetry
        self._voice_listener: VoiceListener | None = None
        self._video_alignment = video_alignment
        self._video_wake_events = video_wake_events or []
        self._video_wake_index = 0
        self._video_audio_path = video_audio_path
        self._video_audio_player = WavAudioPlayer() if video_audio_path is not None else None
        self._video_audio_started = False
        self._video_audio_start_pending = False
        self._voice_start_pending = False
        self._frame_index = 0
        self._widgets = widgets

        self.setWindowTitle("家庭机器人 · 实时识别")
        self.resize(1280, 820)

        self.video_label = widgets["QLabel"]("正在加载视频流...")
        self.video_label.setAlignment(widgets["Qt"].AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setSizePolicy(
            widgets["QSizePolicy"].Expanding,
            widgets["QSizePolicy"].Expanding,
        )

        self.status_label = widgets["QLabel"]("就绪")
        self.overlay_log = widgets["QTextEdit"]()
        self.overlay_log.setReadOnly(True)
        self.overlay_log.setMinimumWidth(280)
        self.overlay_log.setPlaceholderText("识别日志会显示在这里")

        self.greeting_label = widgets["QLabel"]("等待识别结果")
        self.greeting_label.setWordWrap(True)

        self.voice_status_label = widgets["QLabel"](
            "语音：自动监听中" if self._voice_controller is not None else "语音：未启用"
        )
        self.voice_status_label.setWordWrap(True)

        self.video_audio_label = widgets["QLabel"](
            "视频音频：自动播放中" if self._video_audio_player is not None else "视频音频：未启用"
        )
        self.video_audio_label.setWordWrap(True)

        self.voice_log = widgets["QTextEdit"]()
        self.voice_log.setReadOnly(True)
        self.voice_log.setMinimumWidth(280)
        self.voice_log.setMinimumHeight(180)
        self.voice_log.setPlaceholderText("语音识别日志会显示在这里")

        left_panel = widgets["QVBoxLayout"]()
        left_panel.addWidget(self.video_label, 1)
        left_panel.addWidget(self.status_label)
        left_panel.addWidget(self.greeting_label)
        left_panel.addWidget(self.voice_status_label)
        left_panel.addWidget(self.video_audio_label)
        left_panel.addWidget(self.voice_log)

        right_panel = widgets["QVBoxLayout"]()
        right_panel.addWidget(widgets["QLabel"]("实时识别日志"))
        right_panel.addWidget(self.overlay_log, 1)

        main_layout = widgets["QHBoxLayout"]()
        main_layout.addLayout(left_panel, 4)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

        self.setStyleSheet(
            """
            QWidget {
                background: #f4f7fb;
                color: #0f172a;
                font-size: 14px;
            }
            QLabel, QTextEdit {
                background: #ffffff;
                border: 1px solid #d7e1ec;
                border-radius: 12px;
                padding: 10px;
            }
            QTextEdit {
                line-height: 1.5;
            }
            """
        )

        self._timer = widgets["QTimer"](self)
        self._timer.timeout.connect(self._tick)  # type: ignore[attr-defined]

    def start(self, interval_ms: int = 30) -> None:
        self._session._input_adapter.open()
        if self._voice_controller is not None:
            self._voice_start_pending = True
            self.voice_status_label.setText("语音：准备监听")
            self._widgets["QTimer"].singleShot(900, self._start_voice_listener)
        if self._video_audio_player is not None and self._video_audio_path is not None:
            self._video_audio_start_pending = True
            self._widgets["QTimer"].singleShot(250, self._start_video_audio)
        self._timer.start(interval_ms)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        self._stop_voice_listener()
        self._stop_video_audio()
        self._session._input_adapter.release()
        if self._speech_service is not None:
            self._speech_service.close()
        if self._voice_controller is not None:
            self._voice_controller.speech_service.close()
        super().closeEvent(event)

    def _tick(self) -> None:
        frame = self._session._input_adapter.read_frame()
        if frame is None:
            self.status_label.setText("视频结束")
            self.close()
            return

        annotated, overlay = self._session.step(frame, self._frame_index)
        self._frame_index += 1
        self._show_frame(annotated)
        self._update_log(overlay)
        self._dispatch_video_wake_events(overlay.frame_index)
        self._poll_voice_events()

    def _show_frame(self, frame: Any) -> None:
        import cv2

        widgets = self._widgets
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = widgets["QImage"](rgb.data, width, height, bytes_per_line, widgets["QImage"].Format_RGB888)
        pixmap = widgets["QPixmap"].fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                widgets["Qt"].KeepAspectRatio,
                widgets["Qt"].SmoothTransformation,
            )
        )

    def _update_log(self, overlay: Any) -> None:
        lines = [
            f"帧 {overlay.frame_index}",
            f"检测 {len(overlay.detections)}",
            f"轨迹 {len(overlay.tracks)}",
        ]
        timings = overlay.timings
        if timings:
            lines.append(
                "time(ms): "
                f"yolo={timings.get('yolo_ms', 0.0):.1f} "
                f"deepsort={timings.get('deepsort_ms', 0.0):.1f} "
                f"identity={timings.get('identity_ms', 0.0):.1f} "
                f"total={timings.get('total_ms', 0.0):.1f}"
            )
            print(
                "[vision] "
                f"frame={overlay.frame_index} "
                f"yolo={timings.get('yolo_ms', 0.0):.1f}ms "
                f"deepsort={timings.get('deepsort_ms', 0.0):.1f}ms "
                f"identity={timings.get('identity_ms', 0.0):.1f}ms "
                f"total={timings.get('total_ms', 0.0):.1f}ms"
            )
        for track_id, label in overlay.labels.items():
            lines.append(f"{track_id}: {label}")
        if overlay.interaction_guess is not None:
            if overlay.interaction_guess.identity is not None:
                lines.append(
                    f"发起者推测: {overlay.interaction_guess.identity.display_name} "
                    f"({overlay.interaction_guess.confidence:.2f}, mock)"
                )
            else:
                lines.append(
                    f"发起者推测: 未能确定 "
                    f"({overlay.interaction_guess.confidence:.2f}, mock)"
                )
        self.overlay_log.setPlainText("\n".join(lines))
        self.status_label.setText(
            f"当前帧 {overlay.frame_index} | 检测 {len(overlay.detections)} | 轨迹 {len(overlay.tracks)}"
        )
        self._update_greeting(overlay)

    def _update_greeting(self, overlay: Any) -> None:
        for track in overlay.tracks:
            greeting = self._session.maybe_greeting_for_track(track.track_id, overlay.frame_index)
            if greeting:
                self.greeting_label.setText(greeting)
                return

        for identity in overlay.identities.values():
            greeting = self._session.greeting_for(identity)
            if greeting:
                self.greeting_label.setText(greeting)
                return

        if overlay.interaction_guess is not None and overlay.interaction_guess.identity is not None:
            self.greeting_label.setText(
                f"推测发起者：{overlay.interaction_guess.identity.display_name}（mock）"
            )
            return

        self.greeting_label.setText("等待稳定身份")

    def _start_voice_listener(self) -> None:
        self._voice_start_pending = False
        if self._voice_controller is None or self._voice_listener is not None:
            return
        print("[voice] listener starting")
        self._voice_listener = VoiceListener(self._voice_controller, self._telemetry or VoiceTelemetrySink())
        self._voice_listener.start()
        self.voice_status_label.setText("语音：自动监听中")

    def _stop_voice_listener(self) -> None:
        if self._voice_listener is None:
            return
        self._voice_listener.stop()
        self._voice_listener.join(timeout=2.0)
        self._voice_listener = None
        self.voice_status_label.setText("语音：已停止")

    def _start_video_audio(self) -> None:
        self._video_audio_start_pending = False
        if self._video_audio_player is None or self._video_audio_path is None:
            return
        if self._video_audio_started:
            return
        print(f"[video-audio] start | {self._video_audio_path}")
        self._video_audio_player.start(self._video_audio_path)
        self._video_audio_started = True
        self.video_audio_label.setText("视频音频：播放中")

    def _stop_video_audio(self) -> None:
        if self._video_audio_player is None:
            return
        self._video_audio_player.stop()
        self._video_audio_started = False
        self.video_audio_label.setText("视频音频：已停止")

    def _poll_voice_events(self) -> None:
        if self._telemetry is None:
            return
        events = self._telemetry.drain()
        if not events:
            return

        lines = self.voice_log.toPlainText().splitlines()
        for event in events:
            lines.append(f"{event.kind.upper()} {event.elapsed_ms:.1f}ms | {event.text}")
        self.voice_log.setPlainText("\n".join(lines[-60:]))

        if self._voice_listener is not None and not self._voice_listener.is_alive():
            self._stop_voice_listener()

    def _dispatch_video_wake_events(self, frame_index: int) -> None:
        if self._video_alignment is None or self._voice_controller is None:
            return
        if self._video_wake_index >= len(self._video_wake_events):
            return

        current_timestamp = self._video_alignment.timestamp_for_frame(frame_index)
        while self._video_wake_index < len(self._video_wake_events):
            event = self._video_wake_events[self._video_wake_index]
            if event.timestamp_seconds > current_timestamp:
                break

            identity_context = self._session.current_interaction_identity()
            command = self._voice_controller.process_transcript(
                event.transcript,
                identity_context=identity_context,
            )
            identity_name = identity_context.display_name if identity_context is not None else "unknown"
            self.voice_log.append(
                f"VIDEO {event.timestamp_seconds:.2f}s | {identity_name} | {event.transcript} | {command.response_text or ''}"
            )
            self._video_wake_index += 1


def build_realtime_window(
    source: int | str | Path = 0,
    model_path: str | Path = "models/yolov8n.pt",
    enrollment_root: str | Path | None = None,
    speak: bool = False,
    video_voice: bool = False,
) -> RealtimeRecognitionWindow:
    resolved_root = default_enrollment_root() if enrollment_root is None else resolve_project_path(enrollment_root)
    telemetry = VoiceTelemetrySink()
    session = build_realtime_session(
        source=source,
        model_path=model_path,
        enrollment_root=resolved_root,
    )
    speech_service = QueuedSpeechService(timing_callback=telemetry.record_tts) if speak else None
    if speech_service is not None:
        speech_service.start()

    voice_controller = _build_voice_controller(
        enrollment_root=resolved_root,
        wake_phrase="你好",
        record_seconds=4.0,
        device=None,
        telemetry=telemetry,
        identity_context_provider=session.current_interaction_identity,
    )

    video_alignment = None
    video_wake_events: list[VideoWakeEvent] = []
    video_audio_path: Path | None = None
    if video_voice and _is_recorded_video_source(source):
        alignment, wake_events = iter_video_wake_events(
            source,
            wake_phrase="你好",
            transcriber=voice_controller.transcriber,
            extractor=VideoAudioExtractor(),
        )
        video_alignment = alignment
        video_wake_events = wake_events
        video_audio_path = alignment.audio_path

    window = RealtimeRecognitionWindow(
        session,
        speech_service=speech_service,
        voice_controller=voice_controller,
        telemetry=telemetry,
        video_alignment=video_alignment,
        video_wake_events=video_wake_events,
        video_audio_path=video_audio_path,
    )
    window.start()
    return window
