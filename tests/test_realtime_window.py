from __future__ import annotations

from pathlib import Path

from familyrobot.enrollment import EnrollmentManifest, EnrollmentRecord, EnrollmentStore
from familyrobot.realtime_window import _build_voice_controller
from familyrobot.realtime_gui import VoiceTelemetrySink


def test_build_voice_controller_uses_enrollment_and_wake_phrase(tmp_path: Path) -> None:
    store = EnrollmentStore(tmp_path / "enrollment")
    store.save(
        EnrollmentManifest(
            records=[
                EnrollmentRecord(
                    identity_id="father",
                    display_name="爸爸",
                    role="父亲",
                )
            ]
        )
    )

    telemetry = VoiceTelemetrySink()
    controller = _build_voice_controller(
        enrollment_root=store.root,
        wake_phrase="你好小家",
        record_seconds=2.5,
        device=None,
        telemetry=telemetry,
    )

    command = controller.router.route("你好小家 爸爸")

    assert controller.transcriber.config.record_seconds == 2.5
    assert controller.router.wake_phrase == "你好小家"
    assert command.activated is True
    assert command.target_identity_id == "father"
    assert command.target_display_name == "爸爸"
