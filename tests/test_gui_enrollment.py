from __future__ import annotations

from pathlib import Path

from familyrobot.enrollment import EnrollmentStore
from familyrobot.gui_enrollment import EnrollmentRegistrationService, EnrollmentSubmission


def test_enrollment_service_writes_manifest_and_copies_samples(tmp_path) -> None:
    store = EnrollmentStore(tmp_path / "enrollment")
    service = EnrollmentRegistrationService(store)

    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"sample-bytes")

    record = service.enroll(
        EnrollmentSubmission(
            identity_id="father",
            display_name="Father",
            role="parent",
            sample_files=(sample,),
        )
    )

    copied = store.root / "images" / "father" / "sample.jpg"
    assert copied.exists()
    assert copied.read_bytes() == b"sample-bytes"
    assert record.identity_id == "father"
    assert record.display_name == "Father"
    assert record.role == "parent"
    assert record.sample_images == ["images/father/sample.jpg"]
    assert store.load().records[0].identity_id == "father"


def test_enrollment_service_deletes_member_and_samples(tmp_path) -> None:
    store = EnrollmentStore(tmp_path / "enrollment")
    service = EnrollmentRegistrationService(store)

    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"sample-bytes")

    service.enroll(
        EnrollmentSubmission(
            identity_id="father",
            display_name="Father",
            role="parent",
            sample_files=(sample,),
        )
    )

    deleted = service.delete("father")

    assert deleted is True
    assert store.load().records == []
    assert not (store.root / "images" / "father").exists()
    assert not (store.root / "images" / "father" / "sample.jpg").exists()
