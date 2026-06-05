from __future__ import annotations

from datetime import datetime, timezone

from familyrobot.enrollment import (
    ENROLLMENT_FORMAT_VERSION,
    ENROLLMENT_IMAGES_DIR,
    ENROLLMENT_MANIFEST_NAME,
    EnrollmentManifest,
    EnrollmentRecord,
    EnrollmentStore,
)


def test_enrollment_record_round_trip() -> None:
    record = EnrollmentRecord(
        identity_id="father",
        display_name="Father",
        role="parent",
        sample_images=["father/1.jpg", "father/2.jpg"],
        metadata={"source": "manual"},
        enrolled_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    )

    data = record.to_dict()
    restored = EnrollmentRecord.from_dict(data)

    assert restored == record


def test_manifest_round_trip_and_version_defaults() -> None:
    manifest = EnrollmentManifest(
        records=[
            EnrollmentRecord(
                identity_id="mother",
                display_name="Mother",
                sample_images=["mother/a.jpg"],
            )
        ]
    )

    data = manifest.to_dict()
    restored = EnrollmentManifest.from_dict(data)

    assert data["version"] == ENROLLMENT_FORMAT_VERSION
    assert restored.version == ENROLLMENT_FORMAT_VERSION
    assert restored.records == manifest.records


def test_store_creates_expected_local_structure(tmp_path) -> None:
    store = EnrollmentStore(tmp_path / "enrollment")
    record = EnrollmentRecord(
        identity_id="grandpa",
        display_name="Grandpa",
        sample_images=["grandpa/1.jpg"],
    )
    manifest = EnrollmentManifest(records=[record])

    store.save(manifest)
    restored = store.load()

    assert store.manifest_path.name == ENROLLMENT_MANIFEST_NAME
    assert store.images_dir().name == ENROLLMENT_IMAGES_DIR
    assert store.identity_dir("grandpa") == tmp_path / "enrollment" / "images" / "grandpa"
    assert restored.records == [record]
