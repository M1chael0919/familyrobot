from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from familyrobot.enrollment import EnrollmentManifest, EnrollmentRecord, EnrollmentStore
from familyrobot.face_embedding import FaceEmbedding
from familyrobot.identity import IdentityMatcher


class FakeExtractor:
    def __init__(self, mapping: dict[str, list[float] | None]) -> None:
        self.mapping = mapping

    def extract(self, image):
        vector = self.mapping.get(str(image))
        if vector is None:
            return None
        return FaceEmbedding(vector=np.array(vector, dtype=np.float32))


def test_matcher_builds_template_from_enrollment_store(tmp_path, monkeypatch) -> None:
    store = EnrollmentStore(tmp_path / "enrollment")
    record = EnrollmentRecord(
        identity_id="father",
        display_name="Father",
        sample_images=["images/father/1.jpg"],
        enrolled_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    store.save(EnrollmentManifest(records=[record]))

    sample_path = store.root / "images" / "father" / "1.jpg"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(b"sample")

    extractor = FakeExtractor({str(sample_path): [1.0, 0.0, 0.0]})

    import cv2

    monkeypatch.setattr(cv2, "imread", lambda path: path)

    matcher = IdentityMatcher.from_enrollment(store, extractor)

    assert len(matcher.templates) == 1
    template = matcher.templates[0]
    assert template.identity.identity_id == "father"
    assert template.sample_count == 1
    assert np.allclose(template.embedding, np.array([1.0, 0.0, 0.0], dtype=np.float32))
