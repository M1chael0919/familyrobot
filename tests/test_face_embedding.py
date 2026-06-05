from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from familyrobot.face_embedding import FaceEmbedding, FaceEmbeddingExtractor, default_insightface_root
from familyrobot.sample_inputs import project_root


@dataclass
class FakeFace:
    embedding: list[float]
    det_score: float = 0.93


class FakeEmbedder:
    def __init__(self, faces: list[FakeFace]) -> None:
        self.faces = faces
        self.calls = 0

    def get(self, image):
        self.calls += 1
        self.last_image = image
        return self.faces


def test_extractor_returns_first_face_embedding() -> None:
    embedder = FakeEmbedder([FakeFace([0.1, 0.2, 0.3]), FakeFace([0.4, 0.5, 0.6])])
    extractor = FaceEmbeddingExtractor(embedder_loader=lambda: embedder)

    embedding = extractor.extract("face-crop")

    assert embedder.calls == 1
    assert embedder.last_image == "face-crop"
    assert embedding is not None
    assert np.array_equal(
        embedding.vector,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    assert embedding.score == 0.93
    assert embedding.face_count == 2


def test_extractor_prefers_higher_score_face_across_candidate_views() -> None:
    embedder = FakeEmbedder([FakeFace([0.1, 0.2, 0.3], det_score=0.50)])

    def loader() -> FakeEmbedder:
        return embedder

    extractor = FaceEmbeddingExtractor(embedder_loader=loader)

    class _DynamicEmbedder(FakeEmbedder):
        def get(self, image):  # noqa: ANN001
            self.calls += 1
            self.last_image = image
            if self.calls == 1:
                return [FakeFace([0.1, 0.2, 0.3], det_score=0.50)]
            return [FakeFace([0.9, 0.8, 0.7], det_score=0.95)]

    dynamic_embedder = _DynamicEmbedder([])
    extractor = FaceEmbeddingExtractor(embedder_loader=lambda: dynamic_embedder)

    embedding = extractor.extract(np.zeros((120, 80, 3), dtype=np.uint8))

    assert dynamic_embedder.calls >= 2
    assert embedding is not None
    assert np.array_equal(
        embedding.vector,
        np.array([0.9, 0.8, 0.7], dtype=np.float32),
    )
    assert embedding.score == 0.95


def test_extractor_returns_none_when_no_face_found() -> None:
    embedder = FakeEmbedder([])
    extractor = FaceEmbeddingExtractor(embedder_loader=lambda: embedder)

    assert extractor.extract("face-crop") is None


def test_default_insightface_root_is_project_local() -> None:
    assert default_insightface_root() == project_root() / "models" / "insightface"
