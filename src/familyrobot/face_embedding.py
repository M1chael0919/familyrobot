"""Face embedding extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from familyrobot.sample_inputs import project_root


FaceImage = Any
EmbeddingVector = np.ndarray


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """A single face embedding result."""

    vector: EmbeddingVector
    score: float | None = None
    face_count: int = 1


class FaceEmbedderLike(Protocol):
    """Minimal interface for a face embedder."""

    def get(self, image: FaceImage) -> list[Any]: ...


EmbedderLoader = Callable[[], FaceEmbedderLike]


def default_insightface_root() -> Path:
    """Return the project-local InsightFace model cache root."""

    return project_root() / "models" / "insightface"


class FaceEmbeddingExtractor:
    """Extract face embeddings from an image crop."""

    def __init__(self, embedder_loader: EmbedderLoader | None = None) -> None:
        self._embedder_loader = embedder_loader or self._load_insightface_embedder
        self._embedder: FaceEmbedderLike | None = None

    def load(self) -> "FaceEmbeddingExtractor":
        """Load the backend embedder if needed."""

        if self._embedder is None:
            self._embedder = self._embedder_loader()
        return self

    def extract(self, image: FaceImage) -> FaceEmbedding | None:
        """Extract the first available face embedding from the image."""

        if self._embedder is None:
            self.load()

        assert self._embedder is not None
        best_vector: np.ndarray | None = None
        best_score = -1.0
        best_face_count = 0

        for view in self._candidate_views(image):
            faces = self._embedder.get(view)
            if not faces:
                continue
            for face in faces:
                embedding = getattr(face, "embedding", None)
                if embedding is None:
                    continue
                score = float(getattr(face, "det_score", 0.0) or 0.0)
                if score > best_score:
                    best_vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                    best_score = score
                    best_face_count = len(faces)

        if best_vector is None:
            return None

        return FaceEmbedding(
            vector=best_vector,
            score=best_score if best_score >= 0.0 else None,
            face_count=best_face_count,
        )

    @staticmethod
    def _load_insightface_embedder() -> FaceEmbedderLike:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "insightface is required for face embedding extraction."
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", root=str(default_insightface_root()), providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

    @staticmethod
    def _candidate_views(image: FaceImage) -> list[FaceImage]:
        if not hasattr(image, "shape"):
            return [image]

        array = np.asarray(image)
        if array.ndim < 2:
            return [image]

        views: list[FaceImage] = [array]
        height, width = array.shape[:2]

        top_limit = max(1, int(height * 0.7))
        if top_limit < height:
            views.append(array[:top_limit, :])

        try:
            import cv2
        except ImportError:  # pragma: no cover - depends on environment
            return views

        longest = max(height, width)
        target_longest = 640
        if longest > 0 and longest < target_longest:
            scale = target_longest / float(longest)
            resized = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            views.append(resized)
            if top_limit < height:
                top_resized = cv2.resize(
                    array[:top_limit, :],
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                views.append(top_resized)

        return views


def load_image(path: str | Path) -> Any:
    """Load an image with a Windows-friendly Unicode fallback."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required to read enrollment images.") from exc

    path = Path(path)
    path_text = str(path)
    if path_text.isascii():
        image = cv2.imread(path_text)
        if image is not None:
            return image

    try:
        buffer = np.fromfile(path_text, dtype=np.uint8)
    except OSError:
        data = path.read_bytes()
        buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image
