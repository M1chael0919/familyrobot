from __future__ import annotations

import numpy as np

from familyrobot.face_embedding import load_image


def test_load_image_reads_unicode_path(tmp_path) -> None:
    path = tmp_path / "图片1.png"
    import cv2

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())

    loaded = load_image(path)

    assert loaded is not None
    assert loaded.shape[:2] == (10, 10)


def test_load_image_skips_imread_for_unicode_path(tmp_path, monkeypatch) -> None:
    path = tmp_path / "图片2.png"
    import cv2

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())

    called = False

    def _imread(_: str):  # noqa: ANN001
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(cv2, "imread", _imread)

    loaded = load_image(path)

    assert loaded is not None
    assert loaded.shape[:2] == (10, 10)
    assert called is False
