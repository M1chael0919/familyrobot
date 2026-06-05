from __future__ import annotations

from pathlib import Path

import pytest

from familyrobot.sample_inputs import bundled_sample_input


def test_bundled_sample_input_resolves_known_aliases() -> None:
    sample = bundled_sample_input("sample")
    test_video = bundled_sample_input("test-video")

    assert sample.exists()
    assert test_video.exists()
    assert sample.suffix == ".mp4"
    assert test_video.suffix == ".mp4"


def test_bundled_sample_input_rejects_unknown_alias() -> None:
    with pytest.raises(ValueError):
        bundled_sample_input("unknown")
