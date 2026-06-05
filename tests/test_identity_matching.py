from __future__ import annotations

import numpy as np
import pytest

from familyrobot.identity import (
    IdentityMatcher,
    IdentityMatch,
    IdentityTemplate,
    PermanentIdentity,
)


def test_identity_matcher_returns_best_cosine_match() -> None:
    father = PermanentIdentity(identity_id="father", display_name="Father")
    mother = PermanentIdentity(identity_id="mother", display_name="Mother")
    matcher = IdentityMatcher(
        [
            IdentityTemplate(
                identity=father,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                sample_count=2,
            ),
            IdentityTemplate(
                identity=mother,
                embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                sample_count=1,
            ),
        ]
    )

    match = matcher.match(np.array([0.8, 0.2, 0.0], dtype=np.float32))

    assert match is not None
    assert match.identity == father
    assert match.sample_count == 2
    assert match.score == pytest.approx(0.9701425, rel=1e-6)


def test_identity_matcher_returns_none_for_empty_gallery() -> None:
    matcher = IdentityMatcher([])

    assert matcher.match(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is None
