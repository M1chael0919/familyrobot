from __future__ import annotations

import numpy as np

from familyrobot.identity import (
    IdentityMatcher,
    IdentityReidentifier,
    IdentityTemplate,
    PermanentIdentity,
)


def test_reidentifier_restores_identity_after_disappearance() -> None:
    father = PermanentIdentity(identity_id="father", display_name="Father")
    matcher = IdentityMatcher(
        [
            IdentityTemplate(
                identity=father,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                sample_count=2,
            )
        ]
    )
    reidentifier = IdentityReidentifier(matcher, match_threshold=0.5, disappearance_window=10)

    first = reidentifier.confirm(7, np.array([0.95, 0.05, 0.0], dtype=np.float32), frame_index=1)
    reidentifier.mark_lost(7, frame_index=5)
    second = reidentifier.reidentify(12, np.array([0.96, 0.04, 0.0], dtype=np.float32), frame_index=8)

    assert first is not None
    assert first.identity.identity_id == "father"
    assert second is not None
    assert second.identity.identity_id == "father"
    assert second.restored_from_lost is True
    assert reidentifier.states["father"].track_id == 12


def test_reidentifier_rejects_low_similarity_vectors() -> None:
    father = PermanentIdentity(identity_id="father", display_name="Father")
    matcher = IdentityMatcher(
        [
            IdentityTemplate(
                identity=father,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                sample_count=1,
            )
        ]
    )
    reidentifier = IdentityReidentifier(matcher, match_threshold=0.8, disappearance_window=10)

    assert reidentifier.reidentify(
        12,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        frame_index=8,
    ) is None


def test_reidentifier_default_threshold_accepts_mid_confidence_match() -> None:
    father = PermanentIdentity(identity_id="father", display_name="Father")
    matcher = IdentityMatcher(
        [
            IdentityTemplate(
                identity=father,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                sample_count=1,
            )
        ]
    )
    reidentifier = IdentityReidentifier(matcher, disappearance_window=10)

    event = reidentifier.reidentify(
        12,
        np.array([0.7, 0.71414286, 0.0], dtype=np.float32),
        frame_index=8,
    )

    assert event is not None
    assert event.identity.identity_id == "father"
