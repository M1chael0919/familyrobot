from __future__ import annotations

import numpy as np

from familyrobot.identity import (
    IdentityMatch,
    PermanentIdentity,
    UnknownIdentity,
    UnknownPersonHandler,
    UnknownPersonPolicy,
)


def test_unknown_person_handler_returns_unknown_when_no_match() -> None:
    handler = UnknownPersonHandler()

    result = handler.resolve(None)

    assert result == UnknownIdentity(reason="no_match")


def test_unknown_person_handler_returns_unknown_when_low_confidence() -> None:
    handler = UnknownPersonHandler(UnknownPersonPolicy(match_threshold=0.8))
    match = IdentityMatch(
        identity=PermanentIdentity(identity_id="father", display_name="Father"),
        score=0.6,
        sample_count=2,
    )

    result = handler.resolve(match)

    assert result == UnknownIdentity(reason="low_confidence", score=0.6)


def test_unknown_person_handler_returns_identity_when_confident() -> None:
    identity = PermanentIdentity(identity_id="mother", display_name="Mother")
    handler = UnknownPersonHandler()
    match = IdentityMatch(identity=identity, score=0.95, sample_count=2)

    result = handler.resolve(match)

    assert result == identity


def test_unknown_person_handler_default_threshold_is_more_permissive() -> None:
    identity = PermanentIdentity(identity_id="father", display_name="Father")
    handler = UnknownPersonHandler()
    match = IdentityMatch(identity=identity, score=0.7, sample_count=1)

    result = handler.resolve(match)

    assert result == identity
