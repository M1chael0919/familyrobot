from __future__ import annotations

from datetime import datetime

from familyrobot.identity import (
    IdentityAssignment,
    IdentityLinkState,
    PermanentIdentity,
    TrackIdentityLink,
)


def test_track_identity_link_defaults_keep_track_and_identity_separate() -> None:
    link = TrackIdentityLink(track_id=7)

    assert link.track_id == 7
    assert link.identity_id is None
    assert link.state is IdentityLinkState.UNKNOWN
    assert link.confidence == 0.0
    assert link.first_seen_frame is None
    assert link.last_seen_frame is None
    assert link.confirmed_at_frame is None


def test_permanent_identity_records_stable_identity_metadata() -> None:
    enrolled_at = datetime(2026, 6, 1, 12, 0, 0)
    identity = PermanentIdentity(
        identity_id="father",
        display_name="Father",
        role="parent",
        enrolled_at=enrolled_at,
        metadata={"source": "manual"},
    )

    assert identity.identity_id == "father"
    assert identity.display_name == "Father"
    assert identity.role == "parent"
    assert identity.enrolled_at == enrolled_at
    assert identity.metadata == {"source": "manual"}


def test_identity_assignment_can_reference_link_and_identity() -> None:
    identity = PermanentIdentity(identity_id="mother", display_name="Mother")
    link = TrackIdentityLink(
        track_id=3,
        identity_id="mother",
        state=IdentityLinkState.CONFIRMED,
        confidence=0.87,
        first_seen_frame=12,
        last_seen_frame=18,
        confirmed_at_frame=14,
    )
    assignment = IdentityAssignment(
        track_id=3,
        identity=identity,
        link=link,
        assigned_at=datetime(2026, 6, 1, 12, 30, 0),
    )

    assert assignment.track_id == 3
    assert assignment.identity == identity
    assert assignment.link == link
    assert assignment.assigned_at == datetime(2026, 6, 1, 12, 30, 0)
