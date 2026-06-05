from __future__ import annotations

from pathlib import Path

from familyrobot import enrollment


def test_default_enrollment_root_prefers_current_store_when_present(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    current = root / "data" / "enrollment"
    legacy = root / "src" / "data" / "enrollment"
    current.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (current / enrollment.ENROLLMENT_MANIFEST_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(enrollment, "project_root", lambda: root)

    assert enrollment.default_enrollment_root() == current


def test_default_enrollment_root_returns_current_store_even_without_manifest(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    current = root / "data" / "enrollment"
    legacy = root / "src" / "data" / "enrollment"
    legacy.mkdir(parents=True)
    (legacy / enrollment.ENROLLMENT_MANIFEST_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(enrollment, "project_root", lambda: root)

    assert enrollment.default_enrollment_root() == current
