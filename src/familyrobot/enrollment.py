"""Local enrollment data format for family identities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ENROLLMENT_MANIFEST_NAME = "enrollment.json"
ENROLLMENT_IMAGES_DIR = "images"
ENROLLMENT_FORMAT_VERSION = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_enrollment_root() -> Path:
    """Return the most likely active enrollment store root."""

    root = project_root()
    current = root / "data" / "enrollment"

    if (current / ENROLLMENT_MANIFEST_NAME).exists():
        return current
    return current


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_iso8601(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _from_iso8601(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


@dataclass(frozen=True, slots=True)
class EnrollmentRecord:
    """One enrolled family member and its local samples."""

    identity_id: str
    display_name: str
    role: str | None = None
    sample_images: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    enrolled_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "role": self.role,
            "sample_images": list(self.sample_images),
            "metadata": dict(self.metadata),
            "enrolled_at": _to_iso8601(self.enrolled_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EnrollmentRecord":
        return cls(
            identity_id=str(data["identity_id"]),
            display_name=str(data["display_name"]),
            role=(str(data["role"]) if data.get("role") is not None else None),
            sample_images=[str(item) for item in data.get("sample_images", [])],
            metadata={str(k): str(v) for k, v in dict(data.get("metadata", {})).items()},
            enrolled_at=_from_iso8601(
                str(data["enrolled_at"]) if data.get("enrolled_at") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class EnrollmentManifest:
    """JSON manifest stored at the root of the enrollment directory."""

    version: int = ENROLLMENT_FORMAT_VERSION
    created_at: datetime = field(default_factory=_utc_now)
    records: list[EnrollmentRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created_at": _to_iso8601(self.created_at),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EnrollmentManifest":
        return cls(
            version=int(data.get("version", ENROLLMENT_FORMAT_VERSION)),
            created_at=_from_iso8601(str(data["created_at"])) if data.get("created_at") else _utc_now(),
            records=[
                EnrollmentRecord.from_dict(record)
                for record in data.get("records", [])
            ],
        )


class EnrollmentStore:
    """File-based enrollment store."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest_path(self) -> Path:
        return self._root / ENROLLMENT_MANIFEST_NAME

    def images_dir(self) -> Path:
        return self._root / ENROLLMENT_IMAGES_DIR

    def identity_dir(self, identity_id: str) -> Path:
        return self.images_dir() / identity_id

    def ensure_structure(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self.images_dir().mkdir(parents=True, exist_ok=True)

    def save(self, manifest: EnrollmentManifest) -> None:
        self.ensure_structure()
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load(self) -> EnrollmentManifest:
        if not self.manifest_path.exists():
            return EnrollmentManifest()
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return EnrollmentManifest.from_dict(data)

    def add_sample_image(self, identity_id: str, file_name: str) -> Path:
        self.ensure_structure()
        image_dir = self.identity_dir(identity_id)
        image_dir.mkdir(parents=True, exist_ok=True)
        return image_dir / file_name
