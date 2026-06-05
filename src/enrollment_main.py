"""Standalone entrypoint for the PySide6 enrollment GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from familyrobot.enrollment import default_enrollment_root
from familyrobot.gui_enrollment import build_enrollment_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family Robot enrollment GUI")
    parser.add_argument(
        "--store-root",
        default=default_enrollment_root(),
        type=Path,
        help="Local enrollment directory root.",
    )
    return parser.parse_args()


def main() -> int:
    from PySide6.QtWidgets import QApplication

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    app = QApplication([])
    window = build_enrollment_window(args.store_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
 
