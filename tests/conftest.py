from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TMP = Path("C:/tmp")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TMP.mkdir(parents=True, exist_ok=True)
for key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[key] = str(TMP)
tempfile.tempdir = str(TMP)
