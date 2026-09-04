#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-third_party/safety-gymnasium-main}"
ZIP_URL="https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip"

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 8):
    raise SystemExit(
        "The latest Safety-Gymnasium main-branch install is expected to run with Python 3.8. "
        "Create and activate the project environment first:\n"
        "  conda env create -f environment.yml\n"
        "  conda activate dmdp-safety-main"
    )
PY

python - "$SOURCE_DIR" "$ZIP_URL" <<'PY'
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
zip_url = sys.argv[2]
workspace = source_dir.parent
archive = workspace / "safety-gymnasium-main.zip"

workspace.mkdir(parents=True, exist_ok=True)
if source_dir.exists():
    shutil.rmtree(source_dir)

print(f"Downloading {zip_url}")
urllib.request.urlretrieve(zip_url, archive)

print(f"Extracting to {workspace}")
with zipfile.ZipFile(archive) as zf:
    zf.extractall(workspace)

extracted = workspace / "safety-gymnasium-main"
if extracted != source_dir:
    if source_dir.exists():
        shutil.rmtree(source_dir)
    extracted.rename(source_dir)

print(source_dir)
PY

python -m pip install -e "$SOURCE_DIR"
python - <<'PY'
import safety_gymnasium

print("Safety-Gymnasium imported from:", safety_gymnasium.__file__)
print("Safety-Gymnasium version:", getattr(safety_gymnasium, "__version__", "unknown"))
PY
