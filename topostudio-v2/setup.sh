#!/usr/bin/env bash
# One-time setup (macOS / Linux).
set -e
cd "$(dirname "$0")"
PY=${PY:-$(command -v python3.12 || command -v python3)}
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done.  Now run:  bash run.sh"
