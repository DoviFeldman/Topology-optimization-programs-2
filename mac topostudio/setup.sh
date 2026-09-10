#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon).
set -e
cd "$(dirname "$0")"
PY=${PY:-/opt/homebrew/bin/python3.12}
[ -x "$PY" ] || PY=$(command -v python3.12 || command -v python3)
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# PyTopo3D: install without pypardiso, which needs Intel MKL (no arm64 build).
[ -d external/PyTopo3D ] || git clone --depth 1 https://github.com/jihoonkim888/PyTopo3D.git external/PyTopo3D
.venv/bin/pip install --no-deps -e external/PyTopo3D
# ToPy is vendored under external/ToPy, already ported to Python 3; it is used
# via engines/topy_compat.py together with external/pysparse_shim.
echo "Done.  Now run:  bash run.sh"
