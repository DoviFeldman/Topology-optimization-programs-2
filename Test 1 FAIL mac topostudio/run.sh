#!/usr/bin/env bash
# Launch the Topology Optimization Studio UI on this Mac.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No virtualenv found. Run:  bash setup.sh"; exit 1
fi
echo "Topology Optimization Studio -> http://localhost:${PORT:-8501}"
exec .venv/bin/python -m streamlit run app.py --server.port "${PORT:-8501}"
