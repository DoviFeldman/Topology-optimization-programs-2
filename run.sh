#!/usr/bin/env bash
# One command to launch the web UI (works in Codespaces, macOS, or Linux).
set -e
cd "$(dirname "$0")"

# Install dependencies if Streamlit isn't available yet.
if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Installing dependencies…"
  pip install -r requirements.txt
fi

echo "Starting Topology Optimization Studio → http://localhost:8501"
exec python -m streamlit run app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0
