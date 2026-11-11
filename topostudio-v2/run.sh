#!/usr/bin/env bash
# Launch the studio, then open http://localhost:8501
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run  bash setup.sh  first."; exit 1; }
# one BLAS thread: the solver's hot loops are single-threaded and extra BLAS
# threads only fight each other
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec .venv/bin/python -m streamlit run app.py --server.port "${PORT:-8501}"
