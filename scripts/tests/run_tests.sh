#!/usr/bin/env bash
# Run the script tests against the stub server. Needs only python3 (3.9+); no model, no network.
#   bash scripts/tests/run_tests.sh
# Optional: MLX_PYTHON=<mlx-env>/bin/python also runs sync_bench.py once as a single process.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"

echo "== unit and end-to-end tests (stub server on 127.0.0.1)"
"$PY" -m unittest discover -s scripts/tests -p 'test_*.py' -v

echo "== sync_bench.py"
if [ -n "${MLX_PYTHON:-}" ]; then
  "$MLX_PYTHON" scripts/sync_bench.py --rows 1 --steps 30 --warmup 3
else
  "$PY" scripts/sync_bench.py --help > /dev/null && echo "sync_bench.py --help OK (set MLX_PYTHON to run it)"
fi
echo "== all tests passed"
