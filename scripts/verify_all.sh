#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

echo "=== [1/2] Running Python PyTest Suites ==="
"$PYTHON_BIN" -m pytest \
  --import-mode=importlib \
  apps/vision-pipeline/tests/ \
  apps/dl-core/tests/ \
  apps/api-server/tests/ \
  -v --basetemp=.pytest_verify_tmp
TEST_STATUS=$?
rm -rf .pytest_verify_tmp
if [[ $TEST_STATUS -ne 0 ]]; then
  exit $TEST_STATUS
fi

echo "=== [2/2] Running Web Dashboard TypeScript Check ==="
(cd apps/web-dashboard && npx tsc --noEmit)

echo "All monorepo verification checks passed!"
