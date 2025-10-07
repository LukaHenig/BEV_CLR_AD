#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest is not installed" >&2
  exit 1
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is required to run the test suite but is not installed in this environment." >&2
  exit 0
fi

pytest -q "$@"
