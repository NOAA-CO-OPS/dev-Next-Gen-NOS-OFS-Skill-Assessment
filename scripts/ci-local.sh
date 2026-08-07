#!/usr/bin/env bash
# Fast local gate for pre-push / `make ci-local`.
# Not the full CI matrix (no conda rebuild, no Windows/macOS, no full suite).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Preflight: this gate needs the project dev environment (ruff, pytest,
# and an importable ofs_skill). Fail early with guidance instead of a
# wall of ModuleNotFoundError from pytest.
if ! python -c "import ofs_skill" >/dev/null 2>&1; then
  echo "ci-local: cannot import ofs_skill with python=$(command -v python || echo '<not found>')." >&2
  echo "Activate the dev environment first (e.g. 'conda activate ofs_dps'," >&2
  echo "or run 'make setup' / 'pip install -e .[dev]')." >&2
  exit 1
fi

echo "==> ruff"
ruff check src bin

echo "==> detect-secrets"
git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline

echo "==> fast pytest subset (package imports)"
pytest tests/test_package_imports.py -q \
  -m "not network and not manual" \
  -o addopts=""

echo "ci-local passed."
