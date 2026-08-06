#!/usr/bin/env bash
# Fast local gate for pre-push / `make ci-local`.
# Not the full CI matrix (no conda rebuild, no Windows/macOS, no full suite).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> ruff"
ruff check src bin

echo "==> detect-secrets"
git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline

echo "==> fast pytest subset (package imports)"
pytest tests/test_package_imports.py -q \
  -m "not network and not manual" \
  -o addopts=""

echo "ci-local passed."
