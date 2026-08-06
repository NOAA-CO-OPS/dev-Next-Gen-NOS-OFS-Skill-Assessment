# Contributing

## Developer setup

Recommended path (conda + Makefile):

```bash
# Requires conda/mamba with CONDA_EXE set (after conda init)
make setup
conda activate ofs_dps
```

What `make setup` does:
1. Create/update the `ofs_dps` conda env from `environment.yml`
2. `pip install -e ".[dev]"`
3. Install **pre-commit** and **pre-push** git hooks

Manual equivalent:

```bash
conda env create -n ofs_dps -f environment.yml   # or: conda env update -n ofs_dps -f environment.yml --prune
conda activate ofs_dps
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

Python support: **3.11+** (`requires-python` in `pyproject.toml`).

## Local checks before push

Pre-push runs a **fast** subset only (not the full CI matrix):

```bash
make ci-local
# or
bash scripts/ci-local.sh
```

This runs:
- `ruff check src bin`
- `detect-secrets` against `.secrets.baseline`
- a small pytest subset (`tests/test_package_imports.py`)

If `ci-local` fails, the push is blocked. Fix locally and try again.
You can bypass with `git push --no-verify` (not recommended).

## Tests and markers

```bash
# Default local suite (includes coverage HTML via pyproject addopts)
pytest

# Match CI (unit + integration; no live network / manual audits; parallel)
pytest -m "not network and not manual" -n auto -o addopts="" --cov=src/ofs_skill --cov-report=term-missing

# Live USGS network tests (needs API_USGS_PAT; also run via network-tests.yml)
API_USGS_PAT=... pytest -m network -o addopts=""
```

| Marker | Meaning | When it runs |
|--------|---------|--------------|
| *(none)* | Normal unit tests | Every PR / local |
| `integration` | Fixture/mock pipeline tests (no live network) | Every PR |
| `network` | Live API / network access | Scheduled / `workflow_dispatch` only (secret `API_USGS_PAT`) |
| `manual` | Opt-in live audits | Opt-in only |

Offline mocks and pipeline sample data: `tests/helpers/api_mocks.py`, `tests/fixtures/` (see `tests/fixtures/README.md`).

## CI expectations

GitHub Actions runs **in parallel** on PRs (`.github/workflows/ci.yml`):

| Job | What |
|-----|------|
| **lint** | `ruff` + `detect-secrets` (pip-only; no full conda env) |
| **types** | `mypy` on `src/ofs_skill` (pip-only) |
| **docs** | `mkdocs build --strict` (pip-only) |
| **tests** | micromamba from `environment.yml` + pytest (`not network and not manual`) |

Live network tests: `.github/workflows/network-tests.yml` (weekly cron + manual), using encrypted secret `API_USGS_PAT`. Do **not** commit keys to `conf/api_keys.conf`.

Caching:
- Micromamba env/downloads keyed by OS + `environment.yml` hash (`-v1` bust suffix)
- Pip cache for lint/types/docs
- Pre-commit cache keyed by `.pre-commit-config.yaml`

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve          # local preview
mkdocs build --strict # same gate as CI
```

## Pull requests

1. Branch from an up-to-date `main`
2. Run `make ci-local` (or rely on pre-push)
3. Open a PR — wait for lint, types, docs, and tests
4. Keep PRs focused; do not commit secrets (`conf/api_keys.conf` is gitignored)
