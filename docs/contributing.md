# Contributing

## Quick start

```bash
make setup
conda activate ofs_dps
make ci-local
```

## What CI runs

On every pull request, GitHub Actions runs these jobs in parallel:

- **lint** — `ruff` and `detect-secrets`
- **types** — `mypy` on `src/ofs_skill`
- **docs** — `mkdocs build --strict`
- **tests** — micromamba + `pytest -m "not network and not manual"`

## Local pre-push

`make setup` installs a **pre-push** hook that runs `scripts/ci-local.sh` (ruff, detect-secrets, and a small pytest smoke). A failed hook blocks `git push`.

## Test markers

- `network` — live API tests (not on normal PR CI)
- `manual` — opt-in audits
- `integration` — fixture-based pipeline tests (for CI)

Full details are in `CONTRIBUTING.md` at the repository root.
