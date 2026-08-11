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
- **package** — `python -m build` and `twine check`
- **tests** — micromamba + `pytest -m "not network and not manual"`

Live USGS network tests run only on the weekly / manual **Network tests** workflow (secret `API_USGS_PAT`). MkDocs is also deployed to GitHub Pages from `main` via `docs-pages.yml`.

## Local pre-push

`make setup` installs a **pre-push** hook that runs `scripts/ci-local.sh`
(pinned `ruff==0.7.0`, detect-secrets, and a small pytest smoke). A failed
hook blocks `git push`.

On Windows, use Git Bash for `make ci-local`, or run `.\scripts\ci-local.ps1`
in PowerShell. If imports fail because another checkout’s editable install is
active (e.g. missing `redact_secrets`, or coverage at 0%), run
`pip install -e ".[dev]"` from this branch checkout first.

## Test markers

- `network` — live API tests (not on normal PR CI)
- `manual` — opt-in audits
- `integration` — fixture-based pipeline tests (for CI)

Full details are in `CONTRIBUTING.md` at the repository root.
