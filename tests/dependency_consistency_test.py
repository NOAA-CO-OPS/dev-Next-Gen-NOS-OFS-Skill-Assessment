"""Guard the agreement between pyproject.toml and environment.yml.

The two files describe the same install from different angles.
``environment.yml`` builds the conda environment that CI and the
recommended install (INSTALL_WINDOWS.md Method A) both start from;
``pyproject.toml`` declares what ``pip install -e .`` adds on top, and on
the pip-only path (Method C) it is the *only* thing constraining
versions.

They drifted apart once already: sixteen of the twenty-one shared
packages disagreed, with pyproject floors old enough to permit
combinations the code cannot run on (numpy 1.20 beside geopandas 0.9),
and an unbounded ``plotly>=5.0.0`` that resolved to a major release on
which the package raised at import-adjacent call time. These tests fail
if that starts happening again.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / 'pyproject.toml'
ENVIRONMENT = REPO_ROOT / 'environment.yml'

# Majors that have broken this codebase before. Each needs an upper bound
# so an upstream release cannot enter the dependency set unannounced.
MUST_HAVE_UPPER_BOUND = ('numpy', 'pandas', 'plotly', 'matplotlib')


def _normalize(name: str) -> str:
    """PEP 503 style normalization so ``scikit_learn`` == ``scikit-learn``."""
    return re.sub(r'[-_.]+', '-', name).lower()


def _pyproject_requirements() -> dict[str, Requirement]:
    """Every non-URL dependency, runtime and extras alike.

    The extras drift for exactly the same reason the runtime block does --
    ``dev`` once asked for ``pytest>=6.2.0`` while CI ran 7.4 -- so they
    are held to the same rule rather than left unguarded.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    project = data['project']
    raw_deps = list(project['dependencies'])
    for extra_deps in project.get('optional-dependencies', {}).values():
        raw_deps.extend(extra_deps)

    reqs = {}
    for raw in raw_deps:
        req = Requirement(raw)
        # Git/URL pins carry no version specifier to compare against; the
        # comments in both files already tie them to the same commit SHA.
        if req.url is None:
            # 'all' repeats 'spatial'; identical entries collapse harmlessly.
            reqs[_normalize(req.name)] = req
    return reqs


def _environment_specs() -> dict[str, SpecifierSet]:
    """Conda specs from environment.yml, as PEP 440 specifier sets.

    Parsed by hand rather than with PyYAML so a missing optional test
    dependency cannot turn this guard into a skip. Entries without a
    version (``- scipy``) are recorded with an empty specifier, which
    constrains nothing and is therefore always satisfied.
    """
    specs: dict[str, SpecifierSet] = {}
    for line in ENVIRONMENT.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped.startswith('- ') or stripped.startswith('- --'):
            continue
        entry = stripped[2:].split('#')[0].strip()
        if not entry or entry.endswith(':') or entry.startswith('git+'):
            continue
        entry = entry.split('::')[-1]  # drop a conda channel prefix
        match = re.match(r'([A-Za-z0-9_.-]+)\s*(.*)$', entry)
        if match is None:
            continue
        name, constraint = match.group(1), match.group(2).strip()
        if name in ('python', 'pip'):
            continue
        specs[_normalize(name)] = _conda_spec_to_specifier(constraint)
    return specs


def _conda_spec_to_specifier(constraint: str) -> SpecifierSet:
    """Translate the conda spellings used in environment.yml.

    conda writes ``=3.8.*`` and ``=5.*`` where PEP 440 wants ``==3.8.*``,
    and it also accepts ``>=3.3.*``, which PEP 440 rejects outright -- a
    wildcard is only legal with ``==``/``!=``. Normalize both forms; an
    unparseable leftover yields an empty specifier so one odd line cannot
    take the whole guard down.
    """
    if not constraint:
        return SpecifierSet('')

    pieces = []
    for part in constraint.split(','):
        part = part.strip()
        if not part:
            continue
        match = re.match(r'(==|!=|<=|>=|<|>|~=|=)?\s*(.+)$', part)
        if match is None:
            continue
        operator, version = match.group(1) or '=', match.group(2).strip()
        if operator == '=':
            operator = '=='
        if version.endswith('.*') and operator not in ('==', '!='):
            # ">=3.3.*" is conda for ">=3.3"; the wildcard adds nothing.
            version = version[:-2]
        pieces.append(f'{operator}{version}')

    try:
        return SpecifierSet(','.join(pieces))
    except Exception:  # pragma: no cover - defensive
        return SpecifierSet('')


SHARED = sorted(set(_pyproject_requirements()) & set(_environment_specs()))


def test_the_two_files_still_share_packages():
    """A parsing regression that found nothing would silently pass."""
    assert len(SHARED) > 15, (
        f'only {len(SHARED)} shared packages found; the parser is probably '
        'broken rather than the files having genuinely diverged'
    )


@pytest.mark.parametrize('package', SHARED)
def test_pyproject_floor_is_not_below_what_we_test(package):
    """pyproject must not permit a version older than environment.yml pins.

    This is the drift that mattered: pyproject said ``numpy>=1.20.0``
    while environment.yml required ``>=2.0.0``, so a pip-only install
    could resolve a numpy the code has never run on.
    """
    env_spec = _environment_specs()[package]
    if not str(env_spec):
        pytest.skip(f'{package} is unpinned in environment.yml')

    pyproject_spec = _pyproject_requirements()[package].specifier

    # The lowest version environment.yml would accept. For an ``==2.2.*``
    # style pin that is 2.2; for ``>=1.0.0`` it is 1.0.0.
    floors = [
        Version(s.version.rstrip('.*'))
        for s in env_spec
        if s.operator in ('>=', '==', '~=')
    ]
    if not floors:
        pytest.skip(f'{package} has no lower bound in environment.yml')
    env_floor = min(floors)

    # Compare the declared floors directly rather than probing a sample
    # version. Probing is what a first cut of this test did, and it let
    # numpy through: with environment.yml at >=2.0.0 the probe was 1.0,
    # which pyproject's >=1.20.0 correctly rejected, so the test passed
    # while 1.20 through 1.99 -- the versions that actually matter -- were
    # still permitted.
    pyproject_floors = [
        Version(spec.version.rstrip('.*'))
        for spec in pyproject_spec
        if spec.operator in ('>=', '==', '~=')
    ]
    assert pyproject_floors, (
        f'{package}: pyproject.toml sets no lower bound, so it permits '
        f'arbitrarily old releases while environment.yml requires '
        f'{env_spec}.'
    )
    assert min(pyproject_floors) >= env_floor, (
        f'{package}: pyproject.toml allows {min(pyproject_floors)} and up, '
        f'but environment.yml requires {env_floor} and up ({env_spec}). '
        f'A pip-only install could resolve a version we never test.'
    )


@pytest.mark.parametrize('package', SHARED)
def test_the_two_files_do_not_contradict_each_other(package):
    """Some version must satisfy both files at once.

    A pyproject cap below the environment.yml floor (or vice versa) would
    make the recommended install unsatisfiable, which is worse than the
    drift this guard replaced.
    """
    env_spec = _environment_specs()[package]
    pyproject_spec = _pyproject_requirements()[package].specifier
    candidates = [
        v for v in _plausible_versions(env_spec, pyproject_spec)
        if env_spec.contains(v) and pyproject_spec.contains(v)
    ]
    assert candidates, (
        f'{package}: environment.yml ({env_spec or "unpinned"}) and '
        f'pyproject.toml ({pyproject_spec or "unpinned"}) cannot both be '
        'satisfied by any version.'
    )


def _plausible_versions(*specs):
    """Version samples drawn from the bounds the specs mention."""
    seen = set()
    for spec in specs:
        for s in spec:
            base = s.version.rstrip('.*')
            seen.add(base)
            try:
                v = Version(base)
            except Exception:  # pragma: no cover - defensive
                continue
            seen.add(f'{v.major}.{v.minor}.0')
            seen.add(f'{v.major}.{v.minor}.99')
            seen.add(f'{v.major}.{v.minor + 1}.0')
    out = []
    for text in seen:
        try:
            out.append(Version(text))
        except Exception:  # pragma: no cover - defensive
            pass
    return out


@pytest.mark.parametrize('package', MUST_HAVE_UPPER_BOUND)
def test_breaking_majors_stay_capped(package):
    """The libraries that have broken us keep an upper bound.

    Without one, an upstream major lands in a pip-only install with no
    warning. That is exactly how plotly 6 became reachable while the code
    still used the ``titlefont`` shorthand it removed.
    """
    spec = _pyproject_requirements()[package].specifier
    assert any(s.operator in ('<', '<=', '==', '~=') for s in spec), (
        f'{package} has no upper bound in pyproject.toml. It is on the '
        'list precisely because a major release broke this codebase '
        'before; removing the cap needs a deliberate decision, not a '
        'default.'
    )


def test_installed_versions_satisfy_pyproject():
    """What we run the suite against must be a legal install.

    Catches the reverse mistake: constraints tightened past the versions
    the environment actually has, so the declared install and the tested
    install are different things.
    """
    import importlib.metadata as md

    violations = []
    for name, req in _pyproject_requirements().items():
        try:
            installed = md.version(req.name)
        except md.PackageNotFoundError:
            continue  # optional extra, or not installed in this env
        if installed == '0.0.0':
            # Some conda-forge builds ship a placeholder dist-info version
            # while the module reports the real one -- pyinterp is 0.0.0
            # here but 2026.2.0 in `pyinterp.__version__`. Comparing the
            # placeholder would flag a package that is in fact current.
            continue
        if not req.specifier.contains(Version(installed), prereleases=True):
            violations.append(f'{name}: installed {installed} violates {req.specifier}')
    assert not violations, (
        'pyproject.toml excludes versions this environment is running:\n  '
        + '\n  '.join(violations)
    )
