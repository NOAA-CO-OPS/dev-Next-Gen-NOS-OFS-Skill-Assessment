"""
Tests for the retirement of the netcdf_dir config setting (issue #213).

Covers local_model_dir (canonical {ofs}/netcdf/ layout with a fallback
to the legacy no-subdirectory layout left behind by the retired blank
netcdf_dir workaround) and the download-failure log classification that
replaced the misleading "NODD S3 is not responding!" catch-all.
"""

from urllib.error import ContentTooShortError, HTTPError, URLError

from bin.utils import get_model_data
from ofs_skill.model_processing.list_of_files import (
    NETCDF_SUBDIR,
    local_model_dir,
)


class MockLogger:  # pylint: disable=too-few-public-methods
    """Capture log calls by level."""

    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **_kwargs):
        """Record an info message."""
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args, **_kwargs):
        """Record a warning message."""
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **_kwargs):
        """Record an error message."""
        self.errors.append(msg % args if args else msg)


# ---------------------------------------------------------------------------
# local_model_dir: canonical layout with legacy fallback
# ---------------------------------------------------------------------------

def test_local_model_dir_canonical(tmp_path):
    """The canonical {ofs}/netcdf/ directory wins when it exists."""
    canonical = tmp_path / 'cbofs' / NETCDF_SUBDIR
    canonical.mkdir(parents=True)
    logger = MockLogger()
    result = local_model_dir(str(tmp_path), 'cbofs', logger)
    assert result == str(canonical)
    assert not logger.infos


def test_local_model_dir_legacy_fallback(tmp_path):
    """A populated legacy tree (no netcdf/ level) is used with a notice."""
    legacy = tmp_path / 'cbofs'
    (legacy / '2025' / '07' / '01').mkdir(parents=True)
    logger = MockLogger()
    result = local_model_dir(str(tmp_path), 'cbofs', logger)
    assert result == str(legacy)
    assert any('legacy layout' in msg for msg in logger.infos)


def test_local_model_dir_fresh_tree(tmp_path):
    """With nothing on disk, the canonical path is returned for creation."""
    logger = MockLogger()
    result = local_model_dir(str(tmp_path), 'cbofs', logger)
    assert result == str(tmp_path / 'cbofs' / NETCDF_SUBDIR)
    assert not logger.infos


def test_local_model_dir_empty_legacy_dir(tmp_path):
    """An empty {ofs}/ directory does not trigger the legacy fallback."""
    (tmp_path / 'cbofs').mkdir()
    logger = MockLogger()
    result = local_model_dir(str(tmp_path), 'cbofs', logger)
    assert result == str(tmp_path / 'cbofs' / NETCDF_SUBDIR)
    assert not logger.infos


# ---------------------------------------------------------------------------
# _log_download_failure: honest cause classification (issue #213)
# ---------------------------------------------------------------------------

def _classify(exc):
    """Run the classifier and return the logged error messages."""
    logger = MockLogger()
    # pylint: disable-next=protected-access
    get_model_data._log_download_failure(
        'https://noaa-nos-ofs-pds.s3.amazonaws.com/cbofs/x.nc', exc, logger)
    return logger.errors


def test_classify_http_404_is_not_an_outage():
    """A 404 names the URL and says the NODD is reachable."""
    errors = _classify(
        HTTPError('http://x', 404, 'Not Found', None, None))
    assert any('404' in msg and 'not a NODD outage' in msg for msg in errors)
    assert not any('not responding' in msg for msg in errors)


def test_classify_http_other_status():
    """Non-404 HTTP errors report the actual status code."""
    errors = _classify(
        HTTPError('http://x', 403, 'Forbidden', None, None))
    assert any('403' in msg and 'responded' in msg for msg in errors)


def test_classify_unreachable():
    """Connection-level failures keep the outage wording."""
    errors = _classify(URLError('connection refused'))
    assert any('Could not reach' in msg for msg in errors)


def test_classify_interrupted_transfer():
    """A transfer that dies mid-download is reported as interrupted."""
    errors = _classify(ContentTooShortError('connection dropped', b''))
    assert any('interrupted' in msg for msg in errors)


def test_classify_generic_exception():
    """Anything unrecognized still logs the URL and the exception."""
    errors = _classify(RuntimeError('boom'))
    assert any('boom' in msg for msg in errors)
