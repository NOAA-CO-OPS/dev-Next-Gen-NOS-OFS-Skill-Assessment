"""
Regression: a failed first NODD download must be described by its cause.

Issue #213: a blank ``netcdf_dir`` dropped a segment from every non-STOFS
URL, so every request 404'd. The log said "NODD S3 is not responding! I'm
out.", which reads as a NOAA outage -- the reporter went looking at the
service before finding the config typo. A 404 means the bucket answered
and the object is not there: the service is up and the path is wrong.
"""
from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest
from bin.utils import get_model_data

_URL = 'https://noaa-nos-ofs-pds.s3.amazonaws.com/cbofs/2025/06/30/x.nc'


def _http_error(code):
    return HTTPError(_URL, code, 'msg', {}, None)


def test_404_blames_the_path_not_the_service():
    message = get_model_data._describe_nodd_failure(_http_error(404), _URL)

    assert 'netcdf_dir' in message
    assert '404' in message
    # Must not read as an outage.
    assert 'not responding' not in message.lower()


def test_404_names_the_url_that_was_tried():
    message = get_model_data._describe_nodd_failure(_http_error(404), _URL)

    assert _URL in message


def test_403_is_reported_as_access_not_outage():
    message = get_model_data._describe_nodd_failure(_http_error(403), _URL)

    assert '403' in message
    assert 'not responding' not in message.lower()


@pytest.mark.parametrize('code', [500, 502, 503])
def test_server_errors_are_attributed_to_the_service(code):
    message = get_model_data._describe_nodd_failure(_http_error(code), _URL)

    assert str(code) in message
    assert 'on the service' in message


def test_connectivity_failure_is_still_reported_as_unreachable():
    """A genuine outage must not be mislabeled as a bad path."""
    message = get_model_data._describe_nodd_failure(
        URLError('Name or service not known'), _URL)

    assert 'Could not reach NODD S3' in message
    assert 'netcdf_dir' not in message


def test_non_http_exception_does_not_claim_a_status():
    message = get_model_data._describe_nodd_failure(ValueError('boom'), _URL)

    assert 'Could not reach NODD S3' in message
