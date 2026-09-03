"""Concurrency tests for the CHS rate limiter.

CHS publishes 3 requests/second and 30 requests/minute and answers HTTP 429
past either. Deciding whether to wait and then recording the request is a
check-then-act sequence, so the limiter must serialize admission: without
it, concurrent callers all observe the same under-cap history and fire
together. Measured with four threads against an unlocked implementation:
11 admissions in one second and 39 in one minute.

These run against the rolling one-second window only, so they stay fast.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ofs_skill.obs_retrieval.chs_utils import (
    _MAX_PER_SECOND,
    RateLimiter,
)


def _admit(limiter, count, workers):
    """Run ``count`` admissions across ``workers`` threads; return stamps."""
    stamps: list[float] = []
    lock = threading.Lock()

    def one(_):
        limiter.wait()
        with lock:
            stamps.append(time.time())

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(one, range(count)))

    return sorted(stamps)


def _worst_window(stamps, span):
    """Largest number of stamps falling in any half-open window of `span`."""
    return max(
        sum(1 for s in stamps if start <= s < start + span)
        for start in stamps
    )


@pytest.mark.parametrize('workers', [1, 2, 4, 8])
def test_per_second_cap_holds_under_concurrency(workers):
    """No one-second window may contain more than the published burst."""
    stamps = _admit(RateLimiter(), 12, workers)

    assert _worst_window(stamps, 1.0) <= _MAX_PER_SECOND


def test_concurrent_callers_are_not_admitted_together():
    """The unlocked version released whole batches early.

    9 admissions capped at 3/sec need two full second-boundaries, so the
    span must exceed 2s. The unlocked implementation completed the same 9
    inside 1.0s, because after sleeping it appended without re-checking.
    """
    limiter = RateLimiter()
    stamps = _admit(limiter, 9, 9)

    assert stamps[-1] - stamps[0] >= 1.9


def test_admission_is_recorded_once_per_call():
    """Each wait() records exactly one request in both rolling windows."""
    limiter = RateLimiter()
    _admit(limiter, 3, 1)

    assert len(limiter.second_history) == 3
    assert len(limiter.minute_history) == 3
