"""
Shared Canadian Hydrographic Service (CHS) HTTP utilities.

Enforces rolling rate limits across all CHS API interactions, matching the
limits CHS publishes at
https://tides.gc.ca/en/web-services-offered-canadian-hydrographic-service:
    - Max 3 requests per second
    - Max 30 requests per minute

Exceeding either returns HTTP 429.
"""

import re
import threading
import time
from collections import deque
from typing import Any

import requests

# Authoritative CHS API Base URLs
CHS_IWLS_BASE_URL = 'https://api-iwls.dfo-mpo.gc.ca/api/v1'
CHS_SINE_BASE_URL = 'https://api-sine.dfo-mpo.gc.ca/api/v1'

# Published CHS limits (see module docstring).
_MAX_PER_SECOND = 3
_MAX_PER_MINUTE = 30

# time.sleep can wake fractionally early, which would otherwise leave the
# window still full and force an extra loop iteration; nudge past the edge.
_SLEEP_EPSILON = 0.005

# CHS backend IDs are 24-character MongoDB ObjectIds (hexadecimal)
_CHS_OBJECT_ID_RE = re.compile(r'^[0-9a-fA-F]{24}$')

def is_chs_uuid(val: Any) -> bool:
    """Return True if val matches a 24-character CHS backend ObjectId."""
    return bool(isinstance(val, str) and _CHS_OBJECT_ID_RE.match(val))

class RateLimiter:
    """Enforces dual rolling rate limits: 3/sec and 30/min.

    Admission is serialized by a lock. Deciding whether to wait and then
    recording the request is a check-then-act sequence, so without the lock
    concurrent callers all observe the same under-cap history, all decline
    to wait, and all fire together. Measured with four threads against the
    unlocked version: 11 requests in a one-second window (cap 3) and 39 in
    a minute (cap 30), which is what CHS answers with HTTP 429.

    The lock is held across the sleep, so each caller reserves its slot in
    turn. It is *not* held during the request itself -- ``chs_get`` returns
    from ``wait`` before issuing the GET -- so transfers still overlap
    across worker threads. Only admission is serialized.
    """

    def __init__(self):
        self.second_history = deque()
        self.minute_history = deque()
        self._lock = threading.Lock()

    def _purge(self, now):
        """Drop timestamps that have aged out of both rolling windows."""
        while self.second_history and now - self.second_history[0] >= 1.0:
            self.second_history.popleft()
        while self.minute_history and now - self.minute_history[0] >= 60.0:
            self.minute_history.popleft()

    def wait(self):
        with self._lock:
            while True:
                now = time.time()
                self._purge(now)

                wait_time = 0.0
                if len(self.second_history) >= _MAX_PER_SECOND:
                    wait_time = max(
                        wait_time, 1.0 - (now - self.second_history[0]))
                if len(self.minute_history) >= _MAX_PER_MINUTE:
                    wait_time = max(
                        wait_time, 60.0 - (now - self.minute_history[0]))

                # Re-check after sleeping rather than admitting straight
                # away: time.sleep can return fractionally early, leaving
                # the window still full, and the previous code appended
                # regardless. That admitted a 4th request inside a
                # one-second window whose cap is 3.
                if wait_time <= 0:
                    break
                time.sleep(wait_time + _SLEEP_EPSILON)

            self.second_history.append(now)
            self.minute_history.append(now)


# Global module singleton
chs_rate_limiter = RateLimiter()


def chs_get(url: str, **kwargs: Any) -> requests.Response:
    """Issue a strictly rate-limited GET request to the CHS API."""
    chs_rate_limiter.wait()
    return requests.get(url, **kwargs)
