"""Shared fixtures.

`tmp` exists because `tests/test_upload.py` asks for it and nothing ever
provided it. Those ten tests have errored at setup since the commit that
introduced them, and pytest reports a missing fixture as an ERROR rather than a
FAILURE -- so the summary line read `21 passed, 10 errors`, which scans as a
green run with some noise after it.

They are not incidental tests. They are the ones guarding the upload path's
memory behaviour, including `test_memory_is_bounded_not_proportional`, which
encodes the measurement that pinned MAX_MB at 25: a 24 MB upload taking the
server from 76 MB RSS to 159 MB. That guard has never once run.

Found 2026-09-03 while adding the video tests, by reading the summary line
instead of the exit status.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp(tmp_path):
    """A temporary directory, under the name the upload tests ask for."""
    return tmp_path
