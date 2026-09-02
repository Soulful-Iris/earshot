"""The static SOURCES table must not drift from the real models.

`available()` used to answer by loading the network, which cost the web server
~600 MB and left it at 96% of its 700 MiB cap for ever. It answers from a table
now. A table is only safe if something notices when it stops being true, so this
loads both models for real and compares.

Slow and memory-hungry on purpose. It is the check that is allowed to be
expensive, because it runs here rather than on every upload.

Run: python3.12 tests/test_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earshot import separate                                       # noqa: E402


def test_sources_table_matches_the_real_models():
    for name, listed in separate.SOURCES.items():
        real = list(separate.model(name).sources)
        assert real == listed, (
            f"{name}: SOURCES says {listed}, the model says {real}. "
            f"The table in separate.py is stale — fix the table, not this test.")


def test_available_agrees_with_and_without_the_table():
    """The cheap path and the expensive path must give the same answer.

    This is the case that would fail if the table were right but `available()`
    filtered it differently from how it filtered the live list — a bug the test
    above cannot see, because it only compares the raw sources.
    """
    for name in separate.SOURCES:
        cheap = separate.available(name)
        saved, separate.SOURCES = separate.SOURCES, {}
        try:
            expensive = separate.available(name)
        finally:
            separate.SOURCES = saved
        assert cheap == expensive, f"{name}: table gives {cheap}, model gives {expensive}"


def test_every_named_part_has_a_description():
    """A part offered on the page with no sentence under it is a mystery box."""
    for name in separate.SOURCES:
        for part in separate.available(name):
            assert part in separate.PARTS, f"{part} is offered but not described"


if __name__ == "__main__":
    import traceback
    fails = 0
    for n, fn in sorted(globals().items()):
        if not n.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {n}")
        except Exception:
            fails += 1
            print(f"  FAIL {n}")
            traceback.print_exc()
    print("all green" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
