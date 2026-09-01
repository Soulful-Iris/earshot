"""earshot -- is the dialogue in this actually audible, and can it be helped.

    earshot FILE_OR_URL              measure it and say so in words
    earshot FILE_OR_URL --json       the same, as data
    earshot FILE_OR_URL --fix [OUT]  write a re-balanced audio file beside it

A URL is fetched to a temporary file first; nothing here streams from the net
while measuring, because a stall halfway through a decode is indistinguishable
from a quiet passage.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from . import fix as fixmod
from . import measure, report

MAX_FETCH_MB = 300


def fetch(url: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="earshot-")) / Path(url.split("?")[0]).name
    if not tmp.suffix:
        tmp = tmp.with_suffix(".bin")
    req = urllib.request.Request(url, headers={"User-Agent": "earshot/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        got = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            got += len(buf)
            if got > MAX_FETCH_MB << 20:
                raise SystemExit(f"refusing to download more than {MAX_FETCH_MB} MB")
            f.write(buf)
    return tmp


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0

    target = argv.pop(0)
    as_json = "--json" in argv
    if as_json:
        argv.remove("--json")
    do_fix = "--fix" in argv
    out_path = None
    if do_fix:
        i = argv.index("--fix")
        argv.pop(i)
        if i < len(argv) and not argv[i].startswith("-"):
            out_path = argv.pop(i)

    path = fetch(target) if target.startswith(("http://", "https://")) else Path(target)
    if not Path(path).exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    if do_fix:
        result = fixmod.rebalance(path, out_path)
        if as_json:
            print(json.dumps({"out": str(result.out_path),
                              "before": result.before.as_dict(),
                              "after": result.after.as_dict()},
                             indent=2, default=str))
        else:
            print(report.text(result.before))
            print("\n" + "-" * 60 + "\n")
            print(fixmod.summary(result))
        return 0

    r = measure.analyse(path)
    print(json.dumps(r.as_dict(), indent=2, default=str) if as_json else report.text(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
