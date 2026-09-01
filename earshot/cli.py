"""earshot -- is the dialogue in this actually audible, and can it be helped.

    earshot FILE_OR_URL                 measure it and say so in words
    earshot FILE_OR_URL --json          the same, as data
    earshot FILE_OR_URL --fix [OUT]     re-balance by level only, no separation
    earshot FILE_OR_URL --enhance [OUT] pull the voice out and set the ratio
    earshot FILE_OR_URL --voice-only    the separated voice, nothing under it

  --to N   with --enhance, how many LU above the background to put the voice
           (default 12; under +4 is where people reach for the subtitles)

`--fix` changes level over time and cannot change the ratio at any instant.
`--enhance` separates the voice from everything else and can, which is the
whole difference. It costs about four times realtime on a machine like this.

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
    target_lu = 12.0
    if "--to" in argv:
        i = argv.index("--to")
        argv.pop(i)
        target_lu = float(argv.pop(i))

    voice_only = "--voice-only" in argv
    if voice_only:
        argv.remove("--voice-only")

    out_path = None
    do_fix = do_enhance = False
    for flag in ("--fix", "--enhance"):
        if flag in argv:
            i = argv.index(flag)
            argv.pop(i)
            if i < len(argv) and not argv[i].startswith("-"):
                out_path = argv.pop(i)
            if flag == "--fix":
                do_fix = True
            else:
                do_enhance = True

    path = fetch(target) if target.startswith(("http://", "https://")) else Path(target)
    if not Path(path).exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    if do_enhance or voice_only:
        from . import separate
        before = measure.analyse(path)
        r = separate.enhance(path, out_path, target_lu=target_lu,
                             keep_background=not voice_only)
        after = measure.analyse(r.out_path, name=str(r.out_path))
        if as_json:
            print(json.dumps({"out": str(r.out_path),
                              "voice_only": str(r.stems.voice),
                              "background": str(r.stems.background),
                              "background_gain_db": r.background_gain_db,
                              "before": before.as_dict(),
                              "after": after.as_dict()}, indent=2, default=str))
        else:
            print(report.text(before))
            print("\n" + "-" * 60 + "\n")
            print(f"wrote {r.out_path}")
            print(f"  voice alone      {r.stems.voice}")
            print(f"  everything else  {r.stems.background}")
            print()
            print(f"  background moved {r.background_gain_db:+.1f} dB")
            print(f"  programme loudness {r.programme_lufs_before:.1f} -> "
                  f"{r.programme_lufs_after:.1f} LUFS")
            b = "--" if before.sbr_lu is None else f"{before.sbr_lu:+.1f}"
            a = "--" if after.sbr_lu is None else f"{after.sbr_lu:+.1f}"
            print(f"  measured ratio   {b} -> {a} LU")
        return 0

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
