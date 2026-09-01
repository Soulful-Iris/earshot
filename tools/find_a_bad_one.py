#!/usr/bin/env python3.12
"""Go and find real content whose dialogue is genuinely buried.

Bruno, on the first before/after pair I sent him: *"They sound identical. Find
an audio that the change would be noticeable to anyone."*

He was right, and the cause was upstream of the demo. Measured: 92.4% of the
energy in my synthetic "music bed" sat below 300 Hz and 1.6% sat in the 300 Hz
to 1 kHz band where speech lives. It was a rumble. It never masked the voice in
the range that matters, a phone speaker barely reproduces it, and the +12 dB
separation figure it produced was for an easier problem than the real one.

So stop inventing backgrounds. This walks real public-domain films, measures
them with the tool, and reports the ones where the dialogue is actually hard to
hear. The measurement half of the project picks the material for the separation
half, which is the first time these two things have been pointed at each other.

Prelinger films are dedicated to the public domain by the archive.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earshot import decode, measure                      # noqa: E402

WORK = Path.home() / "ventures" / "earshot" / "work" / "hunt"
SAMPLE_S = 60


def files_for(identifier: str) -> list[dict]:
    url = f"https://archive.org/metadata/{identifier}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r).get("files", [])


def best_source(identifier: str) -> str | None:
    """The smallest thing with audio in it. Video is only a carrier here.

    Most archive.org film items carry an mp3 derivative, which is ten megabytes
    instead of five hundred for the same soundtrack.
    """
    playable = []
    for f in files_for(identifier):
        name = f.get("name", "")
        if not name.lower().endswith((".mp3", ".ogg", ".mp4", ".ogv", ".mpeg", ".mpg")):
            continue
        try:
            playable.append((int(f.get("size", 0)), name))
        except (TypeError, ValueError):
            continue
    if not playable:
        return None
    playable.sort()
    return f"https://archive.org/download/{identifier}/{playable[0][1]}"


def fetch(url: str, dest: Path, max_mb: int = 120) -> Path | None:
    """Download with urllib, not by handing the URL to ffmpeg.

    ffmpeg on this box cannot resolve hostnames -- `Failed to resolve hostname
    archive.org: System error` -- while curl and urllib can. Worth knowing
    before spending an hour on a network problem that is only in one program.
    """
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "earshot/0.1"})
        with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as f:
            got = 0
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                got += len(buf)
                if got > max_mb << 20:
                    break
                f.write(buf)
    except Exception as e:                                       # noqa: BLE001
        print(f"    download failed: {e}")
        return None
    return dest if dest.stat().st_size > 100_000 else None


def grab(src: Path, start: float, out: Path) -> Path | None:
    """Cut one minute of audio out of a local file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start),
         "-i", str(src), "-t", str(SAMPLE_S), "-vn", "-ac", "2", "-ar", "48000",
         "-c:a", "pcm_f32le", str(out)],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        # A silent skip here is how the first run of this script printed nothing
        # at all and looked like it had found no bad films.
        print(f"    cut at {start}s failed: {r.stderr.strip()[:120]}")
        return None
    if not out.exists() or out.stat().st_size <= 100_000:
        print(f"    cut at {start}s produced nothing usable (past the end?)")
        return None
    return out


def midrange_share(path: Path) -> float:
    """Fraction of energy between 300 Hz and 4 kHz.

    The number my own fixtures failed. A background that is not in this band
    cannot mask speech however loud it measures, so a candidate with a low
    share here will produce another pair of clips that sound identical.
    """
    import numpy as np
    media = decode.probe(path)
    x = decode.read_all(media, channels=1)[:, 0].astype(np.float64)
    n = 1 << 15
    if len(x) < n:
        return 0.0
    x = x[: len(x) // n * n].reshape(-1, n)
    P = (np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / 48000)
    return float(P[(f >= 300) & (f < 4000)].sum() / max(P.sum(), 1e-30))


def main() -> int:
    ids = sys.argv[1:] or [
        "Practici1953", "QuietRev1956", "PlacetoL1948", "RadioatW1944",
        "0725_Reasons_Why_The_09_01_01_00", "0819_Oklahoma_Heartland_USA_02_00_56_00",
        "getting_acquainted_with_engineering_2", "201376_America_in_Turmoil",
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for ident in ids:
        try:
            url = best_source(ident)
        except Exception as e:                                   # noqa: BLE001
            print(f"{ident:<42} metadata failed: {e}")
            continue
        if not url:
            print(f"{ident:<42} nothing playable")
            continue
        local = fetch(url, WORK / Path(url).name)
        if not local:
            print(f"{ident:<42} could not be fetched")
            continue
        for start in (120, 300):
            clip = WORK / f"{ident}_{start}.wav"
            got = clip if clip.exists() else grab(local, start, clip)
            if not got:
                continue
            try:
                r = measure.analyse(got)
            except Exception as e:                               # noqa: BLE001
                print(f"{ident:<42} {start:>4}s  measure failed: {e}")
                continue
            mid = midrange_share(got)
            sbr = "buried" if r.sbr_lu is None else f"{r.sbr_lu:+.1f}"
            rows.append((r.sbr_lu if r.sbr_lu is not None else -99, mid, ident, start,
                         r.speech_fraction))
            print(f"{ident:<42} {start:>4}s  ratio {sbr:>7} LU   "
                  f"speech {r.speech_fraction*100:3.0f}%   "
                  f"midrange {mid*100:4.1f}%")

    rows.sort()
    print("\nworst first, and midrange share is the tiebreak -- a background")
    print("outside 300 Hz to 4 kHz cannot mask a voice however loud it measures:")
    for sbr, mid, ident, start, sf in rows[:6]:
        print(f"  {ident} @{start}s   ratio {sbr:+.1f}   midrange {mid*100:.1f}%   "
              f"speech {sf*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
