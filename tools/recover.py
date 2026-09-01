#!/usr/bin/env python3.12
"""Does the measurement recover the ratio that is actually there?

This is the only thing in the project that can say "you made it worse". The
fixtures are speech mixed under a bed, and both stems are kept, so the answer
exists before anything measures it.

## Two truths, because they turned out to be different numbers

    asked-for   the gap I built the file with: the loudness of the whole voice
                stem minus the loudness of the whole bed stem.

    at-speech   the loudness of the VOICE STEM during the blocks the analyser
                calls speech, minus the loudness of the BED STEM over those same
                blocks.

They agree to within about half a LU on every fixture here. That is worth
saying plainly because I predicted otherwise: when the human-voice cases first
failed by 3 LU I wrote down that `at-speech` would explain it -- that a
continuous reader's fully-voiced blocks are the loud middles of phrases and sit
above that reader's own average. The stems said no. `at-speech` came back at
+14.82 against an asked-for +15, and the 3 LU was a real bias in my estimator,
which turned out to be two: a bed that was quieter in the gaps than under the
speech, and BS.1770's relative gate being applied inside local windows where it
does not belong.

Both columns stay on the page anyway. `at-speech` is the target because it is
the ratio at the moments a listener is trying to follow the words, and
`asked-for` is the one that owes nothing at all to the analyser's opinions.

Both are printed. If only the flattering one were printed this file would be
the sort of instrument the rest of this repo is a monument to.

Note what `at-speech` does and does not test. It takes the analyser's block
CLASSIFICATION as given and tests the ESTIMATOR -- the claim that you can
recover the speech-to-bed ratio from a finished mix by subtracting the power
measured in the gaps. A mask that was wrong in the same way on both stems would
survive this, which is why `asked-for` stays on the page beside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                        # noqa: E402

from earshot import decode, measure                       # noqa: E402
from earshot.loudness import BlockLoudness, gated_lufs    # noqa: E402

# Measured, not chosen: the worst error against the at-speech truth across the
# tuned corpora is ~2.1 LU and the mean is under 1. The held-out cases are
# scored separately below and are the number to believe, because three
# constants in measure.py were swept against everything else here.
TOLERANCE = 2.5   # LU


def blocks_of(path: Path) -> np.ndarray:
    media = decode.probe(path)
    acc = BlockLoudness(media.channels)
    for chunk in decode.stream(media):
        acc.feed(chunk)
    return acc.blocks()


def truth_at_speech(fx: Path, case: dict, mask: np.ndarray) -> float | None:
    """The ratio actually present in the blocks the analyser judged."""
    if "voice_stem" not in case:
        return None
    zv = blocks_of(fx / case["voice_stem"])
    zb = blocks_of(fx / case["bed_stem"])
    n = min(len(zv), len(zb), len(mask))
    m = mask[:n]
    if not m.any():
        return None
    v = gated_lufs(zv[:n], m)
    b = gated_lufs(zb[:n], m)
    if not (np.isfinite(v) and np.isfinite(b)):
        return None
    return float(v - b)


def main() -> int:
    fx = Path(sys.argv[1] if len(sys.argv) > 1 else
              Path(__file__).resolve().parents[1] / "fixtures")
    manifest = json.loads((fx / "manifest.json").read_text())

    print(f"{'fixture':<16}{'asked':>7}{'at-speech':>11}{'raw S-B':>9}"
          f"{'recovered':>11}{'err':>7}  speech%")
    worst = 0.0
    refused: list[tuple] = []
    failures: list[tuple] = []
    held: list[tuple] = []

    for case in manifest["cases"]:
        r = measure.analyse(fx / case["file"], keep_blocks=True)
        asked = case["speech_minus_bed_lu"]
        raw = r.speech_lufs - r.background_lufs
        got = r.sbr_lu
        truth = truth_at_speech(fx, case, r.speech_mask)

        if asked is None:
            print(f"{case['file']:<16}{'--':>7}{'--':>11}{raw:9.2f}"
                  f"{'--' if got is None else f'{got:.2f}':>11}{'':>7}"
                  f"  {r.speech_fraction*100:.0f}%"
                  f"   <- speech only; this is the voice above its own noise floor")
            continue

        target = truth if truth is not None else asked
        a = f"{asked:+.0f}"
        t = "--" if truth is None else f"{truth:+.2f}"

        if got is None:
            refused.append((case["file"], target, raw))
            print(f"{case['file']:<16}{a:>7}{t:>11}{raw:9.2f}{'refused':>11}"
                  f"{'':>7}  {r.speech_fraction*100:.0f}%"
                  f"   <- {raw:.2f} dB apart, too close to invert")
            continue

        err = abs(got - target)
        if case.get("held_out"):
            held.append((case["file"], err))
        else:
            worst = max(worst, err)
        flag = ("  (held out)" if case.get("held_out") else "")
        flag += "" if err <= TOLERANCE else "   <-- OUT"
        if err > TOLERANCE:
            failures.append((case["file"], round(target, 2), round(got, 2)))
        print(f"{case['file']:<16}{a:>7}{t:>11}{raw:9.2f}{got:+11.2f}{err:7.2f}"
              f"  {r.speech_fraction*100:.0f}%{flag}")

    print(f"\ntuned corpora, worst error against at-speech truth: {worst:.2f} LU "
          f"(tolerance {TOLERANCE} LU)")
    if held:
        print(f"HELD OUT, never swept against: worst "
              f"{max(e for _, e in held):.2f} LU over {len(held)} case(s) -- "
              + ", ".join(f"{f} {e:.2f}" for f, e in held))
    else:
        print("HELD OUT: none present. Every number above is a best case.")
    if refused:
        print(f"refused {len(refused)}: " +
              ", ".join(f"{f} (truth {t:+.1f} LU)" for f, t, _ in refused))
        print("  -> a mix whose voice sits at or under its bed cannot be "
              "measured this way.\n     Real coverage limit, stated in the "
              "report rather than hidden.")
    if failures:
        print("FAILED:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
