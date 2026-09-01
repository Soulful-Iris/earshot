#!/usr/bin/env python3.12
"""Does the re-balance make the words easier to GET, not just the numbers nicer?

`tools/recover.py` proves the meter reads the right number. It says nothing
about whether the fix helps, and "swing went down 4 LU" is exactly the kind of
receipt this repo has a long history of mistaking for an answer. Every check I
can write from inside is a check on levels; none of them has a listener in it.

The closest available stand-in for a listener is a speech recogniser, and it is
a fair one because it is not mine, it was never tuned on any of this, and it
fails in roughly the direction people do: when the voice is buried it starts
dropping and inventing words.

    reference    the recogniser's transcript of the CLEAN VOICE STEM
    before       the same recogniser on the mix
    after        the same recogniser on the re-balanced mix

and the score is word error rate against the reference. Using the recogniser's
own reading of the clean stem rather than the script means the number measures
what the MIX did to the words, not the recogniser's opinion of the voice.

If WER does not improve, the fix does not help, whatever the LU columns say.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earshot import fix as fixmod                        # noqa: E402
from make_fixtures import deepgram_key                   # noqa: E402

# Nova-3 understands speech a person would struggle with, so most fixtures sit
# at a WER of zero before anything is done to them and can only stay there. The
# cases worth running are the ones with headroom: a mix where the recogniser is
# already making mistakes.
HARD_CASES = ["gap-05.wav", "gap+00.wav", "real_gap+04.wav", "real_gap+08.wav"]


def transcribe(path: Path, key: str) -> str:
    wav = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ar", "16000", "-ac", "1", "-f", "wav", "-"],
        capture_output=True, timeout=600, check=True).stdout
    req = urllib.request.Request(
        "https://api.deepgram.com/v1/listen?model=nova-3&punctuate=false&smart_format=false",
        data=wav,
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["results"]["channels"][0]["alternatives"][0]["transcript"].lower()


def wer(ref: str, hyp: str) -> float:
    """Levenshtein over words, normalised by reference length."""
    r, h = ref.split(), hyp.split()
    if not r:
        return float("nan")
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def main() -> int:
    fx = Path(__file__).resolve().parents[1] / "fixtures"
    manifest = json.loads((fx / "manifest.json").read_text())
    by_file = {c["file"]: c for c in manifest["cases"]}
    key = deepgram_key()
    out = Path("/tmp/earshot-intel")
    out.mkdir(exist_ok=True)

    print(f"{'fixture':<18}{'WER before':>11}{'WER after':>10}{'change':>9}")
    improved = worse = 0
    for name in HARD_CASES:
        case = by_file.get(name)
        if not case or "voice_stem" not in case:
            continue
        ref = transcribe(fx / case["voice_stem"], key)
        if not ref.strip():
            print(f"{name:<18}  reference transcript empty; skipping")
            continue
        res = fixmod.rebalance(fx / name, out / name)
        b = wer(ref, transcribe(fx / name, key))
        a = wer(ref, transcribe(res.out_path, key))
        d = a - b
        mark = "  better" if d < -0.005 else ("  WORSE" if d > 0.005 else "  same")
        improved += d < -0.005
        worse += d > 0.005
        print(f"{name:<18}{b:11.3f}{a:10.3f}{d:+9.3f}{mark}")

    print(f"\nbetter on {improved}, worse on {worse}")
    print("A recogniser is not an ear. It is the only instrument here that was")
    print("not built by me and does not measure level, which is why it is the")
    print("one that can say the fix did nothing.")
    return 1 if worse > improved else 0


if __name__ == "__main__":
    raise SystemExit(main())
