#!/usr/bin/env python3.12
"""Cross-check earshot's BS.1770 against ffmpeg's ebur128 on signals chosen to
be able to DISAGREE.

Agreement on a steady sine proves almost nothing -- any wrong gate, any wrong
block length, any wrong overlap still gets a steady sine right. So the set here
is built around the things that separate implementations:

  quiet_then_loud   the relative gate has to discard the quiet half
  gapped            digital silence between bursts, which the absolute gate
                    must drop rather than average in
  wide_dynamics     speech-like bursts under a bed, the actual subject
  low_level         everything near the absolute gate, where it either fires
                    or does not

If this printed agreement on the sine alone I would have learned nothing, and
this project has a history of exactly that mistake.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earshot import decode                                   # noqa: E402
from earshot.loudness import BlockLoudness, gated_lufs        # noqa: E402

CASES = {
    # a plain 1 kHz sine, the easy case, kept only as a control
    "sine_1k": "sine=frequency=1000:duration=20,volume=-23dB",
    # loud for 10 s then 20 dB down for 10 s: the relative gate decides this
    "quiet_then_loud":
        "aevalsrc='0.5*sin(2*PI*440*t)*(1+9*lt(t\\,10))/10':d=20",
    # 1 s bursts with 1 s of true silence between them
    "gapped":
        "aevalsrc='0.4*sin(2*PI*300*t)*lt(mod(t\\,2)\\,1)':d=20",
    # a bed with periodic loud stabs, the shape this project is about
    "wide_dynamics":
        "aevalsrc='0.03*sin(2*PI*200*t)+0.6*sin(2*PI*900*t)*lt(mod(t\\,4)\\,0.3)':d=24",
    # everything close to the -70 LUFS absolute gate
    "low_level": "sine=frequency=1000:duration=20,volume=-65dB",
}


def ffmpeg_ebur128(path: Path) -> float:
    """ffmpeg's integrated loudness, with its floor decoded rather than believed.

    Measured 2026-09-01: ffmpeg prints `I: -70.0 LUFS` for a file of pure
    digital silence. So -70.0 is a sentinel meaning "nothing survived the
    absolute gate", not a measurement of anything. earshot returns -inf for the
    same state, which is the honest value; treating the two as disagreeing
    would have had me "fixing" a correct implementation to match a display
    convention.
    """
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(path),
         "-af", "ebur128=peak=none", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300)
    m = re.findall(r"I:\s*(-?\d+\.\d+)\s*LUFS", out.stderr)
    if not m:
        raise RuntimeError(f"no integrated loudness in ffmpeg output for {path.name}")
    v = float(m[-1])
    return float("-inf") if v <= -70.0 else v


def earshot_lufs(path: Path) -> float:
    media = decode.probe(path)
    acc = BlockLoudness(media.channels)
    for chunk in decode.stream(media):
        acc.feed(chunk)
    return gated_lufs(acc.blocks())


def analytic_check(tmp: Path) -> tuple[float, float]:
    """The one case with an answer that owes nothing to either implementation.

    BS.1770's -0.691 offset exists so that a 1 kHz tone reads its own level:
    the K-weighting has +0.691 dB of power gain at 1 kHz and the offset cancels
    it. So a stereo 1 kHz sine whose per-channel mean square is m must read
    exactly 10*log10(2m) LUFS, from arithmetic, with no meter involved.
    """
    import numpy as np
    wav = tmp / "analytic.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=1000:duration=5", "-ac", "2", "-ar", "48000",
         "-c:a", "pcm_f32le", str(wav)], check=True, timeout=120)
    samples = decode.read_all(decode.probe(wav))
    m = float((samples.astype(np.float64) ** 2).mean(axis=0).mean())
    return earshot_lufs(wav), 10.0 * float(np.log10(2 * m))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="earshot-xcheck-"))
    worst = 0.0
    rows = []

    mine, truth = analytic_check(tmp)
    rows.append(("1k sine vs arithmetic", mine, truth, abs(mine - truth)))
    worst = max(worst, abs(mine - truth))
    for name, src in CASES.items():
        wav = tmp / f"{name}.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y",
             "-f", "lavfi", "-i", f"{src}", "-ac", "2", "-ar", "48000",
             "-c:a", "pcm_f32le", str(wav)],
            check=True, timeout=300)
        mine, theirs = earshot_lufs(wav), ffmpeg_ebur128(wav)
        # Both below the absolute gate is agreement, not a nan.
        d = 0.0 if (mine == theirs == float("-inf")) else abs(mine - theirs)
        worst = max(worst, d)
        rows.append((name, mine, theirs, d))

    width = max(len(r[0]) for r in rows)
    print(f"{'case':<{width}}  {'earshot':>9}  {'ffmpeg':>9}  {'diff':>6}")
    for name, mine, theirs, d in rows:
        flag = "" if d <= 0.1 else "   <-- DISAGREE"
        print(f"{name:<{width}}  {mine:9.2f}  {theirs:9.2f}  {d:6.2f}{flag}")

    finite = [r[1] for r in rows if r[1] != float("-inf")]
    print(f"\nworst disagreement: {worst:.3f} LU")
    print(f"spread across the measurable cases: {max(finite) - min(finite):.1f} LU  "
          f"(if this were small the cases would not be discriminating), "
          f"plus one case below the absolute gate where both say so")
    return 0 if worst <= 0.1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
