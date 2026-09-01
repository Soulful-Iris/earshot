"""Making an existing mix easier to follow, and being straight about the limit.

## What this cannot do

It cannot turn the music down under the dialogue. The music and the voice are
one signal by the time anybody outside the mix suite hears them, and separating
them needs a source-separation model and a lot more machine than this. So the
speech-to-background ratio at any single instant is not something this touches,
and anything claiming otherwise from a stereo file is claiming a lot.

## What it can do, and it is most of the actual complaint

Two of the three numbers in a report are about level over TIME, not about the
instant, and both are fixable by a gain envelope:

  spread   the dialogue's own level wandering, so quiet lines vanish while the
           average looks fine. Lift the quiet passages.
  swing    the non-speech material towering over the dialogue, which is what
           makes people ride the volume all evening and then give up. Pull the
           loud parts down toward the voice.

That is night mode done with a speech detector instead of a broadband
compressor -- the difference being that this knows which parts are the talking,
so it can leave them alone and squash what is around them, rather than pumping
the voice every time a door slams.

The envelope moves on a 100 ms grid and is smoothed asymmetrically: quick to
duck a bang, slow to lift, because a fast lift is what audible pumping is.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import decode, vad
from .loudness import BlockLoudness, STEP_S, SR, block_lufs, mean_lufs
from .measure import SPEECH_BLOCK, Report, analyse

MAX_LIFT_DB = 9.0        # never invent more than this much level
MAX_DUCK_DB = 12.0       # nor take more away
HEADROOM_LU = 6.0        # how far above the dialogue non-speech may sit
ATTACK_S = 0.15          # ducking: fast enough to catch an impact
RELEASE_S = 0.8          # lifting: slow enough not to breathe


@dataclass
class FixResult:
    out_path: Path
    before: Report
    after: Report
    peak_before: float
    peak_after: float
    gain_applied_db: tuple[float, float]     # (min, max)


def _envelope(lufs: np.ndarray, speech: np.ndarray) -> np.ndarray:
    """Per-block gain in dB."""
    finite = np.isfinite(lufs)
    sp = speech & finite
    if not sp.any():
        return np.zeros(len(lufs))

    # Aim at the level the dialogue already spends most of its time at, so the
    # programme does not arrive louder or quieter than the listener chose.
    target = float(np.median(lufs[sp]))
    gain = np.zeros(len(lufs))

    gain[sp] = np.clip(target - lufs[sp], -MAX_DUCK_DB, MAX_LIFT_DB)

    other = finite & ~speech
    over = lufs[other] - (target + HEADROOM_LU)
    gain[other] = np.clip(-np.maximum(over, 0.0), -MAX_DUCK_DB, 0.0)
    return gain


def _smooth(gain: np.ndarray) -> np.ndarray:
    """Asymmetric one-pole: fast down, slow up.

    Done as an explicit loop because the coefficient depends on the direction of
    travel at every step, which is the whole point and is not a filter scipy has.
    """
    a_down = float(np.exp(-STEP_S / ATTACK_S))
    a_up = float(np.exp(-STEP_S / RELEASE_S))
    out = np.empty_like(gain)
    y = gain[0] if len(gain) else 0.0
    for i, g in enumerate(gain):
        a = a_down if g < y else a_up
        y = a * y + (1.0 - a) * g
        out[i] = y
    return out


def _blocks_and_speech(media: decode.Media) -> tuple[np.ndarray, np.ndarray]:
    windows = vad.speech_windows(media)
    acc = BlockLoudness(media.channels)
    for chunk in decode.stream(media):
        acc.feed(chunk)
    z = acc.blocks()
    frac = vad.speech_fraction_per_block(windows, len(z))
    return z, frac >= SPEECH_BLOCK


def rebalance(path: str | Path, out_path: str | Path | None = None) -> FixResult:
    src = Path(path)
    out = Path(out_path) if out_path else src.with_name(src.stem + ".earshot.wav")

    before = analyse(src)
    media = decode.probe(src)
    z, speech = _blocks_and_speech(media)
    gain_db = _smooth(_envelope(block_lufs(z), speech))

    # A gain per 100 ms block, stretched to a gain per sample. Linear
    # interpolation, so no step ever lands inside a word.
    block_t = np.arange(len(gain_db)) * STEP_S
    peak_before = 0.0
    peak_after = 0.0

    enc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR),
         "-ac", str(media.channels), "-i", "-", "-c:a", "pcm_s24le", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        pos = 0
        for chunk in decode.stream(media):
            n = chunk.shape[0]
            t = (pos + np.arange(n)) / SR
            g = np.interp(t, block_t, gain_db, left=gain_db[0] if len(gain_db) else 0.0,
                          right=gain_db[-1] if len(gain_db) else 0.0)
            scaled = chunk * (10.0 ** (g / 20.0))[:, None]
            peak_before = max(peak_before, float(np.abs(chunk).max(initial=0.0)))
            peak_after = max(peak_after, float(np.abs(scaled).max(initial=0.0)))
            # A lift that clips has made things worse, not better. Hard-limit
            # rather than let a boosted line arrive distorted.
            np.clip(scaled, -0.999, 0.999, out=scaled)
            enc.stdin.write(scaled.astype(np.float32).tobytes())
            pos += n
        enc.stdin.close()
        if enc.wait(timeout=600) != 0:
            raise decode.DecodeError(enc.stderr.read().decode("utf8", "replace")[:400])
    finally:
        if enc.poll() is None:
            enc.kill()
            enc.wait(timeout=10)

    return FixResult(out_path=out, before=before, after=analyse(out),
                     peak_before=peak_before, peak_after=peak_after,
                     gain_applied_db=(float(gain_db.min()), float(gain_db.max()))
                     if len(gain_db) else (0.0, 0.0))


def summary(r: FixResult) -> str:
    def show(name: str, a, b, better: str) -> str:
        if a is None or b is None:
            return f"  {name:<10} {'--':>8} -> {'--':>8}"
        return (f"  {name:<10} {a:8.1f} -> {b:8.1f}   "
                f"{'better' if better == 'down' and b < a or better == 'up' and b > a else 'no better'}")

    out = [f"wrote {r.out_path}", ""]
    out.append(show("swing", r.before.swing_lu, r.after.swing_lu, "down"))
    out.append(show("spread", r.before.spread_lu, r.after.spread_lu, "down"))
    out.append(show("ratio", r.before.sbr_lu, r.after.sbr_lu, "up"))
    out.append("")
    out.append(f"  gain applied: {r.gain_applied_db[0]:+.1f} to "
               f"{r.gain_applied_db[1]:+.1f} dB")
    out.append(f"  peak {r.peak_before:.3f} -> {r.peak_after:.3f}")
    out.append("")
    out.append("  The ratio is NOT expected to move much: nothing here separates")
    out.append("  the voice from what is under it. Swing and spread are the two")
    out.append("  this can actually change, and they are most of the complaint.")
    return "\n".join(out)
