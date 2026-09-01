#!/usr/bin/env python3.12
"""Score anything that claims to pull the voice out of a mix.

The reason this can exist at all is that the fixtures were built by mixing a
voice stem and a bed stem, and both were kept. So for every candidate separator
there is a true answer sitting on disk, and "it sounds better" never has to be
anybody's opinion.

Four numbers, and they disagree with each other often enough to be worth having:

  SI-SDR      scale-invariant signal-to-distortion against the true voice stem,
              in dB. The standard separation metric. Higher is better; the
              baseline is the mix itself, so what matters is the IMPROVEMENT
              over doing nothing.
  bg drop     how much quieter the background is in the estimate than in the
              mix. A separator can score well here by deleting everything,
              which is why it is never read on its own.
  voice keep  how much of the true voice survives. This is the one that catches
              deleting everything.
  WER         a speech recogniser on the estimate, against its own transcript
              of the clean stem. The only score here with anything
              listener-shaped in it.

A separator that improves SI-SDR while WER gets worse has made the numbers nicer
and the words harder, which is the entire failure mode this repo is built
against.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earshot.loudness import SR                            # noqa: E402

Separator = Callable[[Path, Path], Path]      # (mix, out_dir) -> estimated voice


def read_mono(path: Path) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"],
        capture_output=True, timeout=900, check=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)


def align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def best_lag(estimate: np.ndarray, target: np.ndarray, max_lag: int = 48000) -> int:
    """Samples the estimate lags the target by, found by cross-correlation.

    Not optional, and it cost an hour of being wrong to learn. Every FFT-based
    filter here (dialoguenhance, afftdn, anlmdn) delays its output by a window,
    and SI-SDR is destroyed by a handful of samples of misalignment: the first
    run of this bench scored ffmpeg's own dialogue enhancer at -55 dB and I was
    one step from reporting that it does not work. It was 1024 samples late.

    Correlation is done on a middle slice so a long file does not cost a
    full-length FFT, and the search is capped at a second either way.
    """
    e, t = align(estimate, target)
    n = len(e)
    if n < 4096:
        return 0
    # a slice with something in it, not the head, which is often silence
    lo = n // 4
    hi = min(n, lo + 20 * 48000)
    e, t = e[lo:hi], t[lo:hi]
    size = 1 << int(np.ceil(np.log2(len(e) + len(t))))
    corr = np.fft.irfft(np.fft.rfft(e, size) * np.conj(np.fft.rfft(t, size)), size)
    corr = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    return int(np.argmax(np.abs(corr)) - max_lag)


def undelay(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    lag = best_lag(estimate, target)
    if lag > 0:
        return estimate[lag:]
    if lag < 0:
        return np.concatenate([np.zeros(-lag), estimate])
    return estimate


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Invariant to overall gain, which matters:
    a separator that gets the voice right but 6 dB quiet is not wrong."""
    e, t = align(estimate, target)
    t = t - t.mean()
    e = e - e.mean()
    denom = float(t @ t)
    if denom <= 0:
        return float("nan")
    alpha = float(e @ t) / denom
    proj = alpha * t
    noise = e - proj
    p, n = float(proj @ proj), float(noise @ noise)
    if n <= 0:
        return float("inf")
    return 10.0 * np.log10(p / n) if p > 0 else float("-inf")


def rms_db(x: np.ndarray) -> float:
    return 10.0 * np.log10(float(x @ x) / max(len(x), 1) + 1e-20)


def score(mix: Path, voice_stem: Path, bed_stem: Path, estimate: Path) -> dict:
    voice_full = read_mono(voice_stem)
    est_full = undelay(read_mono(estimate), voice_full)
    mixa, est = align(read_mono(mix), est_full)
    voice, _ = align(voice_full, est)
    bed, _ = align(read_mono(bed_stem), est)

    # How much of the estimate is bed, and how much of the true voice survives.
    # Projecting the estimate onto each stem separates "kept" from "leaked"
    # without needing the separator to agree about gain.
    def proj_energy(sig: np.ndarray) -> float:
        s, e = align(sig, est)
        d = float(s @ s)
        if d <= 0:
            return -np.inf
        a = float(e @ s) / d
        return rms_db(a * s)

    return {
        "si_sdr_mix": si_sdr(mixa, voice),
        "si_sdr_est": si_sdr(est, voice),
        "lag": best_lag(read_mono(estimate), voice_full),
        "voice_keep_db": proj_energy(voice),
        "bed_keep_db": proj_energy(bed),
    }


def run(name: str, sep: Separator, cases: list[dict], fx: Path,
        out_root: Path, transcribe=None) -> list[dict]:
    rows = []
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        mix = fx / case["file"]
        est = sep(mix, out_dir)
        s = score(mix, fx / case["voice_stem"], fx / case["bed_stem"], est)
        s["case"] = case["file"]
        s["truth_lu"] = case["speech_minus_bed_lu"]
        s["estimate"] = str(est)
        if transcribe is not None:
            ref = transcribe(fx / case["voice_stem"])
            s["wer_mix"] = transcribe.wer(ref, transcribe(mix))
            s["wer_est"] = transcribe.wer(ref, transcribe(est))
        rows.append(s)
    return rows


def table(name: str, rows: list[dict]) -> str:
    out = [f"\n=== {name} ==="]
    out.append(f"{'case':<18}{'SI-SDR mix':>11}{'SI-SDR est':>11}{'gain':>7}"
               f"{'voice kept':>11}{'bed kept':>10}"
               + ("      WER mix   WER est" if "wer_mix" in (rows[0] if rows else {}) else ""))
    for r in rows:
        line = (f"{r['case']:<18}{r['si_sdr_mix']:11.2f}{r['si_sdr_est']:11.2f}"
                f"{r['si_sdr_est'] - r['si_sdr_mix']:+7.2f}"
                f"{r['voice_keep_db']:11.1f}{r['bed_keep_db']:10.1f}")
        if "wer_mix" in r:
            line += f"      {r['wer_mix']:8.3f}  {r['wer_est']:8.3f}"
        out.append(line)
    gains = [r["si_sdr_est"] - r["si_sdr_mix"] for r in rows if np.isfinite(r["si_sdr_est"])]
    if gains:
        out.append(f"{'mean SI-SDR gain':<18}{np.mean(gains):+11.2f} dB")
    return "\n".join(out)
