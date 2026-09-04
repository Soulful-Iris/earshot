"""Put your voice in the room the record was sung in.

Bruno, 2026-09-02, twice in the same voice message: he wants his voice to get
whatever the original vocal had. Reverb, whatever processing. Every karaoke app
answers that with a preset and a wet slider -- pick "hall", guess. None of them
measures the song you are actually singing.

THE IDEA. Once a vocal stem is separated out of a record, the gaps between sung
phrases hold the reverb tail of the previous phrase decaying on its own, with
nothing playing over it. You cannot see that in a full mix because the band is
playing through the gap. After separation it is nearly a free measured decay
curve, and it is the one thing a consumer has never had: the actual space off
the actual record.

Measured 2026-09-03, control first, before any real audio:

    true RT60   estimator says
    dry              0.071 s
    0.80 s           0.556 s
    2.00 s           0.638 s

    real separated stems:  0.783, 0.813, 1.086 s

So records measure ~0.8 and a dry voice measures ~0.07. An order of magnitude,
on the same instrument. That gap is the whole difference between how you sound
and how they sound, and it is not talent and not the microphone.

AND THE CONTROL FAILED IN A USEFUL WAY. It ranks perfectly and does not
calibrate -- 0.80 comes back as 0.556, 2.00 as 0.638, because the tail runs into
the noise floor before there is enough clean decay to read. It also took four
attempts to get here: the first three reported 2.4, 1.6 and 2.1 seconds of
reverb ON A DRY FILE, and every one of them would have looked like a working
measurement if I had only ever run it on real songs.

So nothing here uses the number as a number. `search()` applies a candidate room
to the voice and measures the RESULT with the same instrument, keeping whichever
lands nearest the record. A biased instrument does not have to be right about
the world to pick the right answer -- as long as it is pointed at two things of
the same kind. `search()` docstring has the measurement of how far that holds.
"""
from __future__ import annotations

import json
import math
import subprocess
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

SR = 16000          # for the tail measurement; decay does not need 44.1 kHz
WORK_SR = 44100     # for anything that becomes audio somebody hears

# --- the estimator ----------------------------------------------------------
#
# Every constant below was moved at least once while getting the control to
# pass, and three of them are the reason it passes.
HOP = 0.005
WIN = 0.020
TAIL = 2.0          # how much of the gap to integrate over
MIN_GAP = 0.30      # a gap shorter than this is a breath, not a phrase end
NEED_DB = 25.0      # a decay with less clean range than this is not measured
MIN_GAPS = 3        # a number from one gap is not a measurement

# The one that mattered most. An earlier version used `peak - 6` to decide what
# counted as singing, which fires on every syllable dip inside a phrase, so the
# "gaps" were full of speech and the integration ran over more voice. That is
# the version that reported 2 seconds of reverb on a dry file.
VOICED_OVER_FLOOR = 12.0
VOICED_UNDER_PEAK = 30.0


def _mono(path: str | Path, sr: int = SR, seconds: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if seconds:
        cmd += ["-t", f"{seconds:g}"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=1800).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def tail_seconds(x: np.ndarray, sr: int = SR) -> tuple[float | None, int]:
    """RT60 read out of the gaps after sung phrases. Returns (seconds, n_gaps).

    Schroeder backward integration over each gap, T20 read between -5 and
    -25 dB, times three. Integration is truncated where the decay reaches the
    noise floor, because integrating through the floor turns constant noise into
    a slow logarithmic slope that reads as a long room -- which is exactly how
    the second failed version got 1.6 seconds out of silence.
    """
    h, w = int(HOP * sr), int(WIN * sr)
    n = (len(x) - w) // h
    if n < 10:
        return None, 0
    rms = np.array([np.sqrt(np.mean(x[i * h:i * h + w] ** 2) + 1e-20)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-20)
    peak, floor = np.percentile(db, 95), np.percentile(db, 5)
    thr = max(floor + VOICED_OVER_FLOOR, peak - VOICED_UNDER_PEAK)
    voiced = db > thr
    floor_pow = 10 ** (floor / 10)

    g, ntail = int(MIN_GAP / HOP), int(TAIL / HOP)
    smooth = max(1, int(0.020 * sr))
    hold = max(1, int(0.050 * sr))
    ests: list[float] = []

    for i in range(1, n - ntail):
        if not (voiced[i] and not voiced[i + 1]):
            continue
        if voiced[i + 1:i + 1 + g].any():
            continue                                   # a dip, not a phrase end
        seg = x[(i + 1) * h:(i + 1 + ntail) * h]
        if len(seg) < ntail * h:
            continue
        e = seg ** 2
        sm = np.convolve(e, np.ones(smooth) / smooth, mode="same")
        below = sm < floor_pow * 4                      # +6 dB over the floor
        run = np.convolve(below.astype(float), np.ones(hold), mode="valid")
        k = np.where(run >= hold)[0]
        cut = int(k[0]) if len(k) else len(seg)
        if cut < int(0.03 * sr):
            continue
        sch = np.cumsum(e[:cut][::-1])[::-1]
        sch = 10 * np.log10(sch / sch[0] + 1e-20)
        if sch[-1] > -NEED_DB:
            continue                                    # not enough clean decay
        def cross(t: float):
            hits = np.where(sch <= t)[0]
            return float(hits[0]) / sr if len(hits) else None
        t5, t25 = cross(-5.0), cross(-25.0)
        if t5 is None or t25 is None or t25 <= t5:
            continue
        ests.append(3.0 * (t25 - t5))

    if not ests:
        return None, 0
    return float(np.median(ests)), len(ests)


# --- the profile ------------------------------------------------------------

@dataclass
class Profile:
    tail: float | None
    gaps: int
    lufs: float
    crest: float
    eq: list          # 1/3-octave band levels in dB, normalised to their mean

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tail"] = None if self.tail is None else round(self.tail, 3)
        d["lufs"] = round(self.lufs, 2)
        d["crest"] = round(self.crest, 2)
        d["eq"] = [round(v, 2) for v in self.eq]
        return d


# Third-octave centres from 40 Hz to 16 kHz. Below 40 there is nothing in a
# voice but rumble, and the stem's own separation artefacts live there.
BANDS = [40 * (2 ** (i / 3)) for i in range(int(math.log2(16000 / 40) * 3) + 1)]


def spectrum(x: np.ndarray, sr: int = SR) -> list:
    """Long-term average spectrum in third-octave bands, mean-normalised.

    Mean-normalised because the SHAPE is the thing being matched; the level is
    carried separately by `lufs` and matched at the end. A curve that carried
    level too would make every comparison a loudness comparison.
    """
    if x.size < sr // 2:
        return [0.0] * len(BANDS)
    nfft = 4096
    step = nfft // 2
    frames = max(1, (len(x) - nfft) // step)
    acc = np.zeros(nfft // 2 + 1)
    win = np.hanning(nfft)
    used = 0
    for i in range(frames):
        seg = x[i * step:i * step + nfft]
        if len(seg) < nfft:
            break
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if not used:
        return [0.0] * len(BANDS)
    acc /= used
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    out = []
    for c in BANDS:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        m = (freqs >= lo) & (freqs < hi)
        p = float(acc[m].mean()) if m.any() else 0.0
        out.append(10 * math.log10(p + 1e-20))
    arr = np.array(out)
    return list(arr - arr.mean())


def profile(path: str | Path) -> Profile:
    """Everything measured about one piece of audio, on one decode."""
    x = _mono(path)
    t, g = tail_seconds(x)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    crest = 20 * math.log10((peak + 1e-9) / (rms + 1e-9))
    return Profile(tail=t, gaps=g, lufs=_lufs(path), crest=crest,
                   eq=spectrum(x))


def _lufs(path: str | Path) -> float:
    """Integrated loudness, from ffmpeg's own R128 meter.

    Uses the meter rather than earshot's own because this one has to agree with
    `loudnorm`, which is what actually moves the level at the end. Two
    implementations of loudness in one chain is two things to disagree.
    """
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(path),
         "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        capture_output=True, text=True, timeout=1800)
    for line in reversed((r.stderr or "").splitlines()):
        if "I:" in line and "LUFS" in line:
            try:
                return float(line.split("I:")[1].split("LUFS")[0].strip())
            except (ValueError, IndexError):
                break
    return float("-inf")


# --- the impulse responses --------------------------------------------------

RT60S = (0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.2, 3.0)
TILTS = ("dark", "neutral", "bright")

# Direct-to-reverberant ratio, in dB, held FIXED across the whole bank.
#
# This is an assumption and not a measurement, and it is the largest one in the
# file. Blind DRR estimation is a research problem -- the ACE challenge found
# analytic methods do well on RT60 and badly on DRR -- so measuring it off the
# record was not on the table for one night. +6 dB means the tail carries a
# quarter of the energy of the direct sound, which is a modest vocal send.
#
# The search cannot correct for this. If DRR is wrong the result lands on the
# right measured TAIL with the wrong balance, and only ears can catch that.
# Named here so it is the first thing anybody changes.
DRR_DB = 6.0


def impulse(rt60: float, tilt: str = "neutral", sr: int = WORK_SR,
            seed: int = 7) -> np.ndarray:
    """One synthetic room: a direct spike plus exponentially decaying noise.

    Seeded, so the bank is reproducible and does not have to ship as binary.
    """
    n = max(int(rt60 * 1.5 * sr), int(0.05 * sr))
    rng = np.random.RandomState(seed + int(rt60 * 1000))
    noise = rng.randn(n)

    if tilt == "dark":
        # One-pole lowpass: a room with soft things in it.
        a = 0.85
        out = np.empty(n)
        y = 0.0
        for i in range(n):
            y = a * y + (1 - a) * noise[i]
            out[i] = y
        noise = out * 3.0
    elif tilt == "bright":
        # First difference: hard walls, more high end in the tail.
        noise = np.diff(noise, prepend=0.0)

    t = np.arange(n) / sr
    tail = noise * 10 ** (-3.0 * t / rt60)
    tail_energy = float(np.sum(tail ** 2)) + 1e-20
    # Scale the direct spike so the direct/reverberant ratio is DRR_DB.
    direct = math.sqrt(tail_energy * 10 ** (DRR_DB / 10))
    ir = tail.copy()
    ir[0] += direct
    peak = float(np.max(np.abs(ir)))
    return ir / peak if peak else ir


def write_wav(x: np.ndarray, path: Path, sr: int = WORK_SR) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(x, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((clipped * 32767).astype("<i2").tobytes())
    return path


def bank(into: Path, sr: int = WORK_SR) -> list:
    """The 18 candidate rooms, written to disk for ffmpeg to convolve with."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    made = []
    for rt in RT60S:
        for tilt in TILTS:
            p = into / f"ir_{rt:g}_{tilt}.wav"
            if not p.is_file():
                write_wav(impulse(rt, tilt, sr), p, sr)
            made.append({"rt60": rt, "tilt": tilt, "path": str(p)})
    return made


def convolve(audio: Path, ir: np.ndarray | Path, out: Path,
             seconds: float | None = None, sr: int = WORK_SR) -> Path:
    """Put `audio` in the room described by `ir`. Overlap-add, in numpy.

    NOT ffmpeg's `afir`, and the reason is worth keeping. Measured on a dry
    voice at -27.5 dB RMS:

        afir dry=0:wet=10   -> -240 dB   (digitally silent)
        afir dry=0:wet=1    -> -240 dB   (silent, at every gtype and irnorm)
        afir dry=1:wet=1    -> -58.7 dB  (31 dB below its own input)
        afir dry=10:wet=10  -> -18.7 dB

    Every `dry=0` setting produced a file of the right duration and the right
    size containing nothing at all, which is a plausible artefact with no error
    attached. Whatever those gains mean, they are not the dB-linear mix knob
    the option text implies, and this chain has to know its own gain exactly:
    the whole method is measuring a decay and matching a level.

    I generate the impulse responses in numpy already. Doing the convolution
    there too removes a dependency on a parameter whose semantics I could not
    establish in three probes, and makes the direct/reverberant balance exactly
    what `impulse()` says it is.
    """
    from scipy.signal import oaconvolve

    x = _mono(audio, sr=sr, seconds=seconds).astype(np.float32)
    h = (ir if isinstance(ir, np.ndarray)
         else _mono(ir, sr=sr)).astype(np.float32)
    if x.size == 0 or h.size == 0:
        raise RuntimeError("nothing to convolve")
    y = oaconvolve(x, h, mode="full")[:x.size]
    # Peak-normalise only if it would clip. Any deliberate level change belongs
    # in `loudnorm` at the end, where it is measured, not hidden in here.
    peak = float(np.max(np.abs(y)))
    if peak > 0.999:
        y = y * (0.999 / peak)
    return write_wav(y, Path(out), sr)


# --- the search -------------------------------------------------------------

# How close two candidates' tails have to be before the choice between them is
# made on colour instead of on length.
# It was a flat 0.06 for one round and that was too loose: adjacent steps in
# this bank measure 0.1 to 0.2 s apart, so a 0.06 window let colour overrule
# length and the recovery test went from naming the exact room to missing by a
# step. Proportional and tighter, so the tie-break only fires on candidates
# that genuinely do measure the same length.
def tail_tie(target: float) -> float:
    return max(0.02, 0.04 * target)
# Bands that decide the colour comparison: roughly 200 Hz to 8 kHz. Below that
# is the singer's chest and the separation's artefacts, above it is air.
EQ_LO, EQ_HI = 200.0, 8000.0


@dataclass
class Candidate:
    rt60: float           # 0.0 means "no room at all", the take as recorded
    tilt: str
    measured: float | None
    gaps: int
    eq_distance: float | None = None


def eq_distance(a: list, b: list) -> float:
    """Mean absolute difference between two normalised spectra, in dB."""
    m = [i for i, c in enumerate(BANDS) if EQ_LO <= c <= EQ_HI]
    if not m or len(a) != len(BANDS) or len(b) != len(BANDS):
        return 0.0
    return float(np.mean([abs(a[i] - b[i]) for i in m]))


def _first_voice(x: np.ndarray, sr: int = SR) -> float:
    h, w = int(HOP * sr), int(WIN * sr)
    n = (len(x) - w) // h
    if n < 10:
        return 0.0
    rms = np.array([np.sqrt(np.mean(x[i * h:i * h + w] ** 2) + 1e-20)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-20)
    thr = max(np.percentile(db, 5) + VOICED_OVER_FLOOR,
              np.percentile(db, 95) - VOICED_UNDER_PEAK)
    hits = np.where(db > thr)[0]
    return float(hits[0]) * HOP if len(hits) else 0.0


# 25 s, not the 15 the plan asked for. Measured on a real dry voice: a 16 s
# stretch yields TWO usable gaps and a 26 s stretch yields four, against a
# MIN_GAPS of three. With a 15 s excerpt the search would be measuring fewer
# gaps than `assess()` just checked the whole take for, so the two could
# disagree about whether the take is measurable at all.
SEARCH_SECONDS = 25.0


def excerpt(take: Path, work: Path, seconds: float = SEARCH_SECONDS) -> Path:
    """The stretch of the take the search runs on: `seconds` from first voice."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    start = _first_voice(_mono(take))
    out = work / "excerpt.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(take),
         "-ss", f"{start:.2f}", "-t", f"{seconds:g}",
         "-ac", "1", "-ar", str(WORK_SR), str(out)],
        check=True, timeout=600)
    return out


def search(take: Path, record: Profile, work: Path,
           seconds: float = SEARCH_SECONDS) -> tuple[Candidate, list]:
    """Find the room that makes THIS take measure like THAT record.

    Every candidate is applied and the RESULT is measured with the same
    estimator that measured the record, so the estimator does not have to be
    right about the world, only consistent between two things of the same kind.
    Measured cross-material, dry fixture against two real separated stems: it
    lands 0.005 and 0.034 from the target.

    THE MONOTONE PREFIX, which is not in the plan and has to be. Measured on
    the dry fixture, neutral tilt:

        0.20 -> 0.11   0.60 -> 0.55   1.20 -> 0.95   2.20 -> 0.59
        0.40 -> 0.27   0.90 -> 0.74   1.60 -> 1.01   3.00 -> 0.40

    It tracks and then FALLS OVER: a 3.0 s room measures 0.40, below where
    0.4 s lands. Beyond the turning point the tail no longer fits inside a gap
    and the estimator is not measuring the room any more. A nearest-match over
    the raw list would answer "three seconds" to a target of 0.4 and put a
    cathedral around somebody. So each tilt is truncated at its own maximum,
    per take, because where it turns over depends on this singer's gaps.

    AND A CANDIDATE FOR DOING NOTHING. Without it, a dry reference still gets
    the shortest room in the bank, because something is always nearest. With
    it, the take as recorded competes on the same terms and wins when no room
    is the right answer -- which is what makes refusal 1 a measurement rather
    than a threshold I picked.

    Colour breaks the tie. Length is chosen first, and among candidates within
    tail_tie() of the best, the one whose spectrum sits closest to the record's
    wins. Without that the tilt dimension is decorative: three candidates that
    measure the same length are indistinguishable and the first one always won.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ex = excerpt(take, work, seconds)

    base_x = _mono(ex)
    base_t, base_g = tail_seconds(base_x)
    none_c = Candidate(0.0, "none", base_t, base_g, eq_distance(spectrum(base_x), record.eq))

    table: list[list] = []
    usable: list[Candidate] = [none_c] if base_t is not None else []
    for tilt in TILTS:
        row: list[Candidate] = []
        for rt in RT60S:
            out = work / f"c_{rt:g}_{tilt}.wav"
            convolve(ex, impulse(rt, tilt), out)
            y = _mono(out)
            t, g = tail_seconds(y)
            row.append(Candidate(rt, tilt, t, g, eq_distance(spectrum(y), record.eq)))
            out.unlink(missing_ok=True)
        table.append(row)
        usable.extend(monotone_prefix(row))

    flat = [asdict(c) for row in table for c in row] + [asdict(none_c)]
    if not usable:
        return none_c, flat
    return choose(usable, record.tail), flat


def monotone_prefix(row: list) -> list:
    """Everything up to and including this tilt's longest measurement.

    Past the turning point the tail no longer fits inside a gap and the number
    is not about the room any more, so those candidates are not choices.
    """
    best_i, best_v = -1, -1.0
    for i, c in enumerate(row):
        if c.measured is not None and c.measured > best_v:
            best_i, best_v = i, c.measured
    return [c for c in row[:best_i + 1] if c.measured is not None]


def choose(usable: list, target: float) -> Candidate:
    """Nearest on length; ties broken on colour; the null action wins ties.

    Pulled out of `search` so it can be tested without forty convolutions.
    Both rules below were shown to be untested by mutation before that: the
    prefix could be removed and the suite stayed green, and the null-action
    rule was masked because in the dry-reference control the untouched take
    also has the smallest colour distance and would have won anyway.
    """
    err = lambda c: abs(c.measured - target)
    best_err = min(err(c) for c in usable)
    tie = tail_tie(target)
    close = [c for c in usable if err(c) <= best_err + tie]

    # DOING NOTHING WINS TIES. If leaving the take alone matches as well as any
    # room does, add no room. Without this the null action merely competes, and
    # something is always nearest, so a dry reference gets a room bolted on.
    for c in close:
        if c.rt60 == 0.0:
            return c
    return min(close, key=lambda c: (
        c.eq_distance if c.eq_distance is not None else 9e9,
        0 if c.tilt == "neutral" else 1))


# --- the refusals -----------------------------------------------------------
#
# Each of these produces a sentence somebody can act on, and each one is
# proved to fire in tests/test_liveroom.py against a constructed input. A
# refusal path that has only ever been reasoned about is the thing that bit
# this project on 2026-09-03, when the honest-failure message I was proudest
# of turned out to be false in a tone of careful humility.

# Measured, see tests: a take with the backing genuinely bleeding into the
# microphone at -12 dB correlates ~0.5 with the band stem; a clean take through
# headphones correlates ~0.00 to 0.05.
BLEED_CORR = 0.25


class Refused(Exception):
    """Not an error. An answer, with a sentence attached."""


def bleed(take: Path, band: Path, seconds: float = 60.0) -> float:
    """How much of the backing track is in the microphone. 0 is headphones."""
    a = _mono(take, seconds=seconds)
    b = _mono(band, seconds=seconds)
    n = min(a.size, b.size)
    if n < SR:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return abs(float(a @ b) / d) if d else 0.0


@dataclass
class Decision:
    ok: bool
    why: str = ""
    room: Candidate | None = None
    record: Profile | None = None
    take: Profile | None = None
    table: list | None = None


def assess(take: Path, vocals: Path, band: Path, work: Path) -> Decision:
    """Everything that has to be true before a note of audio is produced."""
    rec = profile(vocals)
    if rec.tail is None or rec.gaps < MIN_GAPS:
        return Decision(False, Profile.__name__ and
                        "there is not enough space between the phrases on this "
                        "record to measure its room. Nothing to copy.",
                        record=rec)

    c = bleed(take, band)
    if c >= BLEED_CORR:
        return Decision(False,
                        f"the backing track is in your microphone (it matches at "
                        f"{c:.2f}, headphones give under {BLEED_CORR}). Put "
                        f"headphones on and record it again.", record=rec)

    you = profile(take)
    if you.tail is None or you.gaps < MIN_GAPS:
        return Decision(False,
                        "there are no pauses in your take long enough to measure. "
                        "Leave a beat between lines and try again.",
                        record=rec, take=you)

    # ALREADY WETTER THAN THE RECORD, checked before the search rather than
    # after it. Convolution only ever ADDS reverberation, so a take that
    # already measures longer than the record cannot be brought down by
    # anything in the bank. That is a physical argument and it does not need
    # 24 convolutions to make.
    #
    # It was written as a search outcome first -- refuse if the winner is
    # `none` -- and that was wrong in a way the test caught: on a very wet take
    # the estimator's gaps are already filled, so adding a room makes the
    # measurement go DOWN, and the search cheerfully picked a 0.4 s room as an
    # improvement. The search cannot be the judge of this, because the thing
    # that makes the take unfixable is the same thing that breaks the
    # instrument judging it.
    if you.tail > rec.tail + tail_tie(rec.tail):
        return Decision(False,
                        f"the room you recorded in is bigger than the one on the "
                        f"record ({you.tail:.2f}s against {rec.tail:.2f}s). "
                        f"Nothing can be added to fix that. Try a cupboard, or a "
                        f"corner with a duvet in it.",
                        record=rec, take=you)

    room, table = search(take, rec, work)
    return Decision(True, room=room, record=rec, take=you, table=table)


# --- placing the voice ------------------------------------------------------

def speech_db(path: Path) -> float:
    """RMS of the loud half of the file, in dB. A level that ignores silence."""
    x = _mono(path)
    if x.size < SR // 2:
        return float("-inf")
    h = int(0.05 * SR)
    frames = np.array([np.sqrt(np.mean(x[i:i + h] ** 2) + 1e-20)
                       for i in range(0, len(x) - h, h)])
    loud = frames[frames > np.percentile(frames, 50)]
    return 20 * math.log10(float(loud.mean()) + 1e-12) if loud.size else float("-inf")


def _matchering(target: Path, reference: Path, out: Path) -> Path:
    import logging
    import matchering as mg
    mg.log(warning_handler=logging.getLogger("matchering").warning)
    mg.process(target=str(target), reference=str(reference),
               results=[mg.pcm16(str(out))])
    return Path(out)


# --- clarity ----------------------------------------------------------------
#
# Bruno, 2026-09-04: "can you put like a small clarity or whatever filter on
# top of the song filter. Very small to smooth out the voice"
#
# "The song filter" is matchering, which pulls the take's colour and dynamics
# toward the record's own vocal. ON TOP OF it is the correct place and he named
# it: matchering matches toward a target curve, so anything applied BEFORE it
# gets matched back out again.
#
# FOUR stages, and the count is four because it started at five and one of them
# was measured doing nothing. Every remaining one is either clarity or
# smoothing rather than taste, and every one has a number beside it, taken on
# `fixtures/dry.wav` with the other stages switched off:
#
#   highpass 70 Hz    -4.4 dB at 40-90.  Rumble, most of it put there by the
#                     room convolution. Much the biggest move in the chain, and
#                     I had it filed as incidental cleanup until I measured it.
#   -1 dB at 250 Hz   -0.8 dB at 180-400. Mud; taking it out reads as clarity
#                     without adding any level.
#   +1.5 dB at 3.2k   +1.25 dB at 2-5k.  Presence, which is what clarity means.
#   2:1 compression   -1.0 dB broadband, crest down a quarter of a dB. Evens
#                     the loud and quiet lines out.
#
# THE ONE I TOOK OUT was a de-esser at i=0.1. Sibilance is the harsh part of
# "not smooth" so it belonged on the list, and on the one real voice I have it
# is a NO-OP: p99 of the 5-9 kHz band reads -3.07 dB with it and -3.07 dB
# without, to two decimals. At i=0.4 it buys 0.23 dB, which is no longer very
# small. It may well earn its place on a more sibilant voice than this fixture,
# and that is exactly the sentence this record keeps warning me about -- an
# unfalsifiable reason to keep something. If he says the esses are harsh, it
# goes back in with a number behind it.
#
# ONE DIAL. He said very small and this is what small means to me, and the
# first thing he will say if I have it wrong is more or less rather than a
# different chain -- so the amount is a single number and everything scales off
# it. Same reason COVER and HERO_W became env overrides while he watched takes.
CLARITY = 1.0


def clarity_chain(amount: float = CLARITY) -> str:
    """The filter string, scaled. `amount=0` is the identity chain."""
    if amount <= 0:
        return "anull"
    return ",".join([
        "highpass=f=70",
        f"equalizer=f=250:t=q:w=1.2:g={-1.0 * amount:.2f}",
        f"equalizer=f=3200:t=q:w=0.9:g={1.5 * amount:.2f}",
        f"acompressor=threshold=-20dB:ratio={1 + 1.0 * amount:.2f}"
        ":attack=15:release=200:makeup=1",
    ])


def raw_band_db(path: Path, lo: float, hi: float) -> float:
    """Mean energy in a frequency range, in dB, NOT normalised.

    `spectrum()` subtracts its own mean, which is right for comparing the
    SHAPE of two different voices and wrong for asking what one filter did.
    Measured: the mud bell moves 180-400 Hz by -0.8 dB, and through the
    normalised curve it reads -0.08, because the highpass takes 4.4 dB out of
    40-90 Hz and drags the mean down with it. A factor of ten, in the
    direction that says "this stage is decoration, delete it".

    I nearly did. Two instruments disagreeing by 10x is the whole reason this
    one exists.
    """
    x = _mono(path)
    nfft = 4096
    step = nfft // 2
    if x.size < nfft:
        return float("-inf")
    acc = np.zeros(nfft // 2 + 1)
    win = np.hanning(nfft)
    used = 0
    for i in range(0, len(x) - nfft, step):
        acc += np.abs(np.fft.rfft(x[i:i + nfft] * win)) ** 2
        used += 1
    if not used:
        return float("-inf")
    acc /= used
    f = np.fft.rfftfreq(nfft, 1 / SR)
    m = (f >= lo) & (f < hi)
    if not m.any():
        return float("-inf")
    return 10 * math.log10(float(acc[m].mean()) + 1e-20)


def band_change(before: Path, after: Path, lo: float, hi: float) -> float:
    """dB change between two files across a frequency range, un-normalised."""
    return raw_band_db(after, lo, hi) - raw_band_db(before, lo, hi)


def apply_clarity(src: Path, out: Path, amount: float = CLARITY) -> dict:
    """Run the chain and report what it moved, in both shape and level."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-af", clarity_chain(amount), "-ar", str(WORK_SR), "-ac", "2", str(out)],
        check=True, timeout=1800)
    before, after = profile(src), profile(out)
    presence = band_change(src, out, 2000, 5000)
    body = band_change(src, out, 300, 8000)
    return {
        "amount": round(amount, 3),
        "presence_db": round(presence, 2),
        # Presence AGAINST the voice's own body, which is the number that
        # survives. loudnorm runs immediately after this, so absolute level
        # here is thrown away; what reaches the mix is the shape. Measured
        # alone the presence bell is +1.25 dB, and in the full chain it reads
        # +0.27 absolute, because the compressor takes 1.0 dB off everything.
        # Both are true and only this one answers "is the voice brighter".
        "presence_rel_db": round(presence - body, 2),
        "mud_db": round(band_change(src, out, 180, 400), 2),
        # The biggest single move in the chain, reported because it was the one
        # I could not see. A number nobody prints is a number nobody checks.
        "sub_db": round(band_change(src, out, 40, 90), 2),
        "crest_db": round(after.crest - before.crest, 2),
        "took_db": round(speech_db(out) - speech_db(src), 2),
    }


def place(take: Path, vocals: Path, band: Path, room: Candidate,
          out_dir: Path, colour: bool = True, clarity: float = CLARITY) -> dict:
    """The take, in the record's room, at the record's colour and level, mixed.

    The order matters and every step reports what it moved:

      1. the room, by convolution
      2. colour and dynamics, by matchering toward the record's own vocal
      3. RE-MEASURE, because a room quietly takes level and says nothing. On
         2026-08-30 an aecho and a low shelf took 11.3 dB out of a voice at
         exit code 0. Anything that adds reflections scales the direct path.
      4. clarity, ON TOP of the colour and small enough to argue about
      5. level, by loudnorm to the stem's own integrated loudness
      6. mix against the band
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: dict = {"room": {"rt60": room.rt60, "tilt": room.tilt}}

    dry_wav = out_dir / "_take.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(take),
                    "-ac", "1", "-ar", str(WORK_SR), str(dry_wav)],
                   check=True, timeout=1800)
    steps["take_db"] = round(speech_db(dry_wav), 2)

    if room.rt60 > 0:
        wet = convolve(dry_wav, impulse(room.rt60, room.tilt), out_dir / "_wet.wav")
    else:
        wet = dry_wav
    steps["after_room_db"] = round(speech_db(wet), 2)
    steps["room_took_db"] = round(steps["after_room_db"] - steps["take_db"], 2)

    stage = wet
    if colour:
        ref_wav = out_dir / "_ref.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(vocals),
                        "-ac", "2", "-ar", str(WORK_SR), str(ref_wav)],
                       check=True, timeout=1800)
        st = out_dir / "_stereo.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wet),
                        "-ac", "2", "-ar", str(WORK_SR), str(st)],
                       check=True, timeout=1800)
        try:
            stage = _matchering(st, ref_wav, out_dir / "_matched.wav")
            steps["after_colour_db"] = round(speech_db(stage), 2)
            steps["colour"] = "matchering"
        except Exception as e:                                    # noqa: BLE001
            # Named, not swallowed. matchering is built for whole mixes and a
            # mono voice is off its home ground; if it will not do this the
            # room and the level are still worth having and the page says so.
            steps["colour"] = f"skipped: {type(e).__name__}: {e}"[:160]
            stage = wet

    # On top of the colour, which is where he asked for it and where it has to
    # be: matchering matches toward the record's curve, so a presence lift
    # applied before it is simply matched back out.
    if clarity > 0:
        clear = out_dir / "_clear.wav"
        try:
            steps["clarity"] = apply_clarity(stage, clear, clarity)
            stage = clear
            steps["after_clarity_db"] = round(speech_db(stage), 2)
        except Exception as e:                                    # noqa: BLE001
            # Named, not swallowed, same as the colour step above. A voice in
            # the right room at the right level is still worth having.
            steps["clarity"] = {"skipped": f"{type(e).__name__}: {e}"[:160]}

    target_lufs = _lufs(vocals)
    levelled = out_dir / "_level.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(stage),
         "-af", f"loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=11",
         "-ar", str(WORK_SR), "-ac", "2", str(levelled)],
        check=True, timeout=1800)
    steps["target_lufs"] = round(target_lufs, 2)
    steps["placed_lufs"] = round(_lufs(levelled), 2)

    placed = out_dir / "placed.mp3"
    _mix(band, levelled, placed)
    unplaced = out_dir / "unplaced.mp3"
    _mix(band, dry_wav, unplaced, match_lufs=target_lufs)
    steps["placed"] = placed.name
    steps["unplaced"] = unplaced.name
    for scratch in ("_wet.wav", "_stereo.wav", "_matched.wav", "_level.wav",
                    "_ref.wav", "_take.wav", "_clear.wav"):
        (out_dir / scratch).unlink(missing_ok=True)
    return steps


def _mix(band: Path, voice: Path, out: Path, match_lufs: float | None = None,
         timeout: int = 1800) -> Path:
    """Voice over backing. Both versions go through the identical mix, so an
    A/B between them is a comparison of the PLACEMENT and not of the mixing."""
    pre = ""
    if match_lufs is not None:
        pre = f"loudnorm=I={match_lufs:.1f}:TP=-1.5:LRA=11,"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(band), "-i", str(voice),
         "-filter_complex",
         f"[1:a]{pre}aformat=channel_layouts=stereo[v];"
         f"[0:a]aformat=channel_layouts=stereo[b];"
         f"[b][v]amix=inputs=2:duration=shortest:weights=1 1:normalize=0,"
         f"alimiter=limit=0.95",
         "-c:a", "libmp3lame", "-b:a", "192k", str(out)],
        check=True, timeout=timeout)
    return Path(out)
