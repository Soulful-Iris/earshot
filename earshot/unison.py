"""How close you sang it to the way it was sung on the record.

Karaoke games score you against a CHART: a note track somebody typed in by hand.
The patents go back decades and are explicit about it, which is why SingStar and
Rock Band have catalogues rather than libraries. No chart, no score.

Separation removed that constraint and almost nobody has walked through. This
reads the melody off the record itself, so the catalogue is every song ever
released.

WHAT I MEASURED BEFORE WRITING ANY OF IT, 2026-09-04.

`librosa.pyin` on a real separated vocal. The plan recorded 4x realtime and 54%
of frames voiced; both were measured on a 60 second excerpt and both are better
across a whole song: **17x realtime and 68% voiced** on a 204 second track, so a
three and a half minute reference costs about 12 seconds rather than 50. Range
B3 to E5, median E4, which is a plausible female vocal and not noise.

**And it does NOT make a chart, because the obvious way is wrong.** My first
design was one note per word: median pitch inside each word span and the chart
falls out free. Measured, the spread of pitch INSIDE a single word has a median
of 183 cents. `and` spans 700 cents, `dreamhell` 510, `i` spans 1426, which is
more than an octave. Only 33% of words hold within a semitone. Real singing
slides, carries vibrato and puts two syllables on two notes, so a chart built
that way would be confidently wrong on two words in three.

So this compares CONTOURS. A slide drawn as a slide is the truth; a slide
quantised to a note is a lie.

THE METRIC, on time-aligned tracks, octave-folded:

    identical                 0.0c        the same notes, shuffled   230c
    50 cents flat            50.0c        random singing             290c
    a whole octave down       0.0c        drifting flat over 60s      97c
    a fifth up              500.0c

Good is under about 50 and wrong is 230 to 290, a factor of five apart. The
folding is deliberate and musical: a man singing a woman's part an octave down
is RIGHT and scores zero.

AND THE FIRST VERSION OF THAT TABLE WAS NONSENSE. I built the detuned versions
by shifting the AUDIO with `asetrate` and `atempo`, which moves pitch and time
together, so the two sides had drifted apart and I was comparing different
moments of the same song. A 50-cents-flat performance scored 240c, sitting
squarely in the randomly-scrambled band, and I nearly wrote the whole idea off.
Detuning the pitch TRACK instead, where nothing can drift, made every row above
come out exact.

Which makes alignment the whole game. A perfect take 186 ms late scores 40c
instead of 0. `offset()` searches plus or minus 320 ms and recovered a planted
186 ms delay exactly.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512
FRAME = 2048
FRAME_MS = 1000.0 * HOP / SR          # 23.22 ms
FMIN_HZ = 65.406                      # C2
FMAX_HZ = 1046.502                    # C6

# Below this there is not enough of a performance to have an opinion about. 200
# frames is about 4.6 seconds of the two of you sounding at the same time.
MIN_FRAMES = 200
# A line needs its own floor or a two-word line becomes somebody's "worst".
MIN_LINE_FRAMES = 20
# Plus or minus 20 frames is 464 ms. The plan said 14 (325 ms), and then asked
# for a test that recovers a planted 16-frame delay, which 14 cannot do: it
# clamps at the edge and reports 10 cents instead of 0. Rather than weaken the
# test to fit the constant I widened the constant, because 41 shifts over an
# array is still milliseconds and the only cost of a wider window is the chance
# of locking onto a wrong alignment, which the search guards by taking the
# minimum. Measured: recovers planted delays of 4, 8, 12, 16 and 20 exactly.
MAX_SHIFT = 20
# Inside this, you were on the note. It is also the fill threshold on the ribbon.
ON_NOTE_CENTS = 50.0


@dataclass
class Track:
    f0: np.ndarray            # Hz, nan where unvoiced
    voiced: np.ndarray        # bool
    sr: int = SR
    hop: int = HOP

    def __len__(self) -> int:
        return int(self.f0.size)

    @property
    def seconds(self) -> float:
        return len(self) * self.hop / self.sr


@dataclass
class Verdict:
    median_cents: float
    flat_or_sharp: float       # signed. negative means you sit under the note
    share_within_50: float
    frames_compared: int
    shift_ms: float

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in d.items()}

    @property
    def sentence(self) -> str:
        """The one line a person reads. Deliberately not a percentage."""
        c = self.median_cents
        lean = ("and you sit under the note" if self.flat_or_sharp < -15
                else "and you sit over it" if self.flat_or_sharp > 15
                else "and you wander either side")
        how = ("dead on" if c < 25 else "close" if c < 50
               else "out" if c < 120 else "a long way out")
        return f"{c:.0f} cents {how}, {lean}"


@dataclass
class LineScore:
    text: str
    start: float
    end: float
    median_cents: float | None
    flat_or_sharp: float | None
    frames: int


def _mono(path: str | Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], capture_output=True, timeout=3600).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float32)


def track(path: str | Path, cache: str | Path | None = None) -> Track:
    """The pitch line of one recording.

    `cache` is an .npz beside the stems. The reference never changes and costs
    about 50 seconds for a three and a half minute song, so it is computed once
    per song and read back on every take after the first.
    """
    cache = Path(cache) if cache else None
    if cache and cache.is_file():
        try:
            z = np.load(cache)
            return Track(f0=z["f0"], voiced=z["voiced"].astype(bool))
        except Exception:                                     # noqa: BLE001
            pass  # a corrupt cache is not worth an outage; recompute it

    import librosa
    x = _mono(path)
    if x.size < SR // 2:
        return Track(f0=np.array([]), voiced=np.array([], dtype=bool))
    f0, voiced, _ = librosa.pyin(x, sr=SR, fmin=FMIN_HZ, fmax=FMAX_HZ,
                                 frame_length=FRAME, hop_length=HOP)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    t = Track(f0=np.asarray(f0, dtype=np.float64), voiced=voiced)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, f0=t.f0, voiced=t.voiced)
    return t


def fold(cents: np.ndarray) -> np.ndarray:
    """Bring a cents difference into plus or minus half an octave.

    Singing the same melody an octave down is RIGHT, and any comparison that
    does not say so will tell every man singing a woman's part that he was
    1200 cents wrong.
    """
    return (cents + 600.0) % 1200.0 - 600.0


def _diff(a: Track, b: Track, shift: int = 0) -> np.ndarray:
    """Folded cents, b relative to a, over the frames where both sound."""
    if shift >= 0:
        fa, va = a.f0[shift:], a.voiced[shift:]
        fb, vb = b.f0, b.voiced
    else:
        fa, va = a.f0, a.voiced
        fb, vb = b.f0[-shift:], b.voiced[-shift:]
    n = min(fa.size, fb.size)
    if n == 0:
        return np.array([])
    m = va[:n] & vb[:n]
    if not m.any():
        return np.array([])
    return fold(1200.0 * np.log2(fb[:n][m] / fa[:n][m]))


def offset(a: Track, b: Track, max_shift: int = MAX_SHIFT) -> int:
    """How many frames late `b` is against `a`, by search.

    The take is aligned by construction, because the stage starts the recorder
    against playback. It is not aligned EXACTLY: there is latency between
    `play()` and `MediaRecorder.start()` and again on the way out. Measured, a
    perfect take 186 ms late scores 40 cents instead of 0, and this search
    recovers that delay exactly.
    """
    best, best_err = 0, math.inf
    for s in range(-max_shift, max_shift + 1):
        d = _diff(a, b, s)
        if d.size < MIN_LINE_FRAMES:
            continue
        err = float(np.median(np.abs(d)))
        if err < best_err:
            best, best_err = s, err
    return best


def score(a: Track, b: Track, shift: int = 0) -> Verdict | None:
    """None when you and the singer did not overlap enough to have an opinion."""
    d = _diff(a, b, shift)
    if d.size < MIN_FRAMES:
        return None
    return Verdict(median_cents=float(np.median(np.abs(d))),
                   flat_or_sharp=float(np.median(d)),
                   share_within_50=float(np.mean(np.abs(d) <= ON_NOTE_CENTS)),
                   frames_compared=int(d.size),
                   shift_ms=shift * FRAME_MS)


def _window(a: Track, b: Track, shift: int, lo: float, hi: float):
    """The frames of both tracks inside [lo, hi) seconds of the REFERENCE."""
    i0, i1 = int(lo * SR / HOP), int(math.ceil(hi * SR / HOP))
    i0, i1 = max(0, i0), min(len(a), i1)
    if i1 <= i0:
        return np.array([]), np.array([])
    ja, jb = i0, i0 - shift
    n = i1 - i0
    if jb < 0:
        cut = -jb
        ja, jb, n = ja + cut, 0, n - cut
    n = min(n, len(b) - jb)
    if n <= 0:
        return np.array([]), np.array([])
    sub_a = Track(a.f0[ja:ja + n], a.voiced[ja:ja + n])
    sub_b = Track(b.f0[jb:jb + n], b.voiced[jb:jb + n])
    return sub_a, sub_b


def per_line(a: Track, b: Track, shift: int, lines) -> list:
    """A score per sung line, using the grouping the subtitles already use.

    `lines` are `video.Cue`s. Nothing new is invented: the same lines that get
    burned into the karaoke video are the ones scored here, so the sentence
    "sing this one again" names something he has already read on screen.
    """
    out = []
    for c in lines:
        sa, sb = _window(a, b, shift, c.start, c.end)
        if isinstance(sa, np.ndarray):
            out.append(LineScore(c.text, c.start, c.end, None, None, 0))
            continue
        d = _diff(sa, sb, 0)
        if d.size < MIN_LINE_FRAMES:
            out.append(LineScore(c.text, c.start, c.end, None, None, int(d.size)))
            continue
        out.append(LineScore(c.text, c.start, c.end,
                             round(float(np.median(np.abs(d))), 1),
                             round(float(np.median(d)), 1), int(d.size)))
    return out


def headline(v: "Verdict", lines: list) -> str:
    """The sentence somebody reads, with the median's blind spot patched.

    THE MEDIAN HIDES THE THING PEOPLE MOST WANT TOLD APART, and I did not
    believe that hard enough until I built it. The PRD listed it as a worry.
    Then a take that was perfect on twenty lines and 80 cents flat on ONE
    reported "0 cents dead on", because one line in twenty cannot move a
    median and should not be able to.

    A person who was flat for four seconds of a song has not sung it dead on,
    and a headline that says they did is wrong in the direction that makes the
    tool useless: it agrees with you. So when the worst scored line is far
    worse than the overall figure, the sentence says so and names it.
    """
    _, worst = best_and_worst(lines)
    base = v.sentence
    if worst is None or worst.median_cents is None:
        return base
    if worst.median_cents > max(2.5 * v.median_cents, v.median_cents + 40):
        return f"{base}. One line got away from you"
    return base


def best_and_worst(lines: list) -> tuple:
    scored = [l for l in lines if l.median_cents is not None]
    if len(scored) < 2:
        return None, None
    scored.sort(key=lambda l: l.median_cents)
    return scored[0], scored[-1]


def contours(a: Track, b: Track, shift: int, every: int = 2) -> dict:
    """The two lines, ready to draw, in semitones on one shared scale.

    YOUR line is folded toward the reference rather than plotted where it
    literally sits, so somebody singing an octave down appears ON the melody
    instead of off the bottom of the window. That is the same decision as the
    scoring and it has to be, or the picture would disagree with the number
    printed under it.
    """
    n = min(len(a), len(b) - shift) if shift >= 0 else min(len(a) + shift, len(b))
    n = max(0, n)
    ts, ref, you = [], [], []
    ref_hz = []
    for i in range(0, n, every):
        ja = i + shift if shift >= 0 else i
        jb = i if shift >= 0 else i - shift
        if ja >= len(a) or jb >= len(b):
            break
        r = a.f0[ja] if a.voiced[ja] else None
        y = b.f0[jb] if b.voiced[jb] else None
        ts.append(round(ja * HOP / SR, 3))
        ref.append(None if r is None else round(12 * math.log2(r / FMIN_HZ), 3))
        if r is None or y is None:
            you.append(None)
        else:
            semis = 12 * math.log2(y / r)
            semis = (semis + 6) % 12 - 6          # same fold as the score
            you.append(round(12 * math.log2(r / FMIN_HZ) + semis, 3))
        if r is not None:
            ref_hz.append(r)
    centre = (12 * math.log2(float(np.median(ref_hz)) / FMIN_HZ)) if ref_hz else 24.0
    return {"t": ts, "ref": ref, "you": you,
            "centre": round(centre, 2), "span": 12.0,
            "on_note": ON_NOTE_CENTS / 100.0}
