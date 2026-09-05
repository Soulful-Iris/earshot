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

# --- timing -----------------------------------------------------------------
#
# Bruno, 2026-09-05: "you have to make sure im on the correct tempo with the
# lyrics"
#
# He is right that nothing here could tell him. The global `offset()` search
# does not measure timing, it DELETES it: one shift for the whole take, found
# once and subtracted so that headphone latency is not scored as flat singing.
# Every line is then compared at that one alignment, and any per-line drift is
# folded into the pitch number as though it were tuning.
#
# So the timing is the RESIDUAL: run the same search again inside each line, on
# windows the global shift has already aligned. What is left is how early or
# late that line was against his own baseline, which is the only version of the
# question that is his fault rather than his hardware's.
#
# WHICH OBSERVABLE, measured rather than assumed. I expected pitch matching to
# be poor per line -- sliding a window over a sustained note barely changes the
# pitch difference -- and expected a voiced-envelope onset correlation to beat
# it. Planted known per-line delays into four real vocal stems:
#
#     within 1 frame      pitch          onset
#     4 songs, 129 lines  127 (98.4%)    116 (89.9%)
#
# Pitch won, so this reuses `offset()` rather than adding a second mechanism.
#
# AND THE FIRST CONTROL WAS BROKEN, which is why those are the second numbers.
# It planted delays by overwriting each voiced run with a shifted slice of
# itself, so `voiced` stayed all-True and THE ONSETS NEVER MOVED. It then
# reported that the onset estimator could not recover the delays: true, and
# meaningless, because there was nothing in the signal it reads. Both scored
# about 15% and I nearly concluded neither was usable.
#
# Plus or minus 12 frames is 279 ms, wider than any timing error somebody makes
# on a line they know and narrow enough not to lock onto the next syllable.
LINE_SHIFT = 12
# Under this much pitch agreement the alignment is not measuring time. A line
# sung to a different tune has an arbitrary best shift, and a confident
# millisecond figure on top of that is a lie with a decimal point in it.
TIMING_NEEDS_CENTS = 200.0


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
    frames: int                # frames where BOTH of you were sounding
    ref_frames: int = 0        # frames where the SINGER was sounding
    # Positive means you came in LATE on this line, in milliseconds, after your
    # own latency has been taken out. None when the line was not sung, was too
    # short, or was sung far enough off the tune that its best alignment is not
    # measuring time at all.
    late_ms: float | None = None

    @property
    def unsung(self) -> bool:
        """The singer sang here and you did not.

        The distinction matters and conflating it was the bug Wren found. A
        line can score None for two completely different reasons: you skipped
        it, or the reference has nothing in it either because the transcript
        put words where the vocal stem is silent. Only the first is your
        fault, and only the first should count against you.
        """
        return self.median_cents is None and self.ref_frames >= MIN_LINE_FRAMES


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
    """The shift that ALIGNS `b` to `a`, by search. Feed it to `_diff`.

    SIGN, measured rather than described, because this docstring used to read
    "how many frames late b is against a" and that is the negation of what it
    returns. A take arriving 8 frames late gives -8.

        b late by  -8 frames -> offset(a, b) = +8
        b late by  +8 frames -> offset(a, b) = -8

    The label was harmless for as long as nothing read the number as a
    quantity: `_diff` consumes it to cancel the delay, so any consistent sign
    aligns correctly, and `shift_ms` was reported without anybody interpreting
    its direction. It stopped being harmless the moment `late_ms` tried to tell
    a person whether they were behind the beat, which is why `line_late_ms`
    negates it. Do not "tidy" that minus sign away.

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

    Each line also records how many frames the SINGER was sounding for, which
    is what makes "you skipped this" separable from "there was nothing here".
    """
    out = []
    for c in lines:
        sa, sb = _window(a, b, shift, c.start, c.end)
        if isinstance(sa, np.ndarray):
            out.append(LineScore(c.text, c.start, c.end, None, None, 0, 0))
            continue
        ref_frames = int(sa.voiced.sum())
        d = _diff(sa, sb, 0)
        if d.size < MIN_LINE_FRAMES:
            out.append(LineScore(c.text, c.start, c.end, None, None,
                                 int(d.size), ref_frames))
            continue
        cents = float(np.median(np.abs(d)))
        out.append(LineScore(c.text, c.start, c.end,
                             round(cents, 1),
                             round(float(np.median(d)), 1), int(d.size),
                             ref_frames,
                             late_ms=line_late_ms(sa, sb, cents)))
    return out


def line_late_ms(sa: Track, sb: Track, cents: float) -> float | None:
    """How late this line was, in ms, against the take's own baseline.

    `sa` and `sb` come out of `_window()`, which has already applied the global
    shift, so a further search between them returns the RESIDUAL: the part of
    the timing that is this line rather than the whole take's latency.

    Refuses rather than guesses when the line was sung too far off the tune to
    align. Every millisecond figure here is only as meaningful as the pitch
    agreement underneath it, and a line sung to a different melody will still
    produce a best shift.
    """
    if cents > TIMING_NEEDS_CENTS:
        return None
    # NEGATED on purpose. offset() returns the shift that ALIGNS the take, which
    # is the negation of how late the take is; see its docstring for the
    # measured table. Positive here means behind the words, which is the only
    # direction a person can act on.
    r = offset(sa, sb, max_shift=LINE_SHIFT)
    return round(-r * FRAME_MS, 1)


def timing(lines: list) -> dict | None:
    """Whether you drag or rush, and the line you were furthest out on.

    Reported as a signed median rather than a mean: one line where the search
    hit its stop should not move the verdict, and a singer who is late on half
    the lines and early on the other half is not "on time", which is what a
    mean would call them. `spread_ms` is what says which of those you are.

    Deliberately NOT folded into the pitch headline. Being flat and being late
    are different mistakes with different fixes, and the whole reason this
    exists is that they were being reported as one number.
    """
    vals = [l.late_ms for l in lines if l.late_ms is not None]
    if len(vals) < 3:
        return None
    a = np.array(vals, dtype=float)
    worst = max((l for l in lines if l.late_ms is not None),
                key=lambda l: abs(l.late_ms))
    return {
        "median_ms": round(float(np.median(a)), 1),
        "spread_ms": round(float(np.percentile(a, 90) - np.percentile(a, 10)), 1),
        "share_tight": round(float(np.mean(np.abs(a) <= FRAME_MS * 2)), 3),
        "lines_timed": len(vals),
        "worst": {"text": worst.text, "late_ms": worst.late_ms,
                  "start": worst.start},
    }


def timing_headline(t: dict | None) -> str | None:
    """One sentence, in the register the pitch headline already uses.

    A millisecond figure means nothing to anybody singing. Half a frame either
    way is not a thing a person can act on; being consistently a third of a
    second behind is.
    """
    if not t:
        return None
    m, spread = t["median_ms"], t["spread_ms"]
    if abs(m) <= FRAME_MS * 2 and spread <= FRAME_MS * 6:
        return "Your timing is good, you sit right on the words"
    if abs(m) <= FRAME_MS * 2:
        return "On the beat on average, but you wander either side of it"
    word = "behind" if m > 0 else "ahead of"
    return f"You sing about {abs(m):.0f} ms {word} the words"


def coverage(lines: list) -> tuple:
    """(lines you sang, lines there were to sing).

    Wren, 2026-09-04, on the safeguard built for the median's blind spot:
    "the fix you shipped inherited the identical blind spot one level down --
    unsung is invisible to both the overall score and the thing that exists
    specifically to catch what the overall score misses."

    He is right and it is the same failure this record already has a name for:
    I fixed the call site and left the sibling. `best_and_worst` compares only
    lines that HAVE a score, so skipping nineteen lines out of twenty made them
    vanish from the comparison rather than count against it. Sing two lines
    well and it congratulated you on both.
    """
    there = [l for l in lines if l.ref_frames >= MIN_LINE_FRAMES]
    sang = [l for l in there if l.median_cents is not None]
    return len(sang), len(there)


def headline(v: "Verdict", lines: list) -> str:
    """The sentence somebody reads, with the median's blind spots patched.

    TWO OF THEM, and I only saw the first on my own.

    **The median cannot see one bad line.** The PRD listed that as a worry and
    I did not act on it. Then a take that was perfect on twenty lines and 80
    cents flat on ONE reported "0 cents dead on", because one line in twenty
    cannot move a median and should not be able to. A person who was flat for
    four seconds has not sung it dead on, and a headline saying so is wrong in
    the worst available direction: it agrees with you.

    **And silence is free.** Wren found this within an hour of the first one
    being fixed: sing ten seconds of a two hundred second song accurately, stop,
    and it still says dead on, because the comparison only ever looks at frames
    where both of you are sounding. Worse, the fix above could not even reach
    its own check, since `best_and_worst` compares only lines that HAVE a score
    and nineteen skipped lines simply vanished from the comparison. The
    safeguard built for the first blind spot had inherited it.

    Neither is a threshold problem. The number is not wrong; it is about a
    fraction of the song and used to say so nowhere.
    """
    base = v.sentence
    sang, there = coverage(lines)

    if there >= 4 and sang < there:
        share = sang / there
        if share < 0.6:
            return (f"{base}. Only counting the {sang} of {there} lines you "
                    f"actually sang")
        if share < 0.9:
            return f"{base}, over the {sang} of {there} lines you sang"

    _, worst = best_and_worst(lines)
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
