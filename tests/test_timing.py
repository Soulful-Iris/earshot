"""Are you on the tempo with the words, and is that separable from your latency?

Bruno, 2026-09-05: "you have to make sure im on the correct tempo with the
lyrics"

Nothing could tell him. The global `offset()` search does not measure timing,
it deletes it: one shift for the whole take, subtracted so that headphone
latency is not scored as flat singing. Per-line drift was then folded into the
pitch number as though it were tuning.

The load-bearing test in here is `test_your_latency_is_not_your_fault`. Every
other assertion could pass on an implementation that simply reports the global
offset again for every line, and that implementation would tell a person with
slow headphones that they drag, forever, in a confident millisecond figure.

Synthetic tracks throughout, because these test arithmetic on a pitch contour
and a real voice adds nothing but minutes. The estimator itself was chosen
against four real stems: 127 of 129 planted per-line delays recovered within
one frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from earshot import unison as u


class Cue:
    """The shape `per_line` wants: video.Cue duck-typed."""
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


def melody(n_lines=8, line_frames=60, gap=20, seed=7):
    """A reference track: sung lines with real pitch movement, silence between.

    Movement matters. A track of held notes has no information about WHERE in
    the line you are, so any shift matches as well as any other and the test
    would be measuring nothing.
    """
    rng = np.random.default_rng(seed)
    f0, voiced, cues = [], [], []
    for i in range(n_lines):
        start = len(f0) * u.HOP / u.SR
        base = 220.0 * (2 ** (rng.integers(-4, 5) / 12))
        for k in range(line_frames):
            f0.append(base * (2 ** (np.sin(k / 7.0) * 2 / 12)))
            voiced.append(True)
        end = len(f0) * u.HOP / u.SR
        cues.append(Cue(f"line {i}", start, end))
        f0.extend([np.nan] * gap)
        voiced.extend([False] * gap)
    return (u.Track(np.array(f0, dtype=float), np.array(voiced, dtype=bool)),
            cues)


def delayed(track, frames):
    """The whole track arriving `frames` late. Stands in for system latency."""
    n = len(track)
    f0 = np.full(n, np.nan)
    vo = np.zeros(n, dtype=bool)
    if frames >= 0:
        f0[frames:] = track.f0[:n - frames]
        vo[frames:] = track.voiced[:n - frames]
    else:
        f0[:n + frames] = track.f0[-frames:]
        vo[:n + frames] = track.voiced[-frames:]
    return u.Track(f0, vo)


def shift_one_line(track, cue, frames):
    """Move a single line, leaving every other line where it was."""
    f0, vo = track.f0.copy(), track.voiced.copy()
    s = int(cue.start * u.SR / u.HOP)
    e = int(np.ceil(cue.end * u.SR / u.HOP))
    f0[s:e], vo[s:e] = np.nan, False
    ds, de = s + frames, e + frames
    f0[ds:de] = track.f0[s:e]
    vo[ds:de] = track.voiced[s:e]
    return u.Track(f0, vo)


# --- the one that matters ---------------------------------------------------

def test_your_latency_is_not_your_fault():
    """THE CONTROL.

    A take that is perfectly in time but arrives 8 frames (186 ms) late through
    the browser must read as ON TIME on every line. An implementation that
    reported the global offset per line would pass every other test in this
    file and would tell somebody with slow headphones that they drag.
    """
    ref, cues = melody()
    late = delayed(ref, 8)

    shift = u.offset(ref, late)
    # -8, not +8. offset() returns the shift that ALIGNS the take, which is the
    # negation of how late it is. Asserting +8 here is how I found that its
    # docstring had the sign backwards.
    assert shift == -8, f"the global search should see the latency, got {shift}"

    scored = u.per_line(ref, late, shift, cues)
    timed = [l.late_ms for l in scored if l.late_ms is not None]
    assert len(timed) >= 5, f"only {len(timed)} lines timed"
    assert max(abs(x) for x in timed) <= u.FRAME_MS, (
        f"latency leaked into the per-line timing: {timed}")


def test_a_late_line_is_found_and_the_others_are_not_blamed():
    """One line 5 frames (116 ms) late, the rest exactly on. Only that line
    should say so, or the number is measuring the take rather than the line."""
    ref, cues = melody()
    take = shift_one_line(ref, cues[3], 5)

    scored = u.per_line(ref, take, u.offset(ref, take), cues)
    late = scored[3].late_ms
    assert late is not None and late > 0, f"the late line reads {late}"
    assert abs(late - 5 * u.FRAME_MS) <= u.FRAME_MS, f"got {late} ms"

    others = [l.late_ms for i, l in enumerate(scored)
              if i != 3 and l.late_ms is not None]
    assert others and max(abs(x) for x in others) <= u.FRAME_MS, others


def test_early_reads_negative():
    """Sign convention, pinned. Positive is behind the words. Getting this
    backwards would tell someone who rushes to speed up."""
    ref, cues = melody()
    take = shift_one_line(ref, cues[2], -5)
    scored = u.per_line(ref, take, u.offset(ref, take), cues)
    assert scored[2].late_ms is not None and scored[2].late_ms < 0


# --- refusing rather than guessing ------------------------------------------

def test_a_line_sung_to_a_different_tune_gets_no_timing():
    """Above TIMING_NEEDS_CENTS the best alignment is arbitrary. A confident
    millisecond figure on top of that is a lie with a decimal point in it."""
    ref, cues = melody()
    f0 = ref.f0.copy()
    s = int(cues[1].start * u.SR / u.HOP)
    e = int(np.ceil(cues[1].end * u.SR / u.HOP))
    f0[s:e] = 440.0 * (2 ** (5 / 12))          # a different note, held
    take = u.Track(f0, ref.voiced.copy())

    scored = u.per_line(ref, take, 0, cues)
    assert scored[1].median_cents > u.TIMING_NEEDS_CENTS, scored[1].median_cents
    assert scored[1].late_ms is None


def test_an_unsung_line_gets_no_timing():
    ref, cues = melody()
    f0, vo = ref.f0.copy(), ref.voiced.copy()
    s = int(cues[4].start * u.SR / u.HOP)
    e = int(np.ceil(cues[4].end * u.SR / u.HOP))
    f0[s:e], vo[s:e] = np.nan, False

    scored = u.per_line(ref, u.Track(f0, vo), 0, cues)
    assert scored[4].unsung
    assert scored[4].late_ms is None


# --- the aggregate ----------------------------------------------------------

def test_timing_needs_more_than_two_lines():
    """Two numbers are not a tendency."""
    ref, cues = melody(n_lines=2)
    scored = u.per_line(ref, ref, 0, cues)
    assert u.timing(scored) is None


def test_a_dragging_singer_is_called_dragging():
    """Every line late by the same amount is a habit, not latency -- the global
    search cannot remove it here because the lines move relative to the gaps."""
    ref, cues = melody()
    take = ref
    for c in cues:
        take = shift_one_line(take, c, 6)

    scored = u.per_line(ref, take, 0, cues)
    t = u.timing(scored)
    assert t is not None and t["median_ms"] > u.FRAME_MS * 2, t
    assert "behind" in u.timing_headline(t)


def test_on_time_is_said_plainly():
    ref, cues = melody()
    t = u.timing(u.per_line(ref, ref, 0, cues))
    assert t is not None
    assert abs(t["median_ms"]) <= u.FRAME_MS
    assert t["share_tight"] == 1.0
    assert "timing is good" in u.timing_headline(t)


def test_the_headline_survives_having_nothing_to_say():
    assert u.timing_headline(None) is None


def test_worst_is_the_furthest_out_in_either_direction():
    """max of the ABSOLUTE value. Ranking on the signed number would always
    nominate the latest line and never the one that came in early."""
    ref, cues = melody()
    take = shift_one_line(ref, cues[5], -9)
    take = shift_one_line(take, cues[6], 3)

    t = u.timing(u.per_line(ref, take, 0, cues))
    assert t["worst"]["text"] == "line 5", t["worst"]
    assert t["worst"]["late_ms"] < 0
