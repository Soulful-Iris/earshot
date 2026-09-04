"""Scoring a performance against the record it was sung along to.

Most of these run on pitch tracks built directly as arrays rather than on
audio, and that is deliberate rather than lazy. The comparison is where the
thinking is, and building the two sides as arrays is the only way to be certain
they are time-aligned: the first version of this measurement detuned the AUDIO
with `asetrate` and `atempo`, which shifts pitch and time together, so a
50-cents-flat performance came out at 240 cents and looked exactly like randomly
scrambled singing. That is a comparison measuring its own misalignment, and it
nearly killed the idea before it was built.

One test runs the real tracker on real separated audio, because everything above
would still pass if `track()` returned garbage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from earshot import unison as u

REAL = Path(__file__).resolve().parent.parent / "work" / "jobs"


def melody(n: int = 1200, seed: int = 5) -> u.Track:
    """A plausible sung line: notes with gaps, vibrato, and SLIDES between them.

    The slides are not decoration. The first version of this fixture held each
    note perfectly flat, and `test_latency_costs_something_if_it_is_not_corrected`
    failed on it: an eight frame delay cost 9 cents instead of the 40 the same
    delay costs on a real song. A shift inside a flat note changes nothing, so
    a fixture made of flat notes is maximally forgiving of exactly the error
    the offset search exists to fix.

    Real singing slides. I measured that on a real stem the same night: the
    spread of pitch INSIDE a single word has a median of 183 cents. A fixture
    without portamento is not like the world in the one dimension these tests
    are about.
    """
    rng = np.random.RandomState(seed)
    f0 = np.full(n, np.nan)
    voiced = np.zeros(n, dtype=bool)
    i = 0
    prev = 220.0
    while i < n - 40:
        hold = rng.randint(14, 34)
        semis = rng.choice([0, 2, 4, 5, 7, 9, 11, 12])
        hz = 220.0 * 2 ** (semis / 12)
        glide = min(rng.randint(4, 10), hold // 2)
        curve = np.concatenate([
            np.geomspace(prev, hz, glide),                  # into the note
            np.full(hold - glide, hz)])
        vib = 1 + 0.004 * np.sin(np.linspace(0, 8, hold))
        f0[i:i + hold] = curve * vib
        voiced[i:i + hold] = True
        prev = hz
        i += hold + rng.randint(3, 12)          # a gap between notes
    return u.Track(f0=f0, voiced=voiced)


def detune(t: u.Track, cents: float, lo: int = 0, hi: int | None = None) -> u.Track:
    f0 = t.f0.copy()
    hi = len(t) if hi is None else hi
    f0[lo:hi] = f0[lo:hi] * 2 ** (cents / 1200)
    return u.Track(f0=f0, voiced=t.voiced.copy())


# --- 1. the metric ----------------------------------------------------------

def test_identical_performances_score_zero():
    m = melody()
    assert u.score(m, m).median_cents == pytest.approx(0.0, abs=0.01)


def test_fifty_cents_flat_scores_fifty():
    m = melody()
    v = u.score(m, detune(m, -50))
    assert v.median_cents == pytest.approx(50.0, abs=2.0)
    assert v.flat_or_sharp == pytest.approx(-50.0, abs=2.0), "lost the direction"


def test_an_octave_down_is_right_and_scores_zero():
    """The decision that makes this usable by anybody whose range is not the
    singer's. Unfolded it is 1200 cents wrong, which is the answer every naive
    comparison gives a man singing a woman's part."""
    m = melody()
    down = detune(m, -1200)
    assert u.score(m, down).median_cents == pytest.approx(0.0, abs=0.01)

    both = m.voiced & down.voiced
    raw = np.median(np.abs(1200 * np.log2(down.f0[both] / m.f0[both])))
    assert raw == pytest.approx(1200.0, abs=1.0), "the fold is hiding nothing"


def test_a_fifth_is_wrong_even_folded():
    m = melody()
    assert u.score(m, detune(m, 700)).median_cents == pytest.approx(500.0, abs=2.0)


def test_the_fold_is_an_octave_wide_and_symmetric():
    assert u.fold(np.array([0.0]))[0] == pytest.approx(0.0)
    assert u.fold(np.array([1200.0]))[0] == pytest.approx(0.0)
    assert u.fold(np.array([-1200.0]))[0] == pytest.approx(0.0)
    assert u.fold(np.array([700.0]))[0] == pytest.approx(-500.0)
    assert abs(u.fold(np.array([600.0]))[0]) == pytest.approx(600.0)


# --- 2. discrimination, which is the one that stops the metric being a constant

def test_a_wrong_performance_scores_far_worse_than_a_flat_one():
    """Without this every other test here passes on a function that returns 0.

    A slightly flat performance and a scrambled one have to land in different
    worlds, or the number on the screen means nothing.
    """
    m = melody()
    flat = u.score(m, detune(m, -50)).median_cents

    rng = np.random.RandomState(11)
    blk = 40
    idx = rng.permutation(len(m) // blk)
    shuffled = u.Track(
        np.concatenate([m.f0[i * blk:(i + 1) * blk] for i in idx]),
        np.concatenate([m.voiced[i * blk:(i + 1) * blk] for i in idx]))
    wrong = u.score(m, shuffled).median_cents

    assert flat < 70, f"a 50 cent error scored {flat}"
    assert wrong > 150, f"a scrambled performance scored {wrong}"
    assert wrong > flat * 2


# --- 3. alignment -----------------------------------------------------------

@pytest.mark.parametrize("planted", [4, 8, 16])
def test_the_offset_search_finds_a_planted_delay(planted):
    """Browser latency between play() and MediaRecorder.start() lands in this
    range. Measured on a real song: a perfect take 186 ms late scores 40 cents
    instead of 0, and this search takes it back to 0."""
    m = melody()
    late = u.Track(
        np.concatenate([np.full(planted, np.nan), m.f0])[:len(m)],
        np.concatenate([np.zeros(planted, bool), m.voiced])[:len(m)])
    s = u.offset(m, late)
    assert abs(abs(s) - planted) <= 1, f"planted {planted}, found {s}"
    assert u.score(m, late, s).median_cents < 5


def test_latency_costs_something_if_it_is_not_corrected():
    """The reason the search exists. If this ever stops being true the search
    is guarding nothing and should be re-derived."""
    m = melody()
    planted = 8
    late = u.Track(
        np.concatenate([np.full(planted, np.nan), m.f0])[:len(m)],
        np.concatenate([np.zeros(planted, bool), m.voiced])[:len(m)])
    assert u.score(m, late, 0).median_cents > 10


# --- 4. the refusal ---------------------------------------------------------

def test_it_refuses_when_you_barely_sang():
    """A number from four seconds of overlap is not a verdict. This fires on
    somebody who hums, stops after a line, or records silence."""
    m = melody()
    quiet = u.Track(m.f0.copy(), m.voiced.copy())
    quiet.voiced[u.MIN_FRAMES // 2:] = False        # only a moment of overlap
    assert u.score(m, quiet) is None


def test_it_does_not_refuse_when_there_is_enough():
    m = melody()
    assert u.score(m, detune(m, -30)) is not None


# --- 5. per line, which is the part a person actually reads -----------------

class Cue:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def _lines(n=8, per=2.0):
    return [Cue(i * per, (i + 1) * per, f"line {i}") for i in range(n)]


def test_the_one_line_you_lost_is_the_one_it_names():
    """The test I care about most: it is the only one that checks the sentence
    somebody reads rather than a number in a dict."""
    m = melody(n=2000)
    lines = _lines()
    bad = lines[4]
    i0 = int(bad.start * u.SR / u.HOP)
    i1 = int(bad.end * u.SR / u.HOP)
    take = detune(m, -90, i0, i1)

    scored = u.per_line(m, take, 0, lines)
    best, worst = u.best_and_worst(scored)
    assert worst is not None and worst.text == bad.text, \
        [(l.text, l.median_cents) for l in scored]
    assert worst.median_cents == pytest.approx(90.0, abs=8.0)
    assert worst.flat_or_sharp < 0, "lost that it was FLAT rather than sharp"
    assert best.median_cents < 10


def test_a_line_with_almost_nothing_in_it_gets_no_score_rather_than_a_bad_one():
    m = melody(n=2000)
    lines = _lines()
    scored = u.per_line(m, m, 0, lines + [Cue(999.0, 999.4, "never sung")])
    assert scored[-1].median_cents is None


def test_the_headline_says_so_when_one_line_got_away():
    """The median cannot see it, and the median is the headline.

    Confirmed while building: a take perfect on twenty lines and 80 cents flat
    on ONE reported "0 cents dead on". One line in twenty cannot move a median
    and should not be able to, so the sentence has to carry what the number
    cannot.
    """
    m = melody(n=2000)
    lines = _lines()
    bad = lines[4]
    take = detune(m, -90, int(bad.start * u.SR / u.HOP), int(bad.end * u.SR / u.HOP))

    v = u.score(m, take)
    scored = u.per_line(m, take, 0, lines)
    assert v.median_cents < 25, "the median should NOT see a single bad line"
    assert "got away" in u.headline(v, scored)

    even = detune(m, -45)
    assert "got away" not in u.headline(u.score(m, even),
                                        u.per_line(m, even, 0, lines))


# --- the ribbon's data ------------------------------------------------------

def test_the_drawn_lines_are_folded_the_same_way_the_score_is():
    """Or the picture disagrees with the number printed under it: somebody
    singing an octave down would score zero and be drawn twelve semitones off
    the bottom of the window."""
    m = melody()
    c = u.contours(m, detune(m, -1200), 0, every=1)
    pairs = [(r, y) for r, y in zip(c["ref"], c["you"])
             if r is not None and y is not None]
    assert pairs, "nothing to draw"
    assert max(abs(r - y) for r, y in pairs) < 0.05


def test_the_drawn_lines_separate_when_you_are_flat():
    m = melody()
    c = u.contours(m, detune(m, -100), 0, every=1)
    gaps = [abs(r - y) for r, y in zip(c["ref"], c["you"])
            if r is not None and y is not None]
    assert np.median(gaps) == pytest.approx(1.0, abs=0.05), "a semitone is 1.0"


# --- the tracker itself, on real audio --------------------------------------

def _a_real_stem():
    for p in sorted(REAL.glob("*/out/vocals.mp3")):
        return p
    return None


@pytest.mark.skipif(_a_real_stem() is None, reason="no separated stem on this box")
def test_the_tracker_reads_a_real_separated_vocal(tmp_path):
    """Everything above would pass if track() returned noise. Measured on a
    204 second stem: 68% of frames voiced, B3 to E5, median E4."""
    import librosa
    t = u.track(_a_real_stem(), cache=tmp_path / "p.npz")
    assert len(t) > 100
    assert 0.25 < t.voiced.mean() < 0.95, f"{t.voiced.mean():.2f} voiced is not a vocal"
    good = t.f0[t.voiced]
    lo, hi = np.percentile(good, 5), np.percentile(good, 95)
    assert u.FMIN_HZ < lo < hi < u.FMAX_HZ
    assert hi / lo < 8, "a sung line does not span three octaves"

    cached = u.track(_a_real_stem(), cache=tmp_path / "p.npz")
    assert np.array_equal(np.nan_to_num(t.f0), np.nan_to_num(cached.f0))
    assert np.array_equal(t.voiced, cached.voiced)


# --- what Wren found: silence was free, and the safeguard could not see it ---

def _lines_with_ref(m, n=20, per=2.0):
    """Lines the SINGER actually sings, so 'you skipped it' is separable from
    'there was nothing here'."""
    return [Cue(i * per, (i + 1) * per, f"line {i}") for i in range(n)]


def silent_after(m: u.Track, keep_lines: int, per: float = 2.0) -> u.Track:
    """A take that sings the first few lines perfectly and then stops."""
    t = u.Track(m.f0.copy(), m.voiced.copy())
    cut = int(keep_lines * per * u.SR / u.HOP)
    t.voiced[cut:] = False
    return t


def test_skipping_most_of_the_song_is_not_dead_on():
    """Wren, 2026-09-04: sing ten seconds of a two hundred second song
    accurately, go silent, and score() returns "0 cents dead on".

    The number is not wrong. It is about a fraction of the song and used to
    say so nowhere. No threshold fixes that, so the sentence carries it.
    """
    m = melody(n=2600)
    lines = _lines_with_ref(m)
    # SIX lines, not two. Two is four seconds, which trips MIN_FRAMES and gets
    # refused outright -- correct, and not the hole. The hole is a take with
    # enough overlap to earn a verdict and nowhere near enough to deserve one,
    # which is exactly the ten-seconds-of-two-hundred Wren described.
    take = silent_after(m, 6)

    v = u.score(m, take)
    scored = u.per_line(m, take, 0, lines)
    assert v.median_cents < 5, "the metric itself is still blind, as expected"

    sang, there = u.coverage(scored)
    assert sang < there / 2, f"sang {sang} of {there}"
    said = u.headline(v, scored)
    assert "actually sang" in said, said
    assert str(sang) in said and str(there) in said


def test_the_safeguard_can_still_reach_its_own_check_when_lines_are_missing():
    """The bug underneath the bug. `best_and_worst` compares only lines that
    HAVE a score, so nineteen skipped lines vanished from the comparison
    instead of counting against it, and `headline` could not reach the check it
    exists for. Two sung lines were congratulated on each other."""
    m = melody(n=2600)
    lines = _lines_with_ref(m)
    take = silent_after(m, 6)
    scored = u.per_line(m, take, 0, lines)
    assert u.coverage(scored)[0] == 6
    assert u.coverage(scored)[1] > 10, "the reference should have plenty of lines"


def test_a_line_the_singer_never_sang_is_not_held_against_you():
    """The distinction that makes the above honest. A transcript can put words
    where the vocal stem is silent, and that is not somebody skipping a line."""
    m = melody(n=2600)
    lines = _lines_with_ref(m) + [Cue(9000.0, 9002.0, "not in the song")]
    scored = u.per_line(m, m, 0, lines)
    ghost = scored[-1]
    assert ghost.median_cents is None
    assert ghost.ref_frames == 0
    assert not ghost.unsung, "blamed somebody for a line the record does not have"
    sang, there = u.coverage(scored)
    assert there == len(lines) - 1, "counted a line nobody sings"


def test_singing_all_of_it_says_nothing_about_coverage():
    m = melody(n=2600)
    lines = _lines_with_ref(m)
    said = u.headline(u.score(m, detune(m, -45)),
                      u.per_line(m, detune(m, -45), 0, lines))
    assert "actually sang" not in said and "lines you sang" not in said
