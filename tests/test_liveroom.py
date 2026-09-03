"""The room measurement, the search, and the three refusals.

Everything here runs against a real recorded voice rather than a synthetic
burst signal, because four separate times this week a measurement was fooled by
test material that was not like the world: a bed 92% below 300 Hz, mono
duplicated into stereo, TTS standing in for a human, and a randomly chosen song
that turned out to be the most famously unintelligible vocal ever recorded.

`fixtures/dry.wav` is a real dry voice. Pink noise is added to it because a real
separated stem always has a noise floor and never has digital silence, and an
estimator tested only against true silence is tested against a signal that does
not occur.

EVERY NOISE SOURCE HERE IS SEEDED, and that is not tidiness. `anoisesrc`
defaults to `seed=-1`, which is a new stream of noise on every run, so the
fixture these tests measure was different every time. It showed up as
`test_the_search_recovers_a_known_room` failing under mutations that could not
possibly have touched the search -- once under a change to the level step, once
under a change to the bleed check -- which made every mutation result ambiguous
until I looked at why. A test whose input is randomised is not a test that
went red, it is a coin.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from earshot import liveroom as lr

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DRY = FIXTURES / "dry.wav"

needs_audio = pytest.mark.skipif(
    not DRY.is_file(), reason="fixtures/dry.wav not on this machine")


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    """A dry voice with a realistic noise floor, all 26 s of it.

    NOT trimmed to 16 s, and the reason is a finding rather than a preference:
    measured, a 16 s stretch of this voice yields TWO usable gaps and 26 s
    yields four, against a MIN_GAPS of three. Trimming for speed would have
    made half these tests exercise the not-enough-gaps refusal instead of the
    thing they are named after. It is also the plan's most fragile assumption
    failing exactly where it said it would, and loudly.
    """
    d = tmp_path_factory.mktemp("liveroom")
    out = d / "dry.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(DRY),
         "-f", "lavfi", "-i", "anoisesrc=c=pink:a=0.0018:r=44100:seed=101",
         "-filter_complex",
         "[0][1]amix=inputs=2:duration=first:weights=1 1,volume=2",
         "-ar", "44100", "-ac", "1", str(out)],
        check=True, timeout=600)
    return out


# --- 1. the known-answer control --------------------------------------------

@needs_audio
def test_a_dry_voice_measures_almost_no_room(dry):
    """The one that has to hold. Three earlier versions of this estimator
    reported 2.4, 1.6 and 2.1 seconds of reverb on a dry file, and every one of
    them would have looked like a working measurement on real songs."""
    t, gaps = lr.tail_seconds(lr._mono(dry))
    assert t is not None and gaps >= lr.MIN_GAPS
    assert t < 0.15, f"invented {t:.3f}s of room in a dry recording"


@needs_audio
def test_longer_rooms_measure_longer_until_the_estimator_gives_out(dry, tmp_path):
    """Monotone across the range it is valid over, and NOT beyond it.

    The PRD for this build asked for "0.80 above dry and below 2.00". That is
    false and the test says so rather than being written to pass: a 2.2 s room
    measures 0.59 and a 3.0 s room measures 0.40, below where 0.6 s lands,
    because past a point the tail no longer fits inside a gap and the estimator
    stops measuring the room at all. That turning point is real, it is why
    `search()` truncates each tilt at its own maximum, and a test that asserted
    the plan's version would have had to be broken to pass.
    """
    got = {}
    for rt in (0.4, 0.9, 1.6, 3.0):
        y = lr.convolve(dry, lr.impulse(rt, "neutral"), tmp_path / f"w{rt}.wav")
        got[rt], _ = lr.tail_seconds(lr._mono(y))

    dry_t, _ = lr.tail_seconds(lr._mono(dry))
    assert dry_t < got[0.4] < got[0.9] < got[1.6], f"not monotone: {got}"
    assert got[3.0] < got[1.6], (
        "the estimator no longer turns over at the top, so the monotone-prefix "
        "truncation in search() is now guarding nothing and should be re-derived")


# --- 2. recovery -------------------------------------------------------------

@needs_audio
@pytest.mark.parametrize("true_rt", [0.4, 0.9, 1.6])
def test_the_search_recovers_a_known_room(dry, tmp_path, true_rt):
    """Build a fake record with a known room, then find it from scratch."""
    fake = lr.convolve(dry, lr.impulse(true_rt, "neutral"), tmp_path / "rec.wav")
    record = lr.profile(fake)
    win, _ = lr.search(dry, record, tmp_path / "s")
    assert win.rt60 in lr.RT60S, f"chose no room for a reverberant reference: {win}"
    off = abs(lr.RT60S.index(win.rt60) - lr.RT60S.index(true_rt))
    assert off <= 1, f"true {true_rt}, chose {win.rt60}, {off} steps off"


# --- 3. the negative control -------------------------------------------------

@needs_audio
def test_a_dry_reference_does_not_get_a_room_invented_for_it(dry, tmp_path):
    """The test that would have caught every instrument in validate-the-validator.

    Something is always nearest, so without a candidate for doing nothing the
    search hands a dry reference the shortest room in the bank and reports it
    as a match. It did exactly that until `none` was allowed to win ties.
    """
    win, _ = lr.search(dry, lr.profile(dry), tmp_path / "s")
    assert win.rt60 == 0.0, f"invented a {win.rt60}s room for a dry reference"


@needs_audio
def test_a_long_broken_room_is_never_chosen_over_a_short_honest_one(dry, tmp_path):
    """The monotone prefix, tested by its consequence rather than its shape.

    Past the turning point the estimator stops measuring the room, so a 3.0 s
    cathedral and a ~0.4 s room report almost the same number. Ask for that
    number and a nearest-match search has no way to tell them apart: it is
    exactly as happy to answer "three seconds".

    An earlier version of this test asserted that the winner was not the last
    entry in the bank, which passed whether or not the truncation existed --
    because for a dry reference the winner is `none` either way. Removing the
    truncation left the whole suite green. This one goes red.
    """
    ref = lr.convolve(dry, lr.impulse(0.9, "neutral"), tmp_path / "ref.wav")
    _, table = lr.search(dry, lr.profile(ref), tmp_path / "s")

    longest = [c for c in table
               if c["rt60"] == lr.RT60S[-1] and c["tilt"] == "neutral"][0]
    assert longest["measured"] is not None, "the longest room measured nothing"

    # Ask for precisely what the broken end of the bank reports.
    target = lr.Profile(tail=longest["measured"], gaps=9,
                        lufs=-16.0, crest=15.0, eq=[0.0] * len(lr.BANDS))
    win, _ = lr.search(dry, target, tmp_path / "s2")
    assert win.rt60 != lr.RT60S[-1], (
        f"asked for {longest['measured']:.3f}s and was handed a "
        f"{lr.RT60S[-1]}s room, which is a cathedral")


# --- 4. the refusals ---------------------------------------------------------

@needs_audio
def test_it_refuses_when_the_backing_is_in_the_microphone(dry, tmp_path):
    """Constructed: mix the band into the take at -12 dB, as an open speaker
    would. A clean take must pass the same check."""
    band = tmp_path / "band.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "anoisesrc=c=brown:a=0.5:r=44100:d=16:seed=202",
         "-ar", "44100", "-ac", "1", str(band)], check=True, timeout=300)

    bled = tmp_path / "bled.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(dry), "-i", str(band),
         "-filter_complex",
         "[1:a]volume=-12dB[q];[0:a][q]amix=inputs=2:duration=shortest:normalize=0",
         "-ar", "44100", "-ac", "1", str(bled)], check=True, timeout=300)

    record = lr.convolve(dry, lr.impulse(0.9, "neutral"), tmp_path / "rec.wav")
    clean_c = lr.bleed(dry, band)
    bled_c = lr.bleed(bled, band)
    assert clean_c < lr.BLEED_CORR, f"headphoned take flagged as bleed ({clean_c:.2f})"
    assert bled_c >= lr.BLEED_CORR, f"open speaker not caught ({bled_c:.2f})"

    d = lr.assess(bled, record, band, tmp_path / "as")
    assert not d.ok and "headphones" in d.why


@needs_audio
def test_it_refuses_a_take_with_no_pauses_in_it(dry, tmp_path):
    """A continuous tone has no phrase ends, so there is nothing to measure and
    the honest answer is to say so rather than return a number from one gap."""
    solid = tmp_path / "solid.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=220:duration=16", "-ar", "44100", "-ac", "1",
         str(solid)], check=True, timeout=300)
    band = tmp_path / "band.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "anoisesrc=c=brown:a=0.5:r=44100:d=16:seed=202", "-ar", "44100",
         "-ac", "1", str(band)], check=True, timeout=300)

    record = lr.convolve(dry, lr.impulse(0.9, "neutral"), tmp_path / "rec.wav")
    d = lr.assess(solid, record, band, tmp_path / "as")
    assert not d.ok and "pauses" in d.why


@needs_audio
def test_it_refuses_when_your_room_is_bigger_than_the_records(dry, tmp_path):
    """Constructed the only honest way round: a very wet take against a dry
    record. Nothing can be ADDED to make a big room smaller, and the answer is
    a sentence about where to stand rather than a file that sounds wrong."""
    wet_take = lr.convolve(dry, lr.impulse(2.2, "neutral"), tmp_path / "wet.wav")
    band = tmp_path / "band.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "anoisesrc=c=brown:a=0.5:r=44100:d=16:seed=202", "-ar", "44100",
         "-ac", "1", str(band)], check=True, timeout=300)

    record = lr.convolve(dry, lr.impulse(0.2, "neutral"), tmp_path / "rec.wav")
    d = lr.assess(wet_take, record, band, tmp_path / "as")
    assert not d.ok, "accepted a take wetter than the record"
    assert "bigger than" in d.why


# --- 5. the level trapdoor ---------------------------------------------------

@needs_audio
def test_the_placed_take_lands_at_the_reference_level(dry, tmp_path):
    """Guards the 11.3 dB trapdoor: on 2026-08-30 an aecho and a low shelf took
    that much out of a voice at exit code 0, because anything that adds
    reflections scales the direct path and nothing says so."""
    ref = lr.convolve(dry, lr.impulse(0.9, "neutral"), tmp_path / "ref.wav")
    band = tmp_path / "band.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "anoisesrc=c=brown:a=0.3:r=44100:d=16:seed=303", "-ar", "44100",
         "-ac", "2", str(band)], check=True, timeout=300)

    room = lr.Candidate(0.9, "neutral", 0.7, 5, 0.0)
    steps = lr.place(dry, ref, band, room, tmp_path / "out", colour=False)
    assert abs(steps["placed_lufs"] - steps["target_lufs"]) <= 1.5, steps
    assert abs(steps["room_took_db"]) < 6.0, (
        f"the room quietly took {steps['room_took_db']} dB")
    assert (tmp_path / "out" / "placed.mp3").is_file()
    assert (tmp_path / "out" / "unplaced.mp3").is_file()


@needs_audio
def test_placed_and_unplaced_are_not_the_same_file(dry, tmp_path):
    """The A/B is the product. Two identical files would be a comparison of
    nothing, and both are the right length and size either way."""
    ref = lr.convolve(dry, lr.impulse(0.9, "neutral"), tmp_path / "ref.wav")
    band = tmp_path / "band.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "anoisesrc=c=brown:a=0.3:r=44100:d=16:seed=303", "-ar", "44100",
         "-ac", "2", str(band)], check=True, timeout=300)
    room = lr.Candidate(1.6, "neutral", 1.0, 5, 0.0)
    lr.place(dry, ref, band, room, tmp_path / "out", colour=False)

    a = lr._mono(tmp_path / "out" / "placed.mp3")
    b = lr._mono(tmp_path / "out" / "unplaced.mp3")
    n = min(a.size, b.size)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr < 0.999, "placed and unplaced are the same audio"


# --- the arithmetic, which needs no audio -----------------------------------

def test_the_impulse_carries_the_direct_to_reverberant_ratio_it_claims():
    ir = lr.impulse(0.9, "neutral")
    direct = ir[0] ** 2
    tail = float(np.sum(ir[1:] ** 2))
    drr = 10 * np.log10(direct / tail)
    assert abs(drr - lr.DRR_DB) < 1.5, f"DRR is {drr:.1f} dB, not {lr.DRR_DB}"


def test_the_bank_is_reproducible_without_shipping_binaries():
    assert np.array_equal(lr.impulse(0.9, "dark"), lr.impulse(0.9, "dark"))
    assert not np.array_equal(lr.impulse(0.9, "dark"), lr.impulse(0.9, "bright"))


def test_doing_nothing_wins_a_tie():
    """Ordering, not audio: the null action must beat an equally good room."""
    assert lr.tail_tie(0.8) > 0.02
    assert lr.tail_tie(0.1) == 0.02


# --- the decision logic, without forty convolutions -------------------------
#
# Both rules below survived a mutation of the audio-level tests: the monotone
# prefix could be deleted and the whole suite stayed green, and the null-action
# rule was masked because in the dry-reference control the untouched take also
# has the smallest colour distance. Tested here directly, where the inputs can
# be chosen to make each rule the only thing that decides.

def C(rt60, measured, eq=1.0, tilt="neutral"):
    return lr.Candidate(rt60, tilt, measured, 5, eq)


def test_the_monotone_prefix_drops_everything_past_the_turn():
    row = [C(0.2, 0.11), C(0.4, 0.27), C(0.9, 0.74), C(1.6, 1.01),
           C(2.2, 0.59), C(3.0, 0.40)]
    kept = [c.rt60 for c in lr.monotone_prefix(row)]
    assert kept == [0.2, 0.4, 0.9, 1.6]


def test_a_cathedral_is_not_offered_for_a_small_room():
    """Without the prefix, 3.0 and 0.4 report the same number and the search
    has no way to prefer the honest one."""
    row = [C(0.4, 0.40), C(1.6, 1.01), C(3.0, 0.40)]
    win = lr.choose(lr.monotone_prefix(row), target=0.40)
    assert win.rt60 == 0.4


def test_doing_nothing_wins_a_tie_it_did_not_strictly_win():
    """The null action is 0.01 WORSE on length and has a far worse colour
    match, and still wins, because a room you cannot justify is not added."""
    cands = [C(0.0, 0.081, eq=9.0, tilt="none"), C(0.2, 0.080, eq=0.1)]
    assert lr.choose(cands, target=0.080).rt60 == 0.0


def test_a_room_is_added_when_it_is_genuinely_better():
    """And the rule above must not swallow the real cases."""
    cands = [C(0.0, 0.07, eq=1.0, tilt="none"), C(0.9, 0.74, eq=1.0)]
    assert lr.choose(cands, target=0.75).rt60 == 0.9


def test_colour_breaks_a_genuine_tie():
    cands = [C(0.9, 0.740, eq=5.0, tilt="dark"), C(0.9, 0.741, eq=0.2, tilt="bright")]
    assert lr.choose(cands, target=0.74).tilt == "bright"
