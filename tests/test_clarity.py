"""The clarity stage: does it do anything, and is it actually small?

Bruno, 2026-09-04: "can you put like a small clarity or whatever filter on top
of the song filter. Very small to smooth out the voice"

He said small twice. That is a specification, not a mood, and it is the only
half of this I can check without ears -- whether it sounds better is his call
and no test here pretends otherwise. So these come in pairs: one that it moved
the thing it claims to move, and one that it did not move it far.

The second of each pair is the one that can embarrass me. A clarity stage that
quietly turned into a 6 dB presence lift and a 4:1 compressor would pass every
"did it change anything" assertion ever written, and would sound like a podcast
plugin on somebody's singing.

`fixtures/dry.wav` is a real dry voice, for the reason the liveroom tests give
at length: four times this week a measurement was fooled by material that was
not like the world.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from earshot import liveroom as lr

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DRY = FIXTURES / "dry.wav"

needs_audio = pytest.mark.skipif(
    not DRY.is_file(), reason="fixtures/dry.wav not on this machine")


# --- the chain string, which needs no audio ---------------------------------

def test_zero_is_the_identity_chain():
    """`clarity=0` has to mean OFF, not 'a very small amount of'. The page
    offers no dial yet, so this is the escape hatch if he says take it out."""
    assert lr.clarity_chain(0) == "anull"
    assert lr.clarity_chain(0.0) == "anull"


def test_every_stage_scales_off_the_one_dial():
    """If he says more or less -- which is the likeliest next thing he says --
    it must be one number, not four edits. Same reason COVER and HERO_W became
    per-run overrides while he sat and watched takes."""
    small, big = lr.clarity_chain(1.0), lr.clarity_chain(2.0)
    assert "g=1.50" in small and "g=3.00" in big          # presence
    assert "g=-1.00" in small and "g=-2.00" in big        # mud
    assert "ratio=2.00" in small and "ratio=3.00" in big  # compression


def test_there_is_no_de_esser():
    """It was in the chain and it is not any more, because it was measured
    doing nothing: p99 of 5-9 kHz reads -3.07 dB on this voice with it and
    -3.07 without. Pinned so it does not drift back in on the strength of
    sounding like it belongs -- if it returns it comes with a number."""
    assert "deesser" not in lr.clarity_chain(1.0)
    assert "deesser" not in lr.clarity_chain(5.0)


# --- what it does to a real voice -------------------------------------------

@needs_audio
def test_it_lifts_presence_and_cuts_mud(tmp_path):
    """The half that says it works at all. Both thresholds come from measuring
    each stage in isolation (+1.25 presence, -0.8 mud) rather than from what I
    assumed they would be -- the first version of this test asserted the mud
    cut at a tenth of its real size and failed, because it was reading a
    mean-normalised curve that the highpass had shifted underneath it.

    THE THRESHOLD IS 1.3 AND THAT NUMBER IS THE TEST. Deleting the presence
    bell entirely still measures +0.79, because the compressor takes more out
    of the voice's body than out of its presence band, so this reading has a
    floor of about +0.8 with no lift in the chain at all. The first version
    asserted > 0.5 -- underneath the floor -- and passed happily when the stage
    it was named for was removed. Full chain reads +1.83, so 1.3 sits between
    them with room on both sides.
    """
    got = lr.apply_clarity(DRY, tmp_path / "out.wav")
    assert got["presence_rel_db"] > 1.3, got
    assert got["mud_db"] < -0.3, got


@needs_audio
def test_it_takes_the_rumble_out(tmp_path):
    """The biggest move in the chain, and the one I could not see. Reported
    now, so it is a number somebody could argue with."""
    got = lr.apply_clarity(DRY, tmp_path / "out.wav")
    assert got["sub_db"] < -2.0, f"the highpass is not doing its job: {got}"


@needs_audio
def test_it_is_small(tmp_path):
    """THE CONTROL, and the reason this file exists.

    He asked for very small. Without a bound, "clarity" drifts upward every
    time somebody thinks it could be a bit clearer, and the drift is invisible
    because each step sounds like an improvement on the last one.

    2 dB across the presence band is about the most that can honestly be called
    small; a mastering-grade presence lift is 4-6. The sub band is exempt and
    on purpose: there is nothing in a voice below 70 Hz, so taking 4 dB out of
    it is not a change to the voice at all.
    """
    got = lr.apply_clarity(DRY, tmp_path / "out.wav")
    assert got["presence_rel_db"] < 2.0, f"presence lift is not small: {got}"
    assert abs(got["mud_db"]) < 2.0, f"mud cut is not small: {got}"
    assert abs(got["took_db"]) < 3.0, f"it moved the level by {got['took_db']} dB"


@needs_audio
def test_it_smooths_rather_than_squashes(tmp_path):
    """"Smooth out the voice" is compression, and compression that goes too far
    is the single most audible way to make a voice sound processed. Crest
    factor is the measurable half: it should come down a little and not a lot."""
    got = lr.apply_clarity(DRY, tmp_path / "out.wav")
    assert got["crest_db"] < 0.0, f"nothing was evened out: {got}"
    assert got["crest_db"] > -6.0, f"it squashed the voice: {got}"


@needs_audio
def test_more_is_more(tmp_path):
    """The dial has to be monotonic, or 'a bit less' is not a thing he can ask
    for. Cheap to state and it would catch a scaling sign error."""
    a = lr.apply_clarity(DRY, tmp_path / "a.wav", amount=0.5)
    b = lr.apply_clarity(DRY, tmp_path / "b.wav", amount=2.0)
    assert b["presence_rel_db"] > a["presence_rel_db"], (a, b)
    assert b["mud_db"] < a["mud_db"], (a, b)


@needs_audio
def test_off_leaves_the_audio_alone(tmp_path):
    """`anull` through ffmpeg is still a decode and a re-encode, so this asserts
    the SIGNAL is unchanged rather than the bytes."""
    got = lr.apply_clarity(DRY, tmp_path / "off.wav", amount=0)
    assert abs(got["presence_rel_db"]) < 0.05, got
    assert abs(got["mud_db"]) < 0.05, got
    assert abs(got["took_db"]) < 0.05, got


@needs_audio
def test_band_change_reads_zero_against_a_file_and_itself():
    """The measurement's own control. band_change() is the instrument every
    assertion above leans on, and an instrument that returns a plausible number
    for two identical inputs would make all of them meaningless."""
    assert abs(lr.band_change(DRY, DRY, 2000, 5000)) < 1e-9


@needs_audio
def test_the_instrument_can_see_a_stage_the_normalised_one_hides(tmp_path):
    """Pins the reason raw_band_db() exists at all.

    The mud bell alone moves 180-400 Hz by about -0.8 dB. Read through a
    mean-normalised third-octave curve in the full chain it reads about -0.08,
    because the highpass takes 4.4 dB out of 40-90 Hz and drags the mean with
    it. I had already written "this stage is decoration" in my head when the
    two instruments turned out to disagree by a factor of ten.
    """
    import subprocess
    bell = tmp_path / "bell.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(DRY),
         "-af", "equalizer=f=250:t=q:w=1.2:g=-1.00", "-ar", "44100",
         "-ac", "2", str(bell)], check=True, timeout=300)
    assert lr.band_change(DRY, bell, 180, 400) < -0.4
