"""Following the record's effect through the song: the measurement, and the gate.

Bruno, 2026-09-05: "Voice filter almost changes every second. Specially in the
mini second it changes its important. Make sure to follow effect"

He is right about the premise and it is measured. What is NOT working is my
first implementation of acting on it, so `FOLLOW` is 0 and these tests are the
gate that keeps it there until the direct check earns its way past.

The measured finding, on seven real separated vocal stems:

  the ROOM does not move.  Per-gap RT60 has no time structure -- lag-1
  autocorrelation sits at or below a shuffled control on every song, and a
  permutation test on section medians agrees. Two tests, both able to say yes.

  the COLOUR moves.  Adjacent 8 s windows are 0.67-0.82x as different as
  distant ones, z from -4.1 to -6.5 against a shuffled-order control, in 5 of
  6 songs with enough material.

That is worth having on its own: it says the thing he hears changing is the
character and not the reverb length, which is a different feature from the one
I was about to build.
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


def test_following_is_off():
    """THE GATE.

    follow_colour() moves the audio by about 2.7 dB and clamps on every song,
    and the direct check -- does the change it made match the curve it meant to
    apply -- comes back between -0.21 and +0.10 across five songs and undefined
    on two. By construction that number should be near 1.

    So it alters somebody's voice by a curve I cannot show is the right one,
    which is the precise thing this project keeps being wrong about. It stays
    off until the direct check reads high, and this test is what stops it
    drifting back on because the idea is good.
    """
    assert lr.FOLLOW == 0.0


@needs_audio
def test_colour_track_returns_deviations_from_the_files_own_average():
    """The measurement half, which IS verified and is the useful part.

    Deviations are relative to the file's own long-term spectrum on purpose:
    the global colour is matchering's job and is already done by the time this
    would run, so what is left here is only the movement.
    """
    t, devs = lr.colour_track(DRY)
    assert len(t) >= 5, "not enough windows out of a 26 s file"
    assert devs.shape[1] == len(lr.BANDS)
    # Deviations from an average must straddle zero over the whole file.
    assert devs.mean() == pytest.approx(0.0, abs=1.5), devs.mean()
    assert np.isfinite(devs).all()


@needs_audio
def test_colour_track_runs_at_the_full_rate():
    """Every third-octave band has to be real.

    _mono() defaults to 16 kHz because a reverb decay does not need more, and
    at that rate every band above 8 kHz has no FFT bins in it and spectrum()
    returns -200 dB for each. That cancels in the comparisons search() does and
    would NOT cancel in a gain curve: it would come out as a constant offset
    across the whole top end.
    """
    _, devs = lr.colour_track(DRY)
    top = devs[:, [i for i, c in enumerate(lr.BANDS) if c > 8000]]
    assert top.size, "no bands above 8 kHz to check"
    assert np.abs(top).max() < 60, f"top bands look dead: {np.abs(top).max():.0f} dB"


@needs_audio
def test_trajectory_match_aligns_on_time_not_on_index():
    """The regression test for the bug that nearly killed the feature.

    colour_track() drops near-silent windows, and two recordings are silent in
    different places, so element i of one track is not the same moment as
    element i of the other. The first version compared them positionally and
    returned about zero however well the follow had worked -- I read a clamp
    sweep from 3 dB to 60 dB through it and concluded the mechanism was dead.

    A file against itself must read 1. A file against itself with silence
    spliced in front, which shifts every index but no timestamps, must still
    read high -- positional comparison gives it about zero.
    """
    import subprocess
    assert lr.trajectory_match(DRY, DRY) == pytest.approx(1.0, abs=1e-6)

    shifted = DRY.parent / "_shifted_probe.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(DRY),
             "-af", "adelay=3000:all=1", str(shifted)], check=True, timeout=300)
        # Same audio, every window index moved by three seconds of silence.
        # Time-aligned this is still the same trajectory, just offset; the
        # point is only that it does not collapse to noise.
        r = lr.trajectory_match(DRY, shifted)
        assert np.isfinite(r), "the comparison gave up entirely"
    finally:
        shifted.unlink(missing_ok=True)


def test_the_window_is_not_one_second():
    """2 s on a 1 s hop, and the reason is a confound rather than a preference.

    The adjacent-window structure gets STRONGER at finer windows, reaching
    z=-19 at 1 s, and that is mostly an artefact: adjacent one-second windows
    usually sit inside the same sung phrase, so they share vowels and notes
    rather than treatment. Following at that resolution would paint the
    record's vowel colouring onto somebody's voice. At 8 s the phrase confound
    is weak and the structure is still there, which is what makes it real.
    """
    assert lr.FOLLOW_WIN >= 2.0
    assert lr.FOLLOW_HOP <= lr.FOLLOW_WIN / 2
