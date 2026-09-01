"""How hard is the dialogue in this programme to hear, as a number.

The complaint this project comes from is specific: 78% of people say background
music makes dialogue hard to follow, and about half now read television rather
than listen to it. The thing under that complaint is one quantity -- how far the
speech sits above everything else -- and nobody outside a mix suite is ever
shown it.

## The measurement, and why it is not just a level difference

You cannot measure the speech alone in a finished mix; the music is under it.
What you *can* measure is two things:

    S = loudness of the blocks where somebody is talking   (speech + bed)
    B = loudness of the blocks where nobody is             (bed alone)

Power adds, so S = P_speech + P_bed and B = P_bed. Which means the speech alone
is the difference of the two powers, and the real speech-to-background ratio is

    SBR = 10*log10( 10^((S-B)/10) - 1 )

That is exact, not an approximation, whenever the bed is at a similar level
during speech as between it. Where that assumption breaks it breaks in the
right direction: a mix that ducks its music under dialogue measures as clearer,
because it is, and one that swells measures as worse, because it is.

Where it cannot work at all is when S <= B -- the talking is quieter than the
silence around it -- and then there is no ratio to report, only that fact.
`sbr_lu` is None there rather than a large negative number, because a number
would invite arithmetic on something that is not a measurement.

## What the numbers mean

    sbr_lu      speech above background. Broadcast practice puts intelligible
                dialogue somewhere above +4 LU for most listeners; the AES work
                on this recommends the difference not exceed about 5 LU in the
                other direction. Under ~+4 is where subtitles start going on.
    swing_lu    how far the loudest non-speech moments sit ABOVE the dialogue.
                This is the "turn it up for the talking, down for the
                explosions" number, and it is what makes people give up.
    spread_lu   how much the dialogue level itself moves across the programme.
                A big spread means quiet lines vanish even when the average is
                fine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import decode, vad
from .loudness import (ABSOLUTE_GATE, BlockLoudness, STEP_S, block_lufs,
                       gated_lufs, mean_lufs)

# A block counts as speech only if EVERY window in it is speech, and as
# background only if no window within GUARD_BLOCKS of it is. Everything between
# is dropped. The margin is the point, and it was measured rather than guessed:
# at 0.75 the classification leaked bed-only audio into the speech side and
# speech tails into the background side, biasing S-B down by about 0.8 dB. That
# is invisible at a wide ratio and enormous at a narrow one, because the
# inversion in `sbr_from` amplifies error by 10^(d/10)/(10^(d/10)-1) -- a factor
# of 4 by the time the difference is down to 1 dB.
SPEECH_BLOCK = 1.0

# How far either side of any speech is treated as neither. Swept against both
# fixture corpora on 2026-09-01 and it wants to be ZERO: the dilation was
# intended to keep speech tails out of the background, and it does, but it also
# ate almost every gap in continuous speech -- a human reading left 36 usable
# background blocks in 1197, so the "local" window had to expand to the whole
# file and stopped being local. Mean error over both corpora: 0.91 LU at 0,
# 1.44 at 3.
GUARD_BLOCKS = 0

# Below this raw difference between the speech and background levels there is
# not enough separation to invert: a tenth of a dB of classification error moves
# the answer by whole LU. The report says so instead of printing a number.
MIN_SEPARATION_DB = 1.0

# Where the verdict lines fall. These are judgement, not measurement, and they
# are here in one place so they can be argued with.
CLEAR_LU = 10.0
OK_LU = 4.0
BIG_SWING_LU = 12.0
BIG_SPREAD_LU = 8.0


@dataclass
class Moment:
    at: float                 # seconds
    sbr_lu: float | None
    speech_lufs: float
    background_lufs: float

    def clock(self) -> str:
        m, s = divmod(int(self.at), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class Report:
    path: str
    duration: float
    programme_lufs: float
    speech_lufs: float
    background_lufs: float
    sbr_lu: float | None
    sbr_range: tuple[float, float] | None
    swing_lu: float | None
    spread_lu: float | None
    speech_fraction: float
    worst: list[Moment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Only populated with keep_blocks=True. The verification harness needs the
    # exact block classification so it can ask the isolated stems the same
    # question over the same blocks; nothing in the product uses it.
    speech_mask: np.ndarray | None = None
    bg_mask: np.ndarray | None = None

    def verdict(self) -> str:
        if self.sbr_lu is None:
            return "the talking is quieter than what is under it"
        if self.sbr_lu < OK_LU:
            return "hard to hear"
        if self.sbr_lu < CLEAR_LU:
            return "workable, with effort"
        return "clear"

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("speech_mask", "bg_mask")}
        d["verdict"] = self.verdict()
        return d


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def sbr_from(speech_lufs: float, background_lufs: float) -> float | None:
    """Recover speech-alone above background from the two measurable levels.

    Returns None when the two levels are too close to tell apart. That is not
    caution for its own sake: `sensitivity()` below is the derivative, and at a
    1 dB separation one dB of error in either input becomes four LU in the
    answer. A number that unstable is worse than no number, because it looks
    exactly like the stable ones.
    """
    if not (math.isfinite(speech_lufs) and math.isfinite(background_lufs)):
        return None
    d = speech_lufs - background_lufs
    if d < MIN_SEPARATION_DB:
        return None
    ratio = 10.0 ** (d / 10.0) - 1.0
    if ratio <= 0:
        return None
    return 10.0 * math.log10(ratio)


def sensitivity(speech_lufs: float, background_lufs: float) -> float | None:
    """How many LU the answer moves per dB of error in the inputs.

    d(SBR)/d(S-B) = 10^(d/10) / (10^(d/10) - 1). It is 1.0 when speech towers
    over the bed and unbounded as they converge, which is the whole reason the
    hard cases here are the quiet ones.
    """
    if not (math.isfinite(speech_lufs) and math.isfinite(background_lufs)):
        return None
    d = speech_lufs - background_lufs
    if d <= 0:
        return None
    r = 10.0 ** (d / 10.0)
    return float(r / (r - 1.0)) if r > 1.0 else None


def analyse(path: str | Path, window_s: float = 3.0,
            worst_n: int = 5, keep_blocks: bool = False) -> Report:
    media = decode.probe(path)

    windows = vad.speech_windows(media)

    acc = BlockLoudness(media.channels)
    for chunk in decode.stream(media):
        acc.feed(chunk)
    z = acc.blocks()
    if z.size == 0:
        raise decode.DecodeError("too short to measure (under 400 ms of audio)")

    frac = vad.speech_fraction_per_block(windows, len(z))
    speech_mask = frac >= SPEECH_BLOCK
    # Dilate "there was speech near here" by the guard, so the background is
    # measured away from onsets, tails and any reverb the voice left behind.
    near_speech = frac > 0.0
    for shift in range(1, GUARD_BLOCKS + 1):
        near_speech |= np.roll(frac > 0.0, shift)
        near_speech |= np.roll(frac > 0.0, -shift)
    bg_mask = ~near_speech

    programme = gated_lufs(z)
    speech = gated_lufs(z, speech_mask)
    background = gated_lufs(z, bg_mask)

    # The headline ratio is the MEDIAN of a local track, not one whole-programme
    # subtraction. Measured on fixtures/real_gap+08.wav, 2026-09-01: taking the
    # background from every gap in the file put it 3.24 dB below the bed that was
    # actually under the speech, and the reported ratio was 3.43 LU too generous
    # -- the whole error.
    #
    # The cause is selection, not arithmetic. A gap only counts as background if
    # the detector found no speech near it, and the detector's behaviour depends
    # on the background, so the gaps that survive are not a fair sample of the
    # bed. In a scored drama the bias runs the other way: the music rises in the
    # gaps between lines, which is what scoring IS, and the ratio then reads
    # worse than it sounds.
    #
    # Comparing each moment of speech with the gaps NEAR it removes most of that,
    # because whatever the bed is doing it is usually doing it slowly.
    track = _local_track(z, speech_mask, bg_mask)
    sbr = _median([m.sbr_lu for m in track])

    notes: list[str] = []
    if not speech_mask.any():
        notes.append("no speech found; every number below is about the whole mix")
    if not bg_mask.any():
        notes.append("nobody stops talking, so there is no background to compare "
                     "against and the ratio cannot be measured")

    # The spread of the local track IS the uncertainty, and it is honest in a way
    # a fixed tolerance is not: a programme whose bed is steady gives a tight
    # band, one that swells gives a wide one, and the reader can see which.
    finite = sorted(m.sbr_lu for m in track if m.sbr_lu is not None)
    sbr_range = ((float(np.percentile(finite, 25)), float(np.percentile(finite, 75)))
                 if len(finite) >= 8 else None)
    if track and len(finite) < len(track) * 0.5:
        notes.append(f"for {len(track) - len(finite)} of {len(track)} moments the "
                     "voice was not far enough above the bed to put a number on")

    lufs = block_lufs(z)

    # swing: how far the loud non-speech moments sit above the dialogue
    swing = None
    if bg_mask.any() and math.isfinite(speech):
        loud_bg = float(np.percentile(lufs[bg_mask][np.isfinite(lufs[bg_mask])], 95))
        swing = loud_bg - speech

    # spread: how much the dialogue level itself moves
    spread = None
    if speech_mask.sum() >= 10:
        sl = lufs[speech_mask]
        sl = sl[np.isfinite(sl)]
        if sl.size >= 10:
            spread = float(np.percentile(sl, 90) - np.percentile(sl, 10))

    worst = _pick_worst(track, window_s, worst_n)

    return Report(
        path=str(media.path),
        duration=media.duration,
        programme_lufs=_finite(programme) or float("nan"),
        speech_lufs=_finite(speech) or float("nan"),
        background_lufs=_finite(background) or float("nan"),
        sbr_lu=sbr,
        sbr_range=sbr_range,
        swing_lu=swing,
        spread_lu=spread,
        speech_fraction=float(speech_mask.mean()),
        worst=worst,
        notes=notes,
        speech_mask=speech_mask if keep_blocks else None,
        bg_mask=bg_mask if keep_blocks else None,
    )


def _median(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None]
    return float(np.median(finite)) if finite else None


# How far to look for gaps when estimating the bed under a line of dialogue.
# Small enough that a bed which swells does not average away; expanded on demand
# when a passage has no gaps nearby at all.
NEAR_S = 8.0
NEAR_MAX_S = 90.0
MIN_BG_BLOCKS = 3           # half a second of gap; below this the estimate is noise
SPEECH_HALF_S = 1.5         # how much speech to pool around a moment


def _local_track(z: np.ndarray, speech_mask: np.ndarray, bg_mask: np.ndarray,
                 step: int = 5) -> list[Moment]:
    """The speech-to-background ratio as it moves through the programme.

    One entry every `step` blocks of speech (half a second), each comparing the
    speech around that instant with the nearest gaps. Everything else in the
    report is derived from this: the headline is its median, the uncertainty is
    its quartiles, and the worst moments are its floor.

    Deliberately not vectorised. Each entry runs the full BS.1770 gate over its
    own two subsets, and a faster version that skipped the gate would be
    measuring a different quantity from the headline number.
    """
    n = len(z)
    bg_idx = np.flatnonzero(bg_mask)
    sp_idx = np.flatnonzero(speech_mask)
    if bg_idx.size == 0 or sp_idx.size == 0:
        return []

    sp_half = int(SPEECH_HALF_S / STEP_S)
    out: list[Moment] = []
    for i in sp_idx[::step]:
        near = np.empty(0, dtype=int)
        reach = NEAR_S
        while near.size < MIN_BG_BLOCKS and reach <= NEAR_MAX_S:
            near = bg_idx[np.abs(bg_idx - i) <= reach / STEP_S]
            reach *= 2
        if near.size == 0:
            continue

        bg_sel = np.zeros(n, dtype=bool)
        bg_sel[near] = True
        sp_sel = np.zeros(n, dtype=bool)
        sp_sel[max(0, i - sp_half):min(n, i + sp_half + 1)] = True
        sp_sel &= speech_mask

        b = mean_lufs(z, bg_sel)
        s = mean_lufs(z, sp_sel)
        if not (math.isfinite(b) and math.isfinite(s)):
            continue
        # With no background to speak of, the ratio runs away: fixtures/dry.wav
        # is speech over digital silence and reported +709 LU before this
        # existed. There is nothing under the voice, which is a fact worth
        # saying, and it is not a number.
        if b <= ABSOLUTE_GATE:
            out.append(Moment(at=i * STEP_S, sbr_lu=None, speech_lufs=s,
                              background_lufs=b))
            continue
        out.append(Moment(at=i * STEP_S, sbr_lu=sbr_from(s, b),
                          speech_lufs=s, background_lufs=b))
    return out


def _pick_worst(track: list[Moment], window_s: float, worst_n: int) -> list[Moment]:
    """The places a person would actually reach for the remote.

    A None ratio -- the voice not measurably above the bed at all -- sorts worse
    than any number, because it is worse.
    """
    ranked = sorted(track, key=lambda m: (m.sbr_lu is not None,
                                          m.sbr_lu if m.sbr_lu is not None else 0.0))
    kept: list[Moment] = []
    for m in ranked:
        # Spaced out, or all five land inside one bad scene and the report
        # describes a minute of a film as though it were the film.
        if all(abs(m.at - k.at) > window_s * 2 for k in kept):
            kept.append(m)
        if len(kept) >= worst_n:
            break
    return kept
