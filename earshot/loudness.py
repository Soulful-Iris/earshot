"""ITU-R BS.1770-4 loudness, implemented so it can be asked about a *subset*.

Why not just call ffmpeg's `ebur128` filter: it reports loudness of a whole
programme, or a rolling window. The question this project asks is different --
"how loud is the programme *while somebody is talking*, compared with the parts
where nobody is" -- and that needs the gated loudness of an arbitrary set of
400 ms blocks, chosen after the fact.

So the blocks are computed here and the gating is a separate function that takes
a mask. `tests/test_loudness.py` checks this implementation against ffmpeg's on
whole files, which is the only claim both can answer, and against signals whose
loudness is known analytically.

Everything assumes 48 kHz, because the filter coefficients in the standard are
given at 48 kHz and `decode.py` always asks ffmpeg for 48 kHz.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 48000

# BS.1770-4 Tables 1 and 2, the two stages of "K" weighting at 48 kHz.
_STAGE1_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_STAGE1_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_STAGE2_B = np.array([1.0, -2.0, 1.0])
_STAGE2_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

BLOCK_S = 0.400          # gating block length
STEP_S = 0.100           # 75% overlap
ABSOLUTE_GATE = -70.0    # LUFS
RELATIVE_GATE = -10.0    # LU below the ungated level

BLOCK_N = int(round(BLOCK_S * SR))
STEP_N = int(round(STEP_S * SR))


class KWeighting:
    """The two-stage filter, carrying its state so audio can be streamed.

    One instance per channel. Feeding it in chunks gives bit-identical output to
    feeding it the whole signal, which `tests/test_loudness.py` asserts -- that
    is the property the whole streaming design rests on.
    """

    def __init__(self) -> None:
        self._zi1 = np.zeros(2)
        self._zi2 = np.zeros(2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self._zi1 = signal.lfilter(_STAGE1_B, _STAGE1_A, x, zi=self._zi1)
        y, self._zi2 = signal.lfilter(_STAGE2_B, _STAGE2_A, y, zi=self._zi2)
        return y


def channel_weights(n_channels: int) -> np.ndarray:
    """G_i from BS.1770-4. Mono and stereo are all this project ever sees.

    The standard weights surround channels at 1.41 and the LFE at 0. ffmpeg
    hands us mono or stereo (see decode.py), so anything else is a caller bug
    rather than something to guess at.
    """
    if n_channels in (1, 2):
        return np.ones(n_channels)
    raise ValueError(f"BS.1770 weights not defined here for {n_channels} channels")


class BlockLoudness:
    """Streaming accumulator: audio in, per-block mean square out.

    `blocks()` returns the weighted sum of channel mean squares for every
    400 ms block, one every 100 ms. Converting to LUFS is `-0.691 + 10log10(z)`
    and is deliberately NOT done here, because gating has to happen on the
    linear values.
    """

    def __init__(self, n_channels: int) -> None:
        self.n_channels = n_channels
        self._filters = [KWeighting() for _ in range(n_channels)]
        self._weights = channel_weights(n_channels)
        # A ring of the last BLOCK_N filtered squares per channel, kept as a
        # plain list of chunks and consumed as whole blocks become available.
        self._tail = np.zeros((0, n_channels))
        self._z: list[float] = []
        self.n_samples = 0

    def feed(self, chunk: np.ndarray) -> None:
        """chunk: (samples, channels) float32/float64 in [-1, 1]."""
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.shape[1] != self.n_channels:
            raise ValueError("channel count changed mid-stream")
        self.n_samples += chunk.shape[0]

        filtered = np.empty_like(chunk, dtype=np.float64)
        for c in range(self.n_channels):
            filtered[:, c] = self._filters[c](chunk[:, c].astype(np.float64))

        squares = filtered * filtered
        self._tail = np.concatenate([self._tail, squares]) if self._tail.size else squares

        # Emit every complete block whose start is on the 100 ms grid.
        start = 0
        while start + BLOCK_N <= self._tail.shape[0]:
            block = self._tail[start:start + BLOCK_N]
            per_channel = block.mean(axis=0)
            self._z.append(float((self._weights * per_channel).sum()))
            start += STEP_N
        if start:
            self._tail = self._tail[start:]

    def blocks(self) -> np.ndarray:
        return np.asarray(self._z, dtype=np.float64)

    def block_times(self) -> np.ndarray:
        """Start time in seconds of each block."""
        return np.arange(len(self._z)) * STEP_S


def block_lufs(z: np.ndarray) -> np.ndarray:
    """Loudness of individual blocks. -inf where the block is digital silence."""
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(z)


def gated_lufs(z: np.ndarray, mask: np.ndarray | None = None) -> float:
    """BS.1770-4 gated loudness over the blocks selected by `mask`.

    This is the function the whole module exists for. The standard applies both
    gates to the whole programme; applying them to a subset is the same
    arithmetic asked of fewer blocks, and it is what makes "how loud is this
    while someone is talking" a well-defined question.

    Returns -inf when nothing survives the gates, which is a real answer (there
    was no measurable audio here) and must not be turned into 0 or None by a
    caller wanting a tidier type.
    """
    z = np.asarray(z, dtype=np.float64)
    if mask is not None:
        z = z[np.asarray(mask, dtype=bool)]
    if z.size == 0:
        return float("-inf")

    l = block_lufs(z)
    above_absolute = l > ABSOLUTE_GATE
    if not above_absolute.any():
        return float("-inf")

    ungated = -0.691 + 10.0 * np.log10(z[above_absolute].mean())
    keep = above_absolute & (l > ungated + RELATIVE_GATE)
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(z[keep].mean()))


def mean_lufs(z: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Loudness of a set of blocks with NO gating at all.

    The gates in `gated_lufs` exist to stop the silence in a whole programme
    dragging its integrated loudness down. Inside a three-second window around
    one line of dialogue there is no such problem, and the relative gate does
    active harm: it drops the quietest blocks, so the level it reports is the
    level of the loudest part of what you asked about.

    Measured 2026-09-01 -- with a deliberately flat bed, where the answer is
    exactly recoverable by arithmetic, gating the local windows put the recovered
    ratio 2.5 LU too high, and every bit of that was the relative gate throwing
    away quiet speech. Power is what adds, so power is what this averages.
    """
    z = np.asarray(z, dtype=np.float64)
    if mask is not None:
        z = z[np.asarray(mask, dtype=bool)]
    if z.size == 0 or z.mean() <= 0:
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(z.mean()))


def lufs_of(samples: np.ndarray) -> float:
    """Whole-signal gated loudness. (samples, channels) or (samples,)."""
    if samples.ndim == 1:
        samples = samples[:, None]
    acc = BlockLoudness(samples.shape[1])
    acc.feed(samples)
    return gated_lufs(acc.blocks())
