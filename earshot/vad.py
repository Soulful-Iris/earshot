"""Where is somebody talking.

Silero VAD v5 (ONNX, 2.3 MB, MIT) rather than an energy threshold, because the
whole subject here is speech that is quiet *while something else is loud* --
which is exactly the case an energy gate calls silence. A detector that only
finds speech when speech is the loudest thing would agree with me about every
mix and disagree about none, and this record has a long list of instruments
like that.

The output is a per-sample-block probability at 16 kHz, turned into a mask over
the 100 ms loudness grid so it can be handed straight to `loudness.gated_lufs`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import decode
from .loudness import BLOCK_S, STEP_S

VAD_SR = 16000
WINDOW = 512                      # samples the v5 model requires at 16 kHz
WINDOW_S = WINDOW / VAD_SR        # 32 ms
MODEL = Path(__file__).resolve().parents[1] / "models" / "silero_vad.onnx"

# The model wants the last 64 samples of the PREVIOUS window prepended to the
# current one, so each call sees 576 samples. This is not optional and it is
# not documented in the graph: without it the model returns about 0.001 for
# every window of clean, loud speech. Measured 2026-09-01 on fixtures/dry.wav,
# which is nothing but a voice -- the detector said 0% speech and every number
# downstream was quietly meaningless while the code ran without an error.
# `tools/recover.py` is what caught it, because a fixture of pure speech has an
# answer I knew before I measured it.
CONTEXT = 64

# Above this the window counts as speech. 0.5 is the model's own default and is
# a hair trigger on music; measured on the fixtures, 0.6 keeps music out
# without losing quiet dialogue. Changing it changes every number downstream,
# so it lives here and nowhere else.
SPEECH_P = 0.6


class Vad:
    def __init__(self, model: Path = MODEL, threshold: float = SPEECH_P) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._s = ort.InferenceSession(str(model), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._sr = np.array(VAD_SR, dtype=np.int64)
        self.probs: list[float] = []

    def feed(self, mono16k: np.ndarray) -> None:
        x = np.concatenate([self._carry, np.asarray(mono16k, dtype=np.float32).ravel()])
        n = (len(x) // WINDOW) * WINDOW
        for i in range(0, n, WINDOW):
            window = x[i:i + WINDOW]
            out, self._state = self._s.run(
                None, {"input": np.concatenate([self._context, window])[None, :],
                       "state": self._state,
                       "sr": self._sr})
            self._context = window[-CONTEXT:]
            self.probs.append(float(out[0][0]))
        self._carry = x[n:]

    def window_speech(self) -> np.ndarray:
        return np.asarray(self.probs) >= self.threshold


def speech_windows(media: decode.Media, threshold: float = SPEECH_P) -> np.ndarray:
    """Boolean per 32 ms window, for the whole programme."""
    v = Vad(threshold=threshold)
    for chunk in decode.stream(media, sr=VAD_SR, channels=1):
        v.feed(chunk[:, 0])
    return v.window_speech()


def speech_fraction_per_block(windows: np.ndarray, n_blocks: int) -> np.ndarray:
    """What share of each 400 ms loudness block is speech.

    The loudness blocks overlap by 75%, so a window belongs to four of them.
    Returning the fraction rather than a boolean is what lets `measure.py` keep
    a deliberately empty margin between "speech" and "background" instead of
    forcing every ambiguous block into one of them.
    """
    out = np.zeros(n_blocks)
    per_block = max(1, int(round(BLOCK_S / WINDOW_S)))
    for b in range(n_blocks):
        start = int(round(b * STEP_S / WINDOW_S))
        seg = windows[start:start + per_block]
        out[b] = float(seg.mean()) if seg.size else 0.0
    return out
