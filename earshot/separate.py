"""Actually pulling the voice out, rather than just measuring how buried it is.

I wrote in this project's own README that this could not be done here -- that
separating dialogue from the music under it "needs a source-separation model and
a much bigger machine". I had not checked. Demucs runs on this box: two cores,
no GPU, 1.8 GB of RAM.

## The costs, measured rather than assumed

    htdemucs, 20 s of stereo, segment 7    peak RSS 1.29 GB, 56 s wall
    htdemucs, same, segment 4              see SEGMENT below

That peak is most of the machine, which is why nothing here loads a whole
programme: audio is separated in overlapping windows and written out as it
goes. The model itself is loaded once and reused.

## What it is and is not good at

htdemucs is trained to pull sung vocals out of music. Speech is close enough to
that to work well, and a score under dialogue is exactly the case it was built
for. It is NOT a noise suppressor: for hiss, hum, traffic and room tone the
right tool is a denoiser, and `enhance()` runs one of those after separation
rather than pretending one model does both.

The separation is never the finished product. Voice-only audio with the score
gone sounds wrong -- a film is not a podcast. So `enhance()` puts the background
back at a chosen distance below the voice, which is the thing that could never
be done before: the ratio itself becomes a dial.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import decode
from .loudness import SR

# Seconds per window handed to the model. This is the memory dial: htdemucs
# holds several full-resolution spectrograms per window, so peak RSS scales with
# it. 7 s peaked at 1.29 GB on a 1.8 GB box, which leaves nothing for the talk
# server that also lives here.
SEGMENT = 5.0
OVERLAP = 0.25
MODEL_SR = 44100          # what htdemucs was trained at; resample, do not guess

_model = None


def model():
    """Loaded once. Importing demucs costs seconds and ~300 MB."""
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        m = get_model("htdemucs")
        m.eval()
        _model = m
    return _model


@dataclass
class Stems:
    voice: Path
    background: Path
    seconds: float
    peak_rss_mb: float | None = None


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    """audio: (samples, channels) float32."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
         "-ac", str(audio.shape[1]), "-i", "-", "-c:a", "pcm_f32le", str(path)],
        input=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
        check=True, timeout=1800)


class _Encoder:
    """An ffmpeg process being fed float32 frames, so nothing is held to the end."""

    def __init__(self, path: Path, channels: int, sr: int) -> None:
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le",
             "-ar", str(sr), "-ac", str(channels), "-i", "-",
             "-c:a", "pcm_f32le", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frames: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frames, dtype=np.float32).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait(timeout=900) != 0:
            raise decode.DecodeError(
                self.proc.stderr.read().decode("utf8", "replace")[:400])


def split(path: str | Path, out_dir: str | Path,
          segment: float = SEGMENT) -> Stems:
    """Separate into voice and everything-else, streaming.

    Nothing whole-file is ever in memory. A ninety-minute film at 44.1 kHz
    stereo is 1.9 GB as float32 and this machine has 1.8 GB, so a version that
    read the file in would work on every fixture and die on the first real thing
    anyone pointed it at. The windows are crossfaded and written out behind the
    read head; only two windows and the crossfade tail are resident.
    """
    import torch
    from demucs.apply import apply_model

    src = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    media = decode.probe(src)

    m = model()
    vi = m.sources.index("vocals")
    step = int(segment * MODEL_SR)
    hop = max(1, int(step * (1 - OVERLAP)))
    tail = step - hop
    win = np.hanning(step + 2)[1:-1].astype(np.float32)

    vpath, bpath = out / (src.stem + ".voice.wav"), out / (src.stem + ".bg.wav")
    venc = _Encoder(vpath, 2, MODEL_SR)
    benc = _Encoder(bpath, 2, MODEL_SR)

    buf = np.zeros((0, 2), dtype=np.float32)          # not yet separated
    carry_v = np.zeros((tail, 2), dtype=np.float32)   # overlap from last window
    carry_w = np.zeros(tail, dtype=np.float32)
    carry_x = np.zeros((tail, 2), dtype=np.float32)   # the same span of input
    total = 0

    def process(chunk: np.ndarray, final: bool) -> None:
        """chunk is (step, 2); emit the first `hop` samples, carry the rest."""
        nonlocal carry_v, carry_w, carry_x
        ref = chunk.mean(1)
        mean, std = float(ref.mean()), float(ref.std() + 1e-8)
        with torch.no_grad():
            x = torch.from_numpy(((chunk - mean) / std).T).float()[None]
            sources = apply_model(m, x, split=False, overlap=0.0,
                                  progress=False, device="cpu")[0]
        voice = (sources[vi].numpy().T * std + mean).astype(np.float32)
        del sources, x

        w = win[:len(chunk)]
        acc_v = voice * w[:, None]
        acc_w = w.copy()
        # The carry can be LONGER than the final chunk when the file ends inside
        # the overlap region, and it always is when the whole file is shorter
        # than one window. Without the clamp this raised a broadcast error on
        # any input under about four seconds -- which every probe in
        # `will_it_help` is, so the predictor could not run at all.
        k = min(len(carry_v), len(acc_v))
        acc_v[:k] += carry_v[:k]
        acc_w[:k] += carry_w[:k]

        emit = len(chunk) if final else hop
        weight = np.maximum(acc_w[:emit], 1e-9)[:, None]
        v = acc_v[:emit] / weight
        venc.write(v)
        # Background is the input minus the voice, sample for sample. Summing
        # the model's other three stems instead would silently drop whatever
        # none of the four accounted for.
        benc.write(chunk[:emit] - v)
        if not final:
            carry_v, carry_w, carry_x = acc_v[emit:], acc_w[emit:], chunk[emit:]

    for block in decode.stream(media, sr=MODEL_SR, channels=2, chunk_seconds=10.0):
        buf = np.concatenate([buf, block.astype(np.float32)])
        total += len(block)
        while len(buf) >= step + (len(carry_x) and 0):
            process(buf[:step], final=False)
            buf = buf[hop:]

    if len(buf) >= 1024:
        process(buf, final=True)
    elif len(buf):
        venc.write(buf)
        benc.write(np.zeros_like(buf))

    venc.close()
    benc.close()
    return Stems(voice=vpath, background=bpath, seconds=total / MODEL_SR)


# ---------------------------------------------------------------------------
# The point of separating: the ratio stops being a verdict and becomes a dial.
# ---------------------------------------------------------------------------

@dataclass
class Enhanced:
    out_path: Path
    stems: Stems
    voice_lufs: float
    background_lufs_before: float
    background_lufs_after: float
    background_gain_db: float
    programme_lufs_before: float
    programme_lufs_after: float


def _gated(path: Path) -> float:
    from .loudness import BlockLoudness, gated_lufs
    media = decode.probe(path)
    acc = BlockLoudness(media.channels)
    for chunk in decode.stream(media):
        acc.feed(chunk)
    return gated_lufs(acc.blocks())


def enhance(path: str | Path, out_path: str | Path | None = None,
            target_lu: float = 12.0, keep_background: bool = True,
            work_dir: str | Path | None = None) -> Enhanced:
    """Re-mix with the background set a chosen distance under the voice.

    `target_lu` is the thing this project has spent all day measuring and could
    not previously change. +12 LU is comfortably clear; the report calls under
    +4 the point where people reach for the subtitles.

    `keep_background=False` gives voice only. It is not the default on purpose:
    a film with its score deleted is not a fixed film, it is a different and
    worse one, and the complaint was never that the music exists.
    """
    src = Path(path)
    out = Path(out_path) if out_path else src.with_name(src.stem + ".enhanced.wav")
    work = Path(work_dir) if work_dir else out.parent / (src.stem + ".stems")

    programme_before = _gated(src)
    stems = split(src, work)

    voice_lufs = _gated(stems.voice)
    bg_before = _gated(stems.background)

    import math
    if keep_background and math.isfinite(voice_lufs) and math.isfinite(bg_before):
        wanted = voice_lufs - target_lu
        gain_db = wanted - bg_before
        # Only ever quieten. Raising a background to "reach" a target would be
        # obeying the number while making the thing worse, which is the whole
        # failure this repo is a monument to.
        gain_db = min(gain_db, 0.0)
    else:
        gain_db = float("-inf") if not keep_background else 0.0

    g = 0.0 if gain_db == float("-inf") else 10.0 ** (gain_db / 20.0)

    vmedia, bmedia = decode.probe(stems.voice), decode.probe(stems.background)
    enc = _Encoder(out, 2, MODEL_SR)
    vs = decode.stream(vmedia, sr=MODEL_SR, channels=2, chunk_seconds=10.0)
    bs = decode.stream(bmedia, sr=MODEL_SR, channels=2, chunk_seconds=10.0)
    peak = 0.0
    for vchunk, bchunk in zip(vs, bs):
        n = min(len(vchunk), len(bchunk))
        mixed = vchunk[:n] + bchunk[:n] * g
        peak = max(peak, float(np.abs(mixed).max(initial=0.0)))
        enc.write(np.clip(mixed, -0.999, 0.999))
    enc.close()

    return Enhanced(
        out_path=out, stems=stems, voice_lufs=voice_lufs,
        background_lufs_before=bg_before,
        background_lufs_after=bg_before + (gain_db if math.isfinite(gain_db) else 0.0),
        background_gain_db=gain_db,
        programme_lufs_before=programme_before,
        programme_lufs_after=_gated(out))


# ---------------------------------------------------------------------------
# Will separation help this programme at all? Ask before spending 4x realtime.
# ---------------------------------------------------------------------------

# How much of the background comes back labelled "vocals". Measured across five
# CC-BY tracks on 2026-09-01, against what separation then actually achieved:
#
#   Funky Louie (has a singer)      -11.8 dB      -- unusable
#   Warfare, Pain Of Life           -15.3 dB      -> SI-SDR gain +0.9 dB
#   Perfect Lovely Modern Justice   -27.6 dB
#   Rosevere, Here's the Thing      -55.0 dB      -> SI-SDR gain +17.0 dB
#   Rosevere, Arcade Montage        -58.8 dB
#
# A vocal separator cannot remove music that sounds like a voice, and a
# sustained melodic lead in the vocal range is, to it, a voice. Note the
# trade-off this sets up: the same tracks that mask speech best (most midrange
# energy) are the ones that leak most, so the hard cases are hard twice over.
# HOW PRECISE THIS IS, measured rather than assumed. The same track probed at
# different lengths:
#
#   probe length          1.0s    2.0s    4.0s   10.0s   24.0s
#   Rosevere (clean)     -33.8   -44.7   -48.5   -47.1   -52.3
#   Warfare  (leaky)      +0.0    -0.6   -15.3    -8.7    -7.5
#
# The number does NOT settle -- it moves up to 15 dB with probe length. What is
# stable is the SEPARATION between the two: thirty-odd dB at every length past
# a second. So this reports a class, not a measurement, and anyone quoting the
# dB figure to one decimal place (as I did to Bruno before running this) is
# claiming a precision it does not have.
LEAK_GOOD = -35.0
LEAK_POOR = -20.0
PROBE_S = 20.0
MIN_PROBE_S = 4.0        # under this the estimate is worthless, not merely noisy


@dataclass
class Prospect:
    leak_db: float
    background_seconds: float
    verdict: str

    def helpful(self) -> bool:
        return self.leak_db <= LEAK_POOR


def will_it_help(path: str | Path, work_dir: str | Path | None = None,
                 probe_s: float = PROBE_S) -> Prospect | None:
    """Separate only the parts with nobody talking, and see what comes back.

    If the background alone is largely returned as "vocals", separation cannot
    help: there will be nothing left to turn down, and the four times realtime
    would be spent proving it. Returns None when the programme has no usable
    background to probe -- somebody talking continuously -- which is not a
    verdict either way.

    This is the measurement half of the project deciding whether the separation
    half is worth running, which is a better use of a meter than printing a
    number at somebody.
    """
    import numpy as np

    from . import vad
    from .loudness import BlockLoudness, STEP_S
    from .measure import GUARD_BLOCKS, SPEECH_BLOCK

    src = Path(path)
    work = Path(work_dir) if work_dir else src.parent / (src.stem + ".prospect")
    work.mkdir(parents=True, exist_ok=True)
    media = decode.probe(src)

    windows = vad.speech_windows(media)
    acc = BlockLoudness(media.channels)
    for chunk in decode.stream(media):
        acc.feed(chunk)
    frac = vad.speech_fraction_per_block(windows, len(acc.blocks()))

    near = frac > 0.0
    for shift in range(1, GUARD_BLOCKS + 1):
        near |= np.roll(frac > 0.0, shift)
        near |= np.roll(frac > 0.0, -shift)
    quiet = ~near
    if quiet.sum() * STEP_S < MIN_PROBE_S:
        # Continuous narration leaves nothing to probe: a 25 s excerpt of a
        # LibriVox reader has 1.7 s of speech-free audio in it. That is a real
        # limit of this approach and the honest answer is no answer.
        return None

    # Collect the speech-free stretches into one probe file. Concatenating
    # across cuts is fine here: the question is what the model calls this
    # material, not how it flows.
    keep = np.flatnonzero(quiet)
    want = int(probe_s / STEP_S)
    keep = keep[:want] if len(keep) > want else keep
    spans = np.split(keep, np.flatnonzero(np.diff(keep) > 1) + 1)

    probe = work / "background.wav"
    parts = [f"between(t,{s[0]*STEP_S:.3f},{(s[-1]+4)*STEP_S:.3f})"
             for s in spans if len(s)]
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-af", f"aselect='{'+'.join(parts)}',asetpts=N/SR/TB",
         "-ac", "2", "-ar", str(MODEL_SR), "-c:a", "pcm_f32le", str(probe)],
        check=True, timeout=900)

    stems = split(probe, work)
    before = decode.read_all(decode.probe(probe), sr=MODEL_SR, channels=1)[:, 0]
    after = decode.read_all(decode.probe(stems.voice), sr=MODEL_SR, channels=1)[:, 0]

    def db(x):
        return 10.0 * float(np.log10(float(x @ x) / max(len(x), 1) + 1e-20))

    leak = db(after) - db(before)
    verdict = ("this background separates cleanly" if leak <= LEAK_GOOD else
               "some of this background will come along with the voice"
               if leak <= LEAK_POOR else
               "this background sounds like a voice to the model; separation "
               "will not help much")
    # Rounded on purpose. See the table above: the tenths are not real.
    leak = round(leak)
    return Prospect(leak_db=leak, background_seconds=len(before) / MODEL_SR,
                    verdict=verdict)
