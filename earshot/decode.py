"""Getting audio out of an arbitrary file, in blocks, without filling memory.

The box this runs on has under a gigabyte free. An hour of 48 kHz stereo float
is 1.4 GB, so nothing here is ever allowed to hold a whole programme; every
consumer takes chunks. `probe()` is separate from `stream()` on purpose: the
duration has to be known before a decode is started so a caller can refuse a
file rather than discover the size halfway through.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .loudness import SR

CHUNK_SECONDS = 10.0


@dataclass
class Media:
    path: Path
    duration: float
    channels: int          # as decoded, which is 1 or 2, not what the file has
    source_channels: int
    codec: str
    sample_rate: int


class DecodeError(RuntimeError):
    pass


def probe(path: str | Path) -> Media:
    path = Path(path)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels,codec_name,sample_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise DecodeError(out.stderr.strip()[:400] or "ffprobe failed")
    try:
        d = json.loads(out.stdout)
        stream = d["streams"][0]
        duration = float(d["format"]["duration"])
    except (KeyError, IndexError, ValueError) as e:
        raise DecodeError(f"no audio stream in {path.name}") from e

    src = int(stream.get("channels", 1))
    return Media(
        path=path,
        duration=duration,
        channels=1 if src == 1 else 2,
        source_channels=src,
        codec=str(stream.get("codec_name", "?")),
        sample_rate=int(stream.get("sample_rate", 0)),
    )


def stream(media: Media, sr: int = SR, channels: int | None = None,
           chunk_seconds: float = CHUNK_SECONDS) -> Iterator[np.ndarray]:
    """Yield (samples, channels) float32 arrays.

    Anything with more than two channels is downmixed to stereo by ffmpeg,
    which is what a person hears on a television or a laptop and is therefore
    the thing worth measuring. A 5.1 mix measured per-channel would score
    better than it sounds in the room this project is about.
    """
    ch = channels if channels is not None else media.channels
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media.path),
           "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le",
           "-ar", str(sr), "-ac", str(ch), "-"]
    frame_bytes = 4 * ch
    want = int(chunk_seconds * sr) * frame_bytes

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        carry = b""
        while True:
            buf = proc.stdout.read(want)
            if not buf:
                break
            buf = carry + buf
            usable = len(buf) - (len(buf) % frame_bytes)
            carry = buf[usable:]
            if usable:
                a = np.frombuffer(buf[:usable], dtype="<f4")
                yield a.reshape(-1, ch)
        proc.stdout.close()
        code = proc.wait(timeout=60)
        if code != 0:
            raise DecodeError(proc.stderr.read().decode("utf8", "replace")[:400])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def read_all(media: Media, sr: int = SR, channels: int | None = None) -> np.ndarray:
    """For tests and for short files only. Deliberately not used by the analyser."""
    return np.concatenate(list(stream(media, sr=sr, channels=channels)))
