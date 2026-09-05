"""The video you keep after you sing.

A take produces a sentence and a number and then deletes itself. This turns it
into one file: the record's own picture, your voice in the room measured off
that record, and your pitch line drawn against the actual singer's, scrolling
under a fixed playhead.

Everything it needs already exists on disk when it runs:

    out/no-voice.mp4   the record's picture, karaoke words already burned in
    out/placed.mp3     your take in the record's measured room, over the band
    status.json        unison.contours -- t, ref, you, centre, span, on_note

THE STRIP, AND WHY IT IS ONE IMAGE. The whole song is drawn once into a single
wide RGBA PNG and scrolled by one `overlay` whose x is a function of t. The
obvious alternative is a PNG per frame, which for a three and a half minute
song at 15fps is about three thousand files on a box with 1.8 GB of memory.
One image is ~11 MB and one filter expression.

`-loop 1` on that image input is load-bearing. Without it the image is a single
frame, `overlay` ends with it, and the output is ONE FRAME LONG while every
other check passes: ffprobe reports a valid file, the audio is right, the
encode exits 0. That happened on the first run of the probe that proved this
technique and it reported as a clean success.

No image library. numpy for the canvas because it is already a dependency and a
2.7 million pixel buffer in pure Python is not worth the purity, and `zlib` plus
`struct` for the PNG itself.
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

# 110 px per second puts about six seconds in the 640px window, which is the
# window the live ribbon on the page uses. Changing this changes how fast the
# song appears to move past the playhead, and it is the one number here that is
# a feel decision rather than a measurement.
PX_PER_SEC = 110
STRIP_H = 120
# TOP, not bottom. The plan's sketch has the record's karaoke words "burned in
# near the top" and put the ribbon along the bottom. Measured on the real
# no-voice.mp4, the words are at y 300-329 of 360 -- the bottom -- so the plan's
# layout put the ribbon directly on top of them, and the first render has both
# sets of marks in the same strip of picture.
TOP_MARGIN = 10

GOLD = (200, 162, 74)
WHITE = (255, 255, 255)
GRID = (22, 26, 32)
FILL_ALPHA = 0.30
REF_WIDTH = 3
YOU_WIDTH = 2

# A scrim under the whole strip. On the page the ribbon sits on a near-black
# panel; over footage the lines had nothing to read against and the gridlines
# came out as dark bars across the picture. 40% black is enough to carry a
# 1px gridline and still show the video through it.
SCRIM = (6, 8, 11)
SCRIM_ALPHA = 0.40
GRID_ALPHA = 0.55

# The vertical window follows the melody, as it does on the page. Centring on
# the song's overall median put every phrase that is not average against an
# edge and clipped half of them off the box. `span` semitones stays fixed so a
# gap always means the same distance; only where the box sits moves.
FOLLOW_WINDOW_S = 6.0
FOLLOW_STEP_S = 0.25

NAME = "encore.mp4"


# --- the picture -------------------------------------------------------------

def _png(rgba: np.ndarray) -> bytes:
    """An RGBA numpy array as PNG bytes. Filter 0 on every row."""
    h, w, _ = rgba.shape
    raw = np.concatenate(
        [np.zeros((h, 1), dtype=np.uint8), rgba.reshape(h, w * 4)], axis=1)
    comp = zlib.compress(raw.tobytes(), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", comp)
            + chunk(b"IEND", b""))


def _resample(t: np.ndarray, v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Contour values on the pixel grid, with the silences preserved.

    Interpolating straight through a null would draw a line across the gaps
    between phrases, which is exactly what the live ribbon refuses to do: it
    lifts the pen at a null rather than joining the last note of one line to
    the first of the next.
    """
    good = np.isfinite(v)
    if good.sum() < 2:
        return np.full(grid.shape, np.nan)
    out = np.interp(grid, t[good], v[good], left=np.nan, right=np.nan)
    # Blank any pixel whose nearest real sample is a null, so gaps stay gaps.
    nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(nearest - 1, 0, len(t) - 1)
    pick = np.where(np.abs(t[nearest] - grid) <= np.abs(t[prev] - grid),
                    nearest, prev)
    out[~good[pick]] = np.nan
    return out


def _follow(t: np.ndarray, ref: np.ndarray, grid: np.ndarray,
            centre: float) -> np.ndarray:
    """The centre of the vertical window at each pixel, tracking the melody."""
    if not np.isfinite(ref).any():
        return np.full(grid.shape, centre)
    coarse = np.arange(grid[0], grid[-1] + FOLLOW_STEP_S, FOLLOW_STEP_S)
    mids = np.empty(coarse.shape)
    half = FOLLOW_WINDOW_S / 2
    for i, c in enumerate(coarse):
        m = (t >= c - half) & (t < c + half) & np.isfinite(ref)
        mids[i] = float(np.median(ref[m])) if m.any() else np.nan
    if np.isfinite(mids).any():
        idx = np.arange(len(mids))
        ok = np.isfinite(mids)
        mids = np.interp(idx, idx[ok], mids[ok])
    else:
        mids[:] = centre
    # Smooth it, or the baseline steps whenever a phrase enters the window and
    # the whole ribbon jumps with it.
    k = max(3, int(round(FOLLOW_WINDOW_S / FOLLOW_STEP_S)) | 1)
    pad = np.pad(mids, k // 2, mode="edge")
    mids = np.convolve(pad, np.ones(k) / k, mode="valid")
    return np.interp(grid, coarse[:len(mids)], mids)


def _plot(canvas: np.ndarray, ys: np.ndarray, colour, width: int) -> None:
    """Draw one line by filling between consecutive columns.

    Column-to-column rather than point-to-point: a pitch line that jumps a
    semitone between two pixels should be a continuous stroke, not two dots
    with a hole between them.
    """
    h = canvas.shape[0]
    valid = np.isfinite(ys)
    r, g, b = colour
    for x in range(len(ys)):
        if not valid[x]:
            continue
        y0 = ys[x]
        y1 = ys[x - 1] if x > 0 and valid[x - 1] else y0
        lo = int(math.floor(min(y0, y1))) - width // 2
        hi = int(math.ceil(max(y0, y1))) + width // 2 + 1
        lo, hi = max(0, lo), min(h, hi)
        if hi <= lo:
            continue
        canvas[lo:hi, x] = (r, g, b, 255)


def strip(contours: dict, seconds: float, shift_ms: float = 0.0) -> np.ndarray:
    """The whole song's two pitch lines, as one wide RGBA image.

    `shift_ms` puts YOUR line where your voice actually sounds, and without it
    the picture disagrees with the audio by however late you were.

    Wren found this, and it was not on the list of five things I asked him to
    check. `unison.contours()` stamps every sample with the REFERENCE frame's
    time (`ts.append(ja * HOP / SR)`) while reading your value from the take
    frame at `jb = i`, which is `shift` frames away. That is correct and
    deliberate for SCORING -- it is exactly what cancels recording latency
    before comparing pitch. But `placed.mp3` is mixed by `_mix()` with no shift
    applied anywhere, sample zero to sample zero, so the audio plays on the
    take's raw clock while the white line was drawn on the reference's.

    They are the same clock only when `shift == 0`. Measured on three real
    takes on this disk: -23.22 ms, -92.88 ms and -371.52 ms. A third of a
    second is not subtle.

    In both branches of `contours()` the stored time is the true time minus
    `shift * HOP / SR`, so the correction is one sign for both: read the stored
    series at `grid + shift_ms/1000`.
    """
    t = np.asarray(contours["t"], dtype=float)
    ref = np.array([np.nan if v is None else v for v in contours["ref"]], dtype=float)
    you = np.array([np.nan if v is None else v for v in contours["you"]], dtype=float)
    span = float(contours.get("span") or 12.0)
    centre = float(contours.get("centre") or 0.0)
    on_note = float(contours.get("on_note") or 0.5)

    w = max(2, int(math.ceil(seconds * PX_PER_SEC)))
    grid = np.arange(w, dtype=float) / PX_PER_SEC
    rs = _resample(t, ref, grid)
    ys = _resample(t, you, grid + shift_ms / 1000.0)
    mid = _follow(t, ref, grid, centre)

    top, bot = mid + span / 2, mid - span / 2
    def to_y(v):
        return STRIP_H - (v - bot) / np.maximum(top - bot, 1e-9) * STRIP_H

    yr, yy = to_y(rs), to_y(ys)
    canvas = np.zeros((STRIP_H, w, 4), dtype=np.uint8)
    canvas[:, :] = (*SCRIM, int(round(255 * SCRIM_ALPHA)))

    # semitone gridlines, faint, so a gap on screen has a scale
    ga = int(round(255 * GRID_ALPHA))
    for s in range(int(math.floor(bot.min())), int(math.ceil(top.max())) + 1):
        gy = to_y(np.full(w, float(s)))
        ok = np.isfinite(gy) & (gy >= 0) & (gy < STRIP_H)
        canvas[gy[ok].astype(int), np.arange(w)[ok]] = (*GRID, ga)

    # the fill first, under both lines: solid means you were on the note
    both = np.isfinite(yr) & np.isfinite(yy) & (np.abs(rs - ys) <= on_note)
    fa = int(round(255 * FILL_ALPHA))
    for x in np.nonzero(both)[0]:
        a, b = sorted((yr[x], yy[x]))
        lo, hi = max(0, int(a)), min(STRIP_H, int(b) + 1)
        if hi > lo:
            canvas[lo:hi, x] = (*GOLD, fa)

    _plot(canvas, yr, GOLD, REF_WIDTH)
    _plot(canvas, yy, WHITE, YOU_WIDTH)
    return canvas


# --- the file ----------------------------------------------------------------

def _duration(path: Path) -> float | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def render(job_dir: Path, timeout: int = 3600) -> Path | None:
    """The take as a video, or None with the reason written to status.

    Refuses rather than producing something empty. A video of silence, or of a
    picture with no ribbon on it, is worse than no video: it looks finished.
    """
    job_dir = Path(job_dir)
    out = job_dir / "out"
    try:
        st = json.loads((job_dir / "status.json").read_text())
    except Exception:                                          # noqa: BLE001
        return None

    u = st.get("unison") or {}
    lr = st.get("liveroom") or {}
    vids = st.get("videos") or {}

    if u.get("state") != "done" or not (u.get("contours") or {}).get("t"):
        return None
    if lr.get("state") != "done" or not lr.get("placed"):
        return None
    no_voice = vids.get("no_voice")
    if not no_voice:
        return None

    picture, audio = out / no_voice, out / lr["placed"]
    if not (picture.is_file() and audio.is_file()):
        return None

    secs = _duration(audio) or 0.0
    if secs < 1.0:
        return None

    # The take's own latency, so the white line lands where the voice sounds
    # rather than where the scoring compared it. See strip().
    shift_ms = float(u.get("shift_ms") or 0.0)
    png = out / "_strip.png"
    png.write_bytes(_png(strip(u["contours"], secs + 2.0, shift_ms)))

    dest = out / NAME
    # overlay and drawbox do NOT share a variable namespace. overlay has W/H
    # for the main input; drawbox has iw/ih and no W at all, so the `(W/2)-2`
    # the plan specified is an "Error when evaluating the expression" and the
    # whole encode exits 234. The plan's chain was quoted as verified, and the
    # half that was actually run used literal pixel numbers in the drawbox.
    y_ov = str(TOP_MARGIN)
    y_box = str(TOP_MARGIN)
    chain = (
        f"[1:v]format=rgba[s];"
        f"[0:v][s]overlay=x='(W/2)-(t*{PX_PER_SEC})':y={y_ov}:eof_action=pass[o];"
        f"[o]drawbox=x='(iw/2)-2':y={y_box}:w=4:h={STRIP_H}:"
        f"color=white@0.55:t=fill[v]"
    )
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y",
             "-i", str(picture),
             # -loop 1 or the overlay ends with the single-frame image and the
             # whole output is one frame long. See the module docstring.
             "-loop", "1", "-i", str(png),
             "-i", str(audio),
             "-filter_complex", chain,
             # -t, not just -shortest. `-shortest` stops READING at the
             # shortest input and then flushes whatever the video encoder has
             # buffered, which came out seven frames long: audio 58.62s,
             # video 59.02s, on a take of 58.67s. The take's length is known
             # here, so say it rather than infer it.
             "-map", "[v]", "-map", "2:a", "-shortest", "-t", f"{secs:.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", str(dest)],
            capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            # Say what ffmpeg said. The first version raised through
            # check=True with the output captured and nothing printing it, so
            # a filter-expression error arrived as "exit status 234" and cost
            # a round trip to see one line that named the problem exactly.
            tail = " / ".join((r.stderr or "").strip().splitlines()[-3:])
            raise RuntimeError(f"ffmpeg did not build the encore: {tail}")
    finally:
        png.unlink(missing_ok=True)

    if not dest.is_file() or dest.stat().st_size < 1024:
        return None
    return dest
