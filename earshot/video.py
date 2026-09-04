"""Two videos out of one link: one with the singer, one without, both subtitled.

Bruno, 2026-09-03: *"You are supposed to have 2 versions when i put a youtube
link. 2 videos. One with voice. One without. Both subtitled"*.

What existed before this was audio stems and a JSON file of word timings, which
are the ingredients and not the dish. This is the last ten feet.

THREE THINGS MEASURED BEFORE ANY OF IT WAS WRITTEN, because each one would have
been a confident wall report otherwise:

1. **The video is already being downloaded and thrown away.** The archive offers
   exactly ONE format for these URLs (fmt 37, mp4, h264 + aac, 1920x1080), so
   `-f bestaudio/best -x` was pulling the whole picture and discarding it.
   Keeping it costs nothing.

2. **`fc-list` reporting zero fonts was `fc-list` not being installed.** An
   absent command prints nothing and exits, which looks exactly like a command
   reporting an empty result. There are 95 font files on this box, unpacked out
   of RPMs with chromium on 2026-09-01, and nothing had told fontconfig they
   were there.

3. **libass rendered anyway, and that was the trap.** With no fontconfig it fell
   back to a thin grey serif at default size: exit 0, twelve PNGs, 180 non-black
   pixels, and legible only because I was looking at a black background. Over
   real video it would have been invisible. `SUB_STYLE` and `_env()` below are
   what make it 1372 pixels of bold outlined white instead, and `subs_landed()`
   is what stops me ever trusting the exit code again.
"""
from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

# Written by hand rather than found: see point 2 above. Shipped in the repo so a
# rebuilt box gets it, and pointed at by _env() rather than by whoever launches
# the process -- the same environment bug has bitten this project three times
# from the launcher side.
FONTS_CONF = Path(__file__).with_name("fonts.conf")

# Bold, white, thick black outline, lifted off the bottom edge. Every part of
# this is doing a job: Outline=3 is what keeps it readable over a bright frame,
# and MarginV keeps it clear of a phone's own chrome.
#
# Fontsize is computed from the OUTPUT HEIGHT, not fixed. ASS sizes are in
# script units and the subtitles filter takes the video resolution as the
# script resolution, so a flat `Fontsize=22` is large on a 360p capture and
# half the relative size on a 720p one. Both are real inputs here.
def sub_style(height: int) -> str:
    size = max(16, round(height / 16))          # 22 at 360p, 45 at 720p
    outline = max(2, round(height / 240))       # scales with it or it vanishes
    return (f"FontName=DejaVu Sans,Fontsize={size},Bold=1,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline={outline},Shadow=1,"
            f"MarginV={round(height / 8)}")

# A subtitle cue breaks when the singing pauses this long, or the line gets this
# wide, or it has been up this long. 42 characters is about the most that fits
# one line on a phone at this size without wrapping into three.
GAP = 0.7
MAX_CHARS = 42
MAX_SECONDS = 5.0

# Measured on a 20 s slice of a real 1080p capture, downscaled to 720p:
#
#     crf 23   2917 kbps   11.3 s
#     crf 26   2064 kbps   10.5 s
#     crf 28   1645 kbps   10.3 s
#
# The knob costs almost no time and a great deal of size. 23 put a four-minute
# song at 99 MB, which is a lot to push down a phone's mobile data for something
# somebody is going to sing over once. 26 is about 62 MB and I cannot see the
# difference on this material; 28 starts to show on the busy frames.
CRF = 26


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list | None = None   # the individual words, for the karaoke tags


def _env() -> dict:
    """ffmpeg's environment, with fontconfig pointed at fonts that exist.

    In here and not in a wrapper script, because the same class of bug has now
    cost this project three separate evenings: `claude` stripping its child's
    token, a systemd unit with no ~/.local/bin on PATH, and this. A fix that
    lives in the launcher only works for the launcher you tested.
    """
    env = dict(os.environ)
    if FONTS_CONF.is_file():
        env["FONTCONFIG_FILE"] = str(FONTS_CONF)
    return env


def cues(words: list, gap: float = GAP, max_chars: int = MAX_CHARS,
         max_seconds: float = MAX_SECONDS) -> list[Cue]:
    """Group timed words into readable subtitle lines.

    Takes `words.Word`s, or anything with .word/.start/.end.
    """
    out: list[Cue] = []
    buf: list = []

    def flush() -> None:
        if not buf:
            return
        # A line must not open on a comma. The recogniser attaches punctuation
        # to the word it follows, so a break can land after "all," and leave
        # the next line starting ", i got real estate", which reads as a typo
        # rather than as a lyric.
        first = buf[0]
        clean = first.word.lstrip(",.;:!?-'\" ")
        if clean != first.word:
            buf[0] = replace(first, word=clean) if is_dataclass(first) else \
                SimpleNamespace(word=clean, start=first.start, end=first.end,
                                confidence=getattr(first, "confidence", 0.0))
        kept = [w for w in buf if w.word.strip()]
        if kept:
            out.append(Cue(kept[0].start, kept[-1].end,
                           " ".join(w.word for w in kept), list(kept)))
        buf.clear()

    for w in words:
        if buf:
            wide = len(" ".join(x.word for x in buf)) + 1 + len(w.word) > max_chars
            long = w.end - buf[0].start > max_seconds
            paused = w.start - buf[-1].end > gap
            if wide or long or paused:
                flush()
        buf.append(w)
    flush()

    # Never let two cues overlap: a reader sees the later one clobber the
    # earlier one and it reads as a glitch rather than as timing.
    for a, b in zip(out, out[1:]):
        if a.end > b.start:
            a.end = b.start - 0.01
    return [c for c in out if c.end > c.start and c.text.strip()]


def _stamp(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round(s % 1 * 1000)):03d}"


def srt(cs: list[Cue]) -> str:
    return "".join(
        f"{i}\n{_stamp(c.start)} --> {_stamp(c.end)}\n{c.text}\n\n"
        for i, c in enumerate(cs, 1))


def write_srt(cs: list[Cue], path: Path) -> Path:
    path = Path(path)
    path.write_text(srt(cs), encoding="utf-8")
    return path


# --- word by word, the way a singalong does it -------------------------------
#
# Bruno, 2026-09-04: "subtitles should be word for word. Like a disney
# singalong". Plain SRT puts a whole line up at once, which tells you WHAT to
# sing and not WHEN. libass does the real thing natively with `\k` karaoke
# tags: the line goes up in one colour and each word flips to another exactly
# as it is sung.
#
# The timings this rides on are the ones read off the isolated vocal, so the
# highlight follows the actual recording rather than a lyrics database's guess
# at where the lines fall. That is also why it is gated: a whole line that
# drifts a little is survivable, but a word highlighting on the wrong syllable
# is obvious to anybody, immediately, and it is the thing you are looking at
# while you sing.

# ASS colours are &HBBGGRR, backwards from hex you would write anywhere else.
SUNG = "&H00E7FF"      # warm gold, the word you are on and the ones behind it
TO_SING = "&HFFFFFF"   # plain white, the ones still coming


def ass(cs: list, height: int) -> str:
    """An .ass subtitle file whose words light up one at a time."""
    size = max(16, round(height / 16))
    outline = max(2, round(height / 240))
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {round(height * 16 / 9)}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sing,DejaVu Sans,{size},{SUNG},{TO_SING},&H000000,&H000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,40,40,{round(height / 8)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""
    lines = []
    for c in cs:
        # A karaoke line has to APPEAR before its first word is sung, or the
        # first word lights up at the same moment the line arrives and there is
        # nothing to read ahead. Quarter of a second is enough to register.
        start = max(0.0, c.start - 0.25)
        text = rf"{{\k{int(round((c.start - start) * 100))}}}"
        prev = c.start
        for w in (c.words or []):
            # The SILENCE before a word gets its own \k, on the space. Without
            # it the durations sum to less than the line lasts and the
            # highlight runs ahead of the singing a little more with every
            # word: 1.85s of tags across a 1.95s line on the first thing I
            # rendered. \k is a duration in centiseconds, never a timestamp,
            # so every hundredth has to be spent somewhere.
            gap = int(round(max(0.0, w.start - prev) * 100))
            if gap:
                text += rf"{{\k{gap}}} "
            elif text and not text.endswith(" "):
                text += " "
            text += rf"{{\k{max(1, int(round((w.end - w.start) * 100)))}}}{w.word}"
            prev = w.end
        lines.append(f"Dialogue: 0,{_ass_stamp(start)},{_ass_stamp(c.end)},"
                     f"Sing,,0,0,0,,{text.strip()}")
    return head + "\n".join(lines) + "\n"


def _ass_stamp(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def write_ass(cs: list, path: Path, height: int) -> Path:
    path = Path(path)
    path.write_text(ass(cs, height), encoding="utf-8")
    return path


def _escape(p: Path) -> str:
    """A path as ffmpeg's filter parser wants to see it."""
    return str(p).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def render_picture(video: Path, sub: Path | None, out: Path,
                   height: int = 720, crf: int = CRF, timeout: int = 3600) -> Path:
    """The picture, scaled and subtitled, with NO sound. Encoded exactly once.

    Both videos carry the identical picture -- the same frames, the same burned
    subtitles -- and the first version of this encoded it twice, which is where
    5m34s of the 5m34s went. Encode once, mux twice: `mux()` stream-copies this,
    so the second video costs seconds instead of minutes, and the two are
    guaranteed to look the same rather than merely intended to.
    """
    video, out = Path(video), Path(out)
    out_h = min(source_height(video) or height, height)
    vf = [f"scale=-2:'min(ih,{height})'"]
    if sub is not None:
        # An .ass file carries its own style block, including the two karaoke
        # colours. Passing force_style as well would overwrite PrimaryColour and
        # SecondaryColour and the words would stop changing colour at all --
        # every \k tag still there, doing nothing visible.
        style = ("" if str(sub).endswith(".ass")
                 else f":force_style='{sub_style(out_h)}'")
        vf.append(f"subtitles=filename='{_escape(Path(sub))}'{style}")
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
         "-vf", ",".join(vf), "-map", "0:v:0", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, text=True, timeout=timeout, env=_env())
    if r.returncode != 0 or not out.is_file() or out.stat().st_size < 10_000:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg did not render the picture: " + " / ".join(tail))
    return out


def mux(picture: Path, audio_from: Path, out: Path, timeout: int = 1800) -> Path:
    """One finished video: the already-encoded picture plus a chosen soundtrack.

    `-c:v copy`, so this is a remux and not a re-encode. Seconds, not minutes.
    """
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-i", str(picture), "-i", str(audio_from),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         # The picture and the chosen audio differ by a few frames; without this
         # the file runs as long as the longer one and ends on a freeze.
         "-shortest", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True, timeout=timeout, env=_env())
    out = Path(out)
    if r.returncode != 0 or not out.is_file() or out.stat().st_size < 10_000:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg did not mux the video: " + " / ".join(tail))
    return out


def burn(video: Path, audio: Path | None, sub: Path | None, out: Path,
         height: int = 720, crf: int = CRF, timeout: int = 3600) -> Path:
    """One finished video: the picture, a chosen audio track, subtitles on top.

    `audio=None` keeps the video's own sound. `sub=None` burns nothing, which is
    what happens when the timings did not survive `alignment()` -- a video with
    no words is honest, a video with words on the wrong beat is not.

    720p is a CEILING, never a target: `min(ih,720)`, so a 360p archive capture
    is not upscaled into a blurrier, slower encode. The first version said
    `scale=-2:720` flatly, and the second video I tested was 640x360 -- one of
    two real archive captures, so half my sample would have been upscaled.
    """
    video, out = Path(video), Path(out)
    out_h = min(source_height(video) or height, height)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video)]
    if audio is not None:
        cmd += ["-i", str(audio)]

    vf = [f"scale=-2:'min(ih,{height})'"]
    if sub is not None:
        # After the scale, so the size is relative to the frame people will
        # actually see rather than to the source.
        vf.append(f"subtitles=filename='{_escape(Path(sub))}'"
                  f":force_style='{sub_style(out_h)}'")
    cmd += ["-vf", ",".join(vf)]

    if audio is not None:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]

    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            # The picture and the chosen audio are different lengths by a few
            # frames; without this the file is as long as the longer one and
            # ends on a freeze.
            "-shortest",
            # Playable while it is still arriving, which is what a phone wants.
            "-movflags", "+faststart", str(out)]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=_env())
    if r.returncode != 0 or not out.is_file() or out.stat().st_size < 10_000:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg did not produce a video: " + " / ".join(tail))
    return out


def subs_landed(plain: Path, subbed: Path, at: float) -> tuple[bool, float, float]:
    """Did the words actually get burned in? Returns (yes, bottom, top).

    THE REASON THIS EXISTS. Ten minutes before it was written, a subtitle burn
    with a broken fontconfig exited 0, wrote every frame, and put a thin grey
    serif on screen that would have vanished over real video. Exit code 0 and a
    file of the right size are exactly what a silent subtitle failure looks
    like.

    So: pull the SAME timestamp out of both files and compare the strip where
    subtitles live. `bottom` should move a lot. `top` is the CONTROL -- it is
    the same picture in both, so if it moved as much as the bottom did, the
    difference is encoder noise and this measurement proves nothing.
    """
    import numpy as np

    def frame(path: Path):
        # `-ss` AFTER `-i`, which decodes to the timestamp instead of seeking to
        # the nearest keyframe. The fast form reads a DIFFERENT frame than the
        # one asked for on h264, and the two files here have different GOP
        # structures, so the fast seek would compare two different moments and
        # the control would blow up along with the signal. This project already
        # lost an afternoon to `cap.set` doing exactly that.
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ss", f"{at:.2f}",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray",
             "-vf", "scale=320:180", "-"],
            capture_output=True, timeout=600, env=_env()).stdout
        a = np.frombuffer(raw, dtype=np.uint8)
        return a[:320 * 180].reshape(180, 320).astype(np.int16) if a.size >= 320 * 180 else None

    a, b = frame(plain), frame(subbed)
    if a is None or b is None:
        return False, 0.0, 0.0
    bottom = float(np.abs(a[130:, :] - b[130:, :]).mean())
    top = float(np.abs(a[:50, :] - b[:50, :]).mean())
    # 4x the control and a real absolute floor. A burn that only just beats the
    # noise is not a burn anybody can read.
    return (bottom > 2.0 and bottom > top * 4), round(bottom, 2), round(top, 2)


WITH_VOICE = "with-voice.mp4"
NO_VOICE = "no-voice.mp4"


def make(out_dir: Path, source: Path, band: Path | None, lyrics,
         timings_usable: bool | None, on_note=None) -> dict:
    """Both videos, and an honest account of what is on them.

    `lyrics` may be None and `timings_usable` may be False; in either case the
    videos are still made, without words. That is the point of the gate.

    THE GATE IS THE WHOLE REASON THIS ORDER EXISTS. Subtitles are where a bad
    clock becomes visible: a `words.json` whose timings are noise is an
    abstraction nobody can see, and the same numbers burned into a picture drift
    out of sync in about two seconds, to anybody, without instruments. So the
    alignment check written an hour before this file stops being a nicety and
    becomes the thing that decides whether words go on the screen at all.

    Measured on the two jobs already on this disk when it was written:
        a talking-head clip   alignment +18.1 dB   ->  subtitles burned
        a sung music video    alignment  -0.1 dB   ->  refused, correctly
    """
    out_dir = Path(out_dir)
    note = on_note or (lambda *_: None)
    result: dict = {"with_voice": None, "no_voice": None, "subtitles": False}

    sub = None
    cs: list[Cue] = []
    if lyrics is not None and getattr(lyrics, "words", None):
        if timings_usable:
            cs = cues(lyrics.words)
            if cs:
                # Word by word, because he asked for a Disney singalong and a
                # whole line appearing at once tells you what to sing without
                # telling you when. The .srt goes out beside it because it is
                # the thing anybody else's player can load.
                sub = write_ass(cs, out_dir / "words.ass", 720)
                write_srt(cs, out_dir / "words.srt")
                result["cues"] = len(cs)
                result["karaoke"] = True
        else:
            # Reports the outcome and stops. An earlier message in this project
            # asserted a CAUSE for a failure -- "some singing genuinely cannot
            # be read" -- and the cause was false; the words were behind a query
            # parameter. A humble wrong explanation is harder to doubt than a
            # confident one.
            result["why_no_subtitles"] = (
                "the words came back, but they do not line up with the singing, "
                "so putting them on screen would drift")

    picture = out_dir / "_picture.mp4"
    try:
        note("burning the subtitles into the picture" if sub is not None
             else "preparing the picture")
        render_picture(source, sub, picture)

        note("adding the original sound")
        with_voice = mux(picture, source, out_dir / WITH_VOICE)
        result["with_voice"] = WITH_VOICE

        if band is not None and Path(band).is_file():
            note("adding the backing track")
            mux(picture, Path(band), out_dir / NO_VOICE)
            result["no_voice"] = NO_VOICE
    finally:
        picture.unlink(missing_ok=True)

    # Did the words actually reach the picture? Exit 0 and a file of the right
    # size are exactly what a silent subtitle failure looks like, and this
    # project has already shipped one instrument that could only ever agree
    # with itself.
    if sub is not None and cs:
        # An EARLY cue on purpose: the comparison decodes to the timestamp
        # rather than seeking, so picking cue 3 rather than one four minutes in
        # is the difference between a second and a minute.
        at = cs[min(2, len(cs) - 1)]
        when = at.start + min(0.3, max(0.05, (at.end - at.start) / 2))
        ok, bottom, top = subs_landed(source, with_voice, when)
        result["subtitles"] = bool(ok)
        result["sub_contrast"] = bottom
        result["sub_control"] = top
        if not ok:
            result["why_no_subtitles"] = (
                "the subtitle track did not reach the picture; the videos are "
                "fine, the words are not on them")
    return result


def source_height(path: Path) -> int | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=120)
    try:
        return int((r.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None
