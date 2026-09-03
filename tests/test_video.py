"""The two videos, and the check that the words reached the picture.

The test that matters most in here is `test_subtitles_actually_reach_the_picture`
and its negative twin. Everything else is arithmetic on timestamps.

Why that one matters: a subtitle burn with a broken fontconfig EXITS 0, writes
every frame, and puts a thin grey serif on screen that vanishes over any real
picture. There is no error, no warning, and no difference in the file size. The
first version of this feature did exactly that on this box and I only found it
by rendering a frame and looking at it. So the suite renders a frame and looks
at it too, every run.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from earshot import video


def W(word, start, end, conf=0.9):
    return SimpleNamespace(word=word, start=start, end=end, confidence=conf)


# --- grouping words into lines ----------------------------------------------

def test_a_pause_starts_a_new_line():
    cs = video.cues([W("hold", 0.0, 0.4), W("on", 0.4, 0.8),
                     W("tight", 5.0, 5.4)])
    assert len(cs) == 2
    assert cs[0].text == "hold on"
    assert cs[1].text == "tight"


def test_a_long_line_is_broken_before_it_runs_off_the_screen():
    ws = [W(f"word{i}", i * 0.3, i * 0.3 + 0.25) for i in range(20)]
    cs = video.cues(ws)
    assert len(cs) > 1
    assert all(len(c.text) <= video.MAX_CHARS for c in cs)


def test_a_line_never_stays_up_longer_than_max_seconds():
    # Words close enough together that only the duration rule can split them.
    ws = [W("la", i * 0.5, i * 0.5 + 0.45) for i in range(40)]
    cs = video.cues(ws, max_chars=10_000)
    assert all(c.end - c.start <= video.MAX_SECONDS + 0.6 for c in cs)


def test_cues_never_overlap():
    # Deepgram does hand back words whose spans touch or cross; two cues on
    # screen at once reads as a glitch rather than as timing.
    ws = [W("a", 0.0, 2.0), W("b", 1.5, 3.0), W("c", 2.9, 4.0)]
    cs = video.cues(ws, max_chars=1)
    for x, y in zip(cs, cs[1:]):
        assert x.end <= y.start


def test_empty_words_make_no_cues():
    assert video.cues([]) == []


# --- the srt itself ----------------------------------------------------------

def test_srt_timestamps_are_hours_minutes_seconds_milliseconds():
    out = video.srt([video.Cue(61.25, 63.5, "hello")])
    assert "00:01:01,250 --> 00:01:03,500" in out
    assert out.strip().endswith("hello")


def test_subtitle_size_scales_with_the_frame():
    # ASS sizes are in script units and the filter takes the video resolution as
    # the script resolution, so a fixed size is twice as big at 360p as at 720p.
    small = video.sub_style(360)
    large = video.sub_style(720)
    assert "Fontsize=22" in small
    assert "Fontsize=45" in large


# --- the part that needs ffmpeg ---------------------------------------------

needs_ffmpeg = pytest.mark.skipif(not video.have_ffmpeg(), reason="no ffmpeg")


def _clip(path: Path, seconds: float = 3.0) -> Path:
    """A few seconds of moving picture, so the comparison is not against a
    still frame that would agree with itself."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size=320x180:rate=10:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(path)], check=True, timeout=300)
    return path


@needs_ffmpeg
def test_subtitles_actually_reach_the_picture(tmp_path):
    """The control for the whole feature. If fonts break, this goes red."""
    src = _clip(tmp_path / "src.mp4")
    sub = video.write_srt([video.Cue(0.2, 2.8, "wanted dead or alive")],
                          tmp_path / "w.srt")
    out = video.render_picture(src, sub, tmp_path / "subbed.mp4")

    ok, bottom, top = video.subs_landed(src, out, at=1.0)
    assert ok, f"subtitles did not reach the picture (bottom={bottom} top={top})"
    # Not just "different": different in the STRIP WHERE SUBTITLES LIVE, and
    # not different in the strip where they do not. Without the second half
    # this passes on any re-encode at all.
    assert bottom > top * 4


@needs_ffmpeg
def test_the_check_says_no_when_nothing_was_burned(tmp_path):
    """The same measurement, on a video with no subtitles on it.

    A check that has only ever been shown to say yes is not a check. This is
    the case that has to fail, and it is the one that catches the version of
    `subs_landed` that answers 'yes' to encoder noise.
    """
    src = _clip(tmp_path / "src.mp4")
    plain = video.render_picture(src, None, tmp_path / "plain.mp4")

    ok, bottom, top = video.subs_landed(src, plain, at=1.0)
    assert not ok, f"claimed subtitles on a video with none (bottom={bottom})"


@needs_ffmpeg
def test_the_two_videos_carry_different_sound(tmp_path):
    """One with the voice, one without, and they must not be the same file.

    `no-voice` is the whole point of the exercise, and "the file exists" is not
    evidence that the voice left it.
    """
    src = _clip(tmp_path / "src.mp4", seconds=3.0)
    # A backing track that is obviously not the source's silence.
    band = tmp_path / "band.mp3"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=3", "-c:a", "libmp3lame",
         str(band)], check=True, timeout=300)
    # The source needs a soundtrack of its own to be muxed into with-voice.
    with_audio = tmp_path / "src_a.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-f", "lavfi", "-i", "sine=frequency=120:duration=3",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)],
        check=True, timeout=300)

    r = video.make(tmp_path, with_audio, band, None, None)
    assert r["with_voice"] and r["no_voice"]

    def tone(p):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", "8000",
             "-f", "f32le", "-"], capture_output=True, timeout=300).stdout
        import numpy as np
        a = np.frombuffer(raw, dtype=np.float32)
        spec = np.abs(np.fft.rfft(a[:8000] * np.hanning(min(8000, len(a)))))
        return float(np.fft.rfftfreq(min(8000, len(a)), 1 / 8000)[spec.argmax()])

    assert 100 < tone(tmp_path / video.WITH_VOICE) < 145, "with-voice lost its own sound"
    assert 420 < tone(tmp_path / video.NO_VOICE) < 465, "no-voice did not get the backing"


@needs_ffmpeg
def test_bad_timings_mean_no_subtitles_and_a_reason(tmp_path):
    """The gate. Words exist, the clock is wrong, so nothing goes on screen.

    Subtitles are where a bad clock becomes visible to anybody in two seconds,
    which is exactly why this refusal has to hold.
    """
    src = _clip(tmp_path / "src.mp4")
    with_audio = tmp_path / "src_a.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-f", "lavfi", "-i", "sine=frequency=120:duration=3",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)],
        check=True, timeout=300)

    lyrics = SimpleNamespace(words=[W("these", 0.1, 0.5), W("drift", 0.6, 1.2)])
    r = video.make(tmp_path, with_audio, None, lyrics, timings_usable=False)

    assert r["subtitles"] is False
    assert r["why_no_subtitles"]
    assert not (tmp_path / "words.srt").exists(), "wrote an srt it refused to use"
    # And the video still got made. A refusal is not a failure.
    assert (tmp_path / video.WITH_VOICE).is_file()


@needs_ffmpeg
def test_the_scratch_picture_is_not_left_behind(tmp_path):
    src = _clip(tmp_path / "src.mp4")
    with_audio = tmp_path / "src_a.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-f", "lavfi", "-i", "sine=frequency=120:duration=3",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)],
        check=True, timeout=300)
    video.make(tmp_path, with_audio, None, None, None)
    assert not (tmp_path / "_picture.mp4").exists()
