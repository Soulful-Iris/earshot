"""Get the audio for a YouTube link, without YouTube.

This box cannot talk to YouTube. Measured 2026-09-02: seven different yt-dlp
player clients, two different videos, all answered "Sign in to confirm you're
not a bot". Datacenter IPs are blocked. archive.org through the same tool worked
fine, and plain https to youtube.com returns 200, so it is not the network and
not the tool.

Bruno read that and said "make it work via archive.org then", which I was about
to explain could not work. It can. The Wayback Machine has archived the media
for a lot of videos, and yt-dlp has a dedicated `web.archive:youtube` extractor
for exactly this. The control I ran to prove the wall was real turned out to be
the door.

WHAT THIS IS HONEST ABOUT. Coverage is partial. Measured on six well-known
videos, five had real media and one was not archived at all, and it will be
thinner for anything recent or obscure. The copy is whatever the crawler grabbed
at the time, so it can be an old capture at modest quality. `probe()` reports
what is actually there before anything is downloaded, so a person finds out
before they wait.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

YT_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/)|youtu\.be/)([\w-]{11})")
WAYBACK = "https://web.archive.org/web/2/https://www.youtube.com/watch?v={vid}"
PY = sys.executable


class NotArchived(Exception):
    """The Wayback Machine has no media for this video. Says so plainly."""


@dataclass
class Found:
    video_id: str
    title: str
    duration: float | None
    ext: str
    filesize: int | None
    source: str = "web.archive.org"

    @property
    def duration_text(self) -> str:
        if not self.duration:
            return "unknown length"
        m, s = divmod(int(self.duration), 60)
        return f"{m}:{s:02d}"


def video_id(url: str) -> str:
    """Pull the 11-character id out of any shape of YouTube URL."""
    m = YT_ID.search(url or "")
    if not m:
        # A bare id is allowed, because people paste those too.
        if re.fullmatch(r"[\w-]{11}", (url or "").strip()):
            return url.strip()
        raise ValueError(f"that does not look like a YouTube link: {url!r}")
    return m.group(1)


def _ytdlp(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", "yt_dlp", "--no-warnings", *args],
                          capture_output=True, text=True, timeout=timeout)


def probe(url: str, timeout: int = 120) -> Found:
    """What the archive actually has, BEFORE downloading anything.

    Separate from `grab` on purpose. A person asking for a song should find out
    in two seconds that it is not archived, rather than after a minute of
    waiting, and should be told what quality is on offer rather than discovering
    it in the result.
    """
    vid = video_id(url)
    r = _ytdlp(["--skip-download", "-J", WAYBACK.format(vid=vid)], timeout)
    if r.returncode != 0 or not r.stdout.strip():
        err = (r.stderr or "").strip().splitlines()
        msg = err[-1] if err else "unknown error"
        if "not archived" in msg.lower():
            raise NotArchived(
                f"the Wayback Machine has no copy of {vid}. That happens - it has "
                f"only archived some videos, and more of the old and famous ones "
                f"than the recent ones.")
        raise NotArchived(f"could not read {vid} from the archive: {msg}")

    d = json.loads(r.stdout)
    best = None
    for f in d.get("formats") or []:
        if f.get("acodec") in (None, "none"):
            continue
        if best is None or (f.get("abr") or 0) > (best.get("abr") or 0):
            best = f
    return Found(
        video_id=vid,
        title=d.get("title") or vid,
        duration=d.get("duration"),
        ext=(best or {}).get("ext") or d.get("ext") or "mp4",
        filesize=(best or {}).get("filesize") or d.get("filesize_approx"),
    )


# NOT a format id. The first version of grab_video said `-f 37`, because that is
# what the probe reported for the video I tested it on. The second real video
# was fmt 18 (640x360) and it failed outright with "Requested format is not
# available" -- one for one, on a sample of two. Archive captures carry whatever
# the crawler happened to take, so ask for the property, never the number.
VIDEO_FORMAT = "best[vcodec!=none][acodec!=none]/best"


def grab_video(url: str, into: Path, timeout: int = 1800) -> Path:
    """Download the picture as well as the sound.

    This costs NOTHING extra and that is worth writing down. These archive URLs
    offer a single progressive format, so `-f bestaudio/best -x` in `grab` was
    already pulling the whole video down and then throwing the picture away.
    Keeping it is the same bytes.
    """
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    vid = video_id(url)
    out = into / f"{vid}.video.%(ext)s"
    r = _ytdlp(["-f", VIDEO_FORMAT, "-o", str(out), WAYBACK.format(vid=vid)],
               timeout)
    got = [p for p in sorted(into.glob(f"{vid}.video.*"))
           if p.suffix in (".mp4", ".webm", ".mkv")]
    if not got:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        raise NotArchived("no video downloaded: " + (err[-1] if err else "no output"))
    return got[0]


def grab(url: str, into: Path, seconds: float | None = None,
         timeout: int = 900) -> Path:
    """Download the audio to `into`, optionally only the first `seconds`.

    The native downloader, not ffmpeg. ffmpeg came back `Input/output error,
    exit 251` on the same URL the native downloader handled fine, and if I had
    stopped at that I would have reported the whole archive route as broken.
    When two downloaders disagree, the one that produces bytes is right.
    """
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    vid = video_id(url)
    out = into / f"{vid}.%(ext)s"

    args = ["-f", "bestaudio/best", "-x", "--audio-format", "mp3",
            "-o", str(out), WAYBACK.format(vid=vid)]

    # NOT `--download-sections`. That silently switches yt-dlp to ffmpeg as the
    # downloader, which is the thing that fails on these archive URLs with
    # `Input/output error, exit 251` - the exact failure the docstring above
    # warns about, which I then caused three lines later on the first real run.
    # Download whole with the native downloader, trim afterwards, locally.
    r = _ytdlp(args, timeout)
    got = sorted(into.glob(f"{vid}.*"))
    got = [p for p in got if p.suffix in (".mp3", ".m4a", ".webm", ".opus")]
    if not got:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        raise NotArchived("nothing downloaded: " + (err[-1] if err else "no output"))
    full = got[0]

    if seconds:
        clip = full.with_name(f"{full.stem}.first{int(seconds)}s.mp3")
        t = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(full),
             "-t", str(seconds), "-c:a", "libmp3lame", "-b:a", "192k", str(clip)],
            capture_output=True, text=True, timeout=300)
        if t.returncode == 0 and clip.exists() and clip.stat().st_size > 1024:
            return clip
        # Trimming is a convenience. If it fails, the whole file is still the
        # answer, and saying "could not trim" beats returning nothing.
        print(f"  (could not trim, using the whole file: "
              f"{(t.stderr or '').strip().splitlines()[-1:] or 'no error'})")
    return full
