"""A public page: paste a link, get told whether the dialogue is audible.

Everything in this file that is not the form is a limit, because this is an
open endpoint on somebody's machine that fetches URLs and burns CPU:

  - https only, and the resolved address must be public. A service that fetches
    what it is told to fetch is a way to read the inside of the network it runs
    in; every address is resolved and checked BEFORE the request, and the
    connection is made to the address that was checked.
  - one job at a time, so the box cannot be made to run twenty decodes.
  - one request per address per minute, and a whole-day ceiling.
  - hard caps on download size and on programme length.

Kill it with `systemctl --user stop earshot`. Nothing else depends on it.
"""
from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import mkdtemp

from . import decode, measure, report, studio

MAX_MB = 80
MAX_MINUTES = 30.0
PER_IP_COOLDOWN = 60.0
DAILY_JOBS = 300
PORT = 8781

_job = threading.Lock()
_seen: dict[str, float] = {}
_day = [0, 0.0]          # [count, day-start epoch]


class Refused(Exception):
    pass


def _public_address(host: str) -> str:
    """Resolve, and refuse anything that is not a public address.

    Returns the address actually checked, so the fetch can connect to THAT and
    not re-resolve into somewhere else between the check and the connection.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise Refused(f"cannot resolve {host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            raise Refused("that address is not on the public internet")
    return infos[0][4][0]


def fetch(url: str, into: Path) -> Path:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise Refused("https links only")
    if not parts.hostname:
        raise Refused("that is not a link")
    _public_address(parts.hostname)

    name = Path(parts.path).name or "media"
    dest = into / name[:80]
    req = urllib.request.Request(url, headers={"User-Agent": "earshot/0.1 (+https://earshot.soulful-ai.dev)"})
    try:
        opened = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        # The far end said no. That is the visitor's problem to fix and they
        # need to be told which one it is, not handed "that broke on my side".
        raise Refused(f"that link returned {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise Refused(f"could not reach that link: {e.reason}") from e
    with opened as r:
        length = r.headers.get("Content-Length")
        if length and int(length) > MAX_MB << 20:
            raise Refused(f"that file is bigger than {MAX_MB} MB")
        got = 0
        with dest.open("wb") as f:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                got += len(buf)
                if got > MAX_MB << 20:
                    raise Refused(f"stopped at {MAX_MB} MB; that file is too big")
                f.write(buf)
    return dest


# Downloads go to disk, not to /tmp. On this box /tmp is a 921 MB tmpfs -- it
# is RAM -- and an open endpoint that writes what strangers hand it into RAM is
# a way to take the machine down with a large file. Found the hard way: a
# leftover 452 MB render had been sitting in there since 2026-08-30 eating half
# of it, and the fixtures filled the rest.
WORK = Path.home() / "ventures" / "earshot" / "work"


def analyse_url(url: str) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = Path(mkdtemp(prefix="earshot-web-", dir=WORK))
    try:
        path = fetch(url, tmp)
        media = decode.probe(path)
        if media.duration > MAX_MINUTES * 60:
            raise Refused(f"that is {media.duration/60:.0f} minutes; "
                          f"the limit here is {MAX_MINUTES:.0f}")
        r = measure.analyse(path, name=url)
        return {"ok": True, "report": r.as_dict(), "text": report.text(r),
                "strip": report.strip(r)}
    finally:
        for p in sorted(tmp.rglob("*"), reverse=True):
            p.unlink(missing_ok=True) if p.is_file() else p.rmdir()
        tmp.rmdir()


def parse_multipart(body: bytes, content_type: str) -> tuple[dict, bytes, str]:
    """One file plus some plain fields. Enough for this form and nothing more.

    Written out rather than using `cgi.FieldStorage`, which is deprecated in
    3.12 and gone in 3.13 -- a dependency with a removal date already published
    is not one to build a new thing on.
    """
    marker = "boundary="
    if marker not in content_type:
        raise studio.Refused("that upload was malformed")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    sep = b"--" + boundary.encode()
    fields: dict[str, str] = {}
    payload, filename = b"", ""
    for part in body.split(sep):
        if b"\r\n\r\n" not in part:
            continue
        head, value = part.split(b"\r\n\r\n", 1)
        value = value.rstrip(b"\r\n-")
        headers = head.decode("utf8", "replace")
        if 'name="' not in headers:
            continue
        name = headers.split('name="', 1)[1].split('"', 1)[0]
        if 'filename="' in headers:
            filename = headers.split('filename="', 1)[1].split('"', 1)[0]
            payload = value
        else:
            fields[name] = fields.get(name, "")
            fields[name] = (fields[name] + "," if fields[name] else "") + \
                value.decode("utf8", "replace")
    return fields, payload, filename


UPLOAD_CHUNK = 1 << 20


def read_multipart_to_disk(rfile, content_type: str, length: int,
                           dest: Path, max_bytes: int) -> tuple[dict, int, str]:
    """Stream one uploaded file straight to `dest`, keeping the fields.

    `parse_multipart` above takes the whole body as bytes and is fine for the
    small forms it was written for. It is the wrong shape for a file: reading
    the body costs one copy, `body.split(sep)` costs another, and slicing the
    payload out costs a third.

    Measured on this box, 2026-09-02: a 24 MB upload took the server from 76 MB
    RSS to 159 MB. **83 MB of memory for a 24 MB file, ~3.5x.** The unit caps
    the service at 700M and `ThreadingHTTPServer` serves uploads concurrently,
    so the multiplier is what pins MAX_MB at 25 rather than any property of the
    audio. It was not failing. It was the reason the limit could not rise.

    `fetch()` six functions above already does this correctly for URLs -- 1 MB
    chunks, straight to disk, cap enforced *during* the read. The upload path
    did the opposite in the same file, which is
    fix-the-sibling-not-just-the-call-site with both siblings visible on one
    screen.

    Returns (fields, bytes_written, filename). `bytes_written` is 0 and
    `filename` is "" when the form carried no file part.

    The cap is enforced as the bytes arrive, so an oversized upload is refused
    partway through rather than after it has all been accepted.
    """
    marker = "boundary="
    if marker not in content_type:
        raise studio.Refused("that upload was malformed")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    sep = b"--" + boundary.encode()

    fields: dict[str, str] = {}
    filename = ""
    written = 0
    remaining = length
    buf = b""
    out = None

    def read_more() -> bool:
        """Pull one chunk off the socket into `buf`. False at end of body."""
        nonlocal buf, remaining
        if remaining <= 0:
            return False
        chunk = rfile.read(min(UPLOAD_CHUNK, remaining))
        if not chunk:
            remaining = 0
            return False
        remaining -= len(chunk)
        buf += chunk
        return True

    def take_header() -> str | None:
        """Consume up to and including the blank line ending a part's header."""
        nonlocal buf
        while b"\r\n\r\n" not in buf:
            if not read_more():
                return None
        head, buf = buf.split(b"\r\n\r\n", 1)
        return head.decode("utf8", "replace")

    try:
        # Skip the preamble up to the first boundary.
        while sep not in buf:
            if not read_more():
                raise studio.Refused("that upload was malformed")
        buf = buf.split(sep, 1)[1]

        while True:
            if buf[:2] == b"--":               # closing boundary; body is done
                break
            head = take_header()
            if head is None:
                break
            name = (head.split('name="', 1)[1].split('"', 1)[0]
                    if 'name="' in head else "")
            is_file = 'filename="' in head
            if is_file:
                filename = head.split('filename="', 1)[1].split('"', 1)[0]

            if is_file and out is None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                out = dest.open("wb")

            # Read this part's body until the next boundary. A boundary can
            # land across a chunk edge, so always hold back enough bytes that a
            # split one is still whole next time round. Without this the file
            # is written correctly for every size that happens not to straddle
            # a 1 MB line, which is most of them -- the bug would pass every
            # small fixture and corrupt real songs.
            hold = len(sep) + 4
            value = b""
            while True:
                idx = buf.find(sep)
                if idx != -1:
                    piece, buf = buf[:idx], buf[idx + len(sep):]
                    piece = piece[:-2] if piece.endswith(b"\r\n") else piece
                    if is_file and out is not None:
                        written += len(piece)
                        if written > max_bytes:
                            raise studio.Refused(
                                f"stopped at {max_bytes >> 20} MB; that file is too big")
                        out.write(piece)
                    else:
                        value += piece
                    break
                if len(buf) > hold:
                    piece, buf = buf[:-hold], buf[-hold:]
                    if is_file and out is not None:
                        written += len(piece)
                        if written > max_bytes:
                            raise studio.Refused(
                                f"stopped at {max_bytes >> 20} MB; that file is too big")
                        out.write(piece)
                    else:
                        value += piece
                if not read_more():
                    # Truncated body. Whatever is left belongs to this part.
                    if is_file and out is not None:
                        out.write(buf)
                        written += len(buf)
                    else:
                        value += buf
                    buf = b"--"
                    break

            if not is_file and name:
                prev = fields.get(name, "")
                fields[name] = ((prev + "," if prev else "")
                                + value.decode("utf8", "replace"))
    finally:
        if out is not None:
            out.close()
        # Never leave the rest of the request in the socket: under keep-alive
        # the next request gets parsed starting mid-payload and surfaces as a
        # mystery 400 on something unrelated. Already learned once, on the talk
        # door, in living-on-the-box.
        while remaining > 0:
            chunk = rfile.read(min(UPLOAD_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    return fields, written, filename


PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>earshot &mdash; is the dialogue actually audible?</title>
<style>
 :root{color-scheme:dark}
 body{background:#0d0f12;color:#e7e3dc;font:16px/1.55 ui-sans-serif,system-ui,sans-serif;
      margin:0;padding:2.2rem 1.2rem;display:flex;justify-content:center}
 main{width:100%;max-width:44rem}
 h1{font-size:1.5rem;margin:0 0 .2rem;letter-spacing:-.01em}
 .sub{color:#9aa0a6;margin:0 0 1.6rem}
 form{display:flex;gap:.5rem;flex-wrap:wrap}
 input{flex:1 1 20rem;background:#16191e;border:1px solid #2b3038;border-radius:.5rem;
       color:inherit;padding:.7rem .8rem;font:inherit}
 button{background:#c8a24a;color:#15120a;border:0;border-radius:.5rem;
        padding:.7rem 1.2rem;font:600 inherit;cursor:pointer}
 button[disabled]{opacity:.5;cursor:default}
 pre{background:#14171c;border:1px solid #23272e;border-radius:.6rem;padding:1rem;
     overflow-x:auto;font:13px/1.5 ui-monospace,monospace;white-space:pre-wrap;margin-top:1.4rem}
 .why{color:#9aa0a6;font-size:.92rem;margin-top:2rem;border-top:1px solid #23272e;padding-top:1.2rem}
 .why b{color:#e7e3dc;font-weight:600}
 a{color:#c8a24a}
</style>
<main>
<h1>earshot</h1>
<p class=sub>Is the dialogue actually audible, or is it you?</p>
<form id=f>
  <input id=u type=url required placeholder="https://&hellip; link to an audio or video file">
  <button id=go>Listen</button>
</form>
<pre id=out hidden></pre>
<div class=why>
<p><b>78%</b> of people say background music makes dialogue hard to hear, and about
half of all viewers now watch with subtitles on. Nearly three in four of them
name the mix rather than their own hearing.</p>
<p>This measures how far the talking sits above everything under it, using the
same gated loudness the broadcast standards use, and tells you where in the
runtime it goes wrong. Direct links to a media file, please &mdash; not a page
that contains one. Under {MAX_MB} MB and under {MAX_MINUTES} minutes.</p>
<p>It is honest about what it cannot do: where the voice and the bed are within
a decibel of each other it says so instead of printing a number.</p>
</div>
</main>
<script>
const f=document.getElementById('f'),u=document.getElementById('u'),
      out=document.getElementById('out'),go=document.getElementById('go');
f.onsubmit=async e=>{
  e.preventDefault(); go.disabled=true; out.hidden=false;
  out.textContent='listening\\u2026 a long programme takes a minute';
  try{
    const r=await fetch('/analyse',{method:'POST',headers:{'content-type':'application/json'},
                                    body:JSON.stringify({url:u.value})});
    const d=await r.json();
    out.textContent = d.ok ? d.text : (d.why || 'that did not work');
  }catch(err){ out.textContent='that did not work: '+err; }
  go.disabled=false;
};
</script>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "earshot"

    def log_message(self, fmt, *args):        # one line, no user agent noise
        print(f"{self.client_address[0]} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _who(self) -> str:
        return self.headers.get("CF-Connecting-IP") or self.client_address[0]

    def _studio_page(self) -> bytes:
        html = (Path(__file__).resolve().parent / "studio.html").read_text()
        return (html.replace("__MAX_MB__", str(studio.MAX_MB))
                    .replace("__MAX_MIN__", str(int(studio.MAX_MINUTES)))
                    .replace("__KEEP__", str(studio.KEEP_HOURS))).encode()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, self._studio_page(), "text/html; charset=utf-8")
            return
        if self.path == "/recent":
            self._json(200, {"jobs": studio.recent()})
            return
        if self.path == "/parts":
            from . import separate
            names = separate.available(separate.SIX)
            self._json(200, {"parts": [{"name": n, "what": separate.PARTS[n]}
                                       for n in names]})
            return
        if self.path.startswith("/jobs/"):
            rest = self.path[len("/jobs/"):].split("?")[0].split("/")
            jid = rest[0]
            if not jid.isalnum() or len(jid) > 32:
                self._send(404, b"no", "text/plain")
                return
            d = studio.JOBS / jid
            if not d.is_dir():
                self._json(404, {"state": "unknown"})
                return
            if len(rest) == 1:
                self._json(200, studio.describe(studio.Job(jid, d)))
                return
            # A stem download. The name is taken from our own status file, never
            # from the URL, so no path from outside chooses what gets served.
            wanted = rest[1]
            st = studio.Job(jid, d).status
            allowed = set((st.get("parts") or {}).values())
            # The two videos, same rule as the stems: the name comes from our
            # own status file, never from the URL.
            for key in ("with_voice", "no_voice"):
                name = (st.get("videos") or {}).get(key)
                if name:
                    allowed.add(name)
            # words.json only becomes servable once the status says it was
            # written. Adding it as a blanket exception would let any job id
            # fish for a file the worker never made.
            if (st.get("lyrics") or {}).get("file"):
                allowed.add(st["lyrics"]["file"])
            if wanted not in allowed:
                self._send(404, b"no", "text/plain")
                return
            f = d / "out" / wanted
            if not f.is_file():
                # Swept locally, but the job was kept in S3. This is what makes
                # a link in an email still work tomorrow, and it is why the
                # email points at this site rather than at a presigned URL.
                if studio.fetch_s3(jid, wanted, f) is None:
                    self._send(404, b"no", "text/plain")
                    return
            # Type by suffix, not a hardcoded audio/mpeg. A video served as
            # audio/mpeg does not play in a <video> tag -- it downloads, or it
            # shows a broken player, which looks exactly like the encode having
            # failed. And `attachment` on a video means a phone can never
            # simply press play, so mp4 is served inline.
            ctype, disp = "audio/mpeg", "attachment"
            if wanted.endswith(".mp4"):
                ctype, disp = "video/mp4", "inline"
            elif wanted.endswith(".json"):
                ctype = "application/json"
            elif wanted.endswith(".srt"):
                ctype = "text/plain; charset=utf-8"

            # STREAMED, and ranges are real rather than advertised.
            #
            # `f.read_bytes()` was fine for a 5 MB stem and is not fine for a
            # 90 MB video on a 1.8 GB box serving requests concurrently. This
            # file already carries the measurement that motivated streaming the
            # UPLOAD path -- 83 MB of RSS for a 24 MB file -- and the download
            # path was still reading whole files in. Both siblings, one screen.
            #
            # And `Accept-Ranges: bytes` is a promise. A phone scrubbing a video
            # sends `Range:`, and answering 200-with-everything to a range
            # request is how a seek turns into a 90 MB refetch.
            size = f.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range", "")
            partial = False
            if rng.startswith("bytes=") and "," not in rng:
                a, _, b = rng[6:].partition("-")
                try:
                    if a:
                        start = int(a)
                        end = int(b) if b else size - 1
                    elif b:                      # bytes=-500, the last 500
                        start = max(0, size - int(b))
                    partial = 0 <= start <= end < size
                except ValueError:
                    partial = False
                if not partial:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Disposition",
                             f'{disp}; filename="{wanted}"')
            self.end_headers()
            remaining = end - start + 1
            with f.open("rb") as fh:
                fh.seek(start)
                while remaining > 0:
                    chunk = fh.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return
        if self.path in ("/measure", "/measure/"):
            # str.replace, not %-formatting: the stylesheet contains `100%`
            # and `%` formatting choked on it, so the page raised before any
            # header was sent and the tunnel returned a bare 502. /health was
            # fine the whole time, which is why I did not notice -- I tested the
            # endpoints and never opened the page.
            body = (PAGE.replace("{MAX_MB}", str(MAX_MB))
                        .replace("{MAX_MINUTES}", str(int(MAX_MINUTES))))
            self._send(200, body.encode(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200, {"ok": True, "busy": _job.locked(), "today": _day[0]})
        else:
            self._send(404, b"no", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)

        # --- a link instead of a file ---------------------------------------
        # This box cannot reach YouTube (measured: seven yt-dlp player clients,
        # two videos, all "Sign in to confirm you're not a bot"). The Wayback
        # Machine has archived media for a lot of videos and yt-dlp has a
        # dedicated extractor for it, which is the route Bruno pointed at.
        #
        # PROBE is synchronous because it takes two seconds and answers "is this
        # even available", which somebody deserves before they wait. The
        # DOWNLOAD is not, because it is minutes, so it runs behind a status the
        # page already knows how to poll.
        if self.path == "/jobs/from-url":
            from . import fetch as ytfetch
            raw = self.rfile.read(min(length, 4096))
            try:
                url = (json.loads(raw or b"{}").get("url") or "").strip()
            except json.JSONDecodeError:
                url = ""
            try:
                found = ytfetch.probe(url)
            except ValueError as e:
                self._json(400, {"ok": False, "why": str(e)})
                return
            except ytfetch.NotArchived as e:
                self._json(404, {"ok": False, "why": str(e)})
                return
            except Exception:                                     # noqa: BLE001
                traceback.print_exc()
                self._json(502, {"ok": False,
                                 "why": "could not reach the archive just now"})
                return

            if found.duration and found.duration > studio.MAX_MINUTES * 60:
                self._json(400, {"ok": False,
                                 "why": f"that is {found.duration/60:.0f} minutes; "
                                        f"the limit here is {studio.MAX_MINUTES:.0f}"})
                return

            jid, staged = studio.staging_path("audio.mp3")
            studio.write_status(jid, state="fetching", title=found.title,
                                source=found.source, duration=found.duration)

            def pull():
                try:
                    # The PICTURE, not just the sound. These archive URLs offer
                    # one progressive format, so the old audio-only call was
                    # downloading exactly these bytes and discarding the video
                    # half. The audio is then extracted locally rather than
                    # fetched twice.
                    vid = None
                    try:
                        vid = ytfetch.grab_video(url, staged.parent)
                        audio = staged.parent / "audio.mp3"
                        subprocess.run(
                            ["ffmpeg", "-nostdin", "-v", "error", "-y",
                             "-i", str(vid), "-vn", "-c:a", "libmp3lame",
                             "-b:a", "192k", str(audio)],
                            check=True, timeout=1800)
                        got = audio
                    except Exception as e:                        # noqa: BLE001
                        # Video is the upgrade, stems are the floor. If the
                        # capture has no usable picture, still give somebody
                        # their stems rather than nothing at all.
                        traceback.print_exc()
                        print(f"job {jid}: no video ({e}); audio only", flush=True)
                        vid = None
                        got = ytfetch.grab(url, staged.parent)

                    # Passed IN, never patched in afterwards: adopt() queues the
                    # job on its last line, so writing job.json after it returns
                    # races the worker reading it.
                    studio.adopt(jid, got, f"{found.title}.mp3", [],
                                 __import__("earshot.separate",
                                            fromlist=["x"]).SIX,
                                 extra={"video": vid.name} if vid else None)
                except Exception as e:                            # noqa: BLE001
                    traceback.print_exc()
                    studio.write_status(jid, state="failed",
                                        why=f"could not fetch: {e}"[:300])

            threading.Thread(target=pull, daemon=True).start()
            self._json(200, {"ok": True, "id": jid, "title": found.title})
            return

        if self.path == "/jobs":
            if length > (studio.MAX_MB + 2) << 20:
                # Refuse before reading. Draining 200 MB into memory to then
                # say no is how you get taken down politely.
                self._json(413, {"ok": False,
                                 "why": f"that file is over {studio.MAX_MB} MB"})
                return
            # The filename is only needed for its suffix, and it arrives inside
            # the body we have not read yet, so stage under a neutral one and
            # let `adopt` see the real name.
            jid, staged = studio.staging_path("upload.mp3")
            adopted = False
            try:
                fields, written, filename = read_multipart_to_disk(
                    self.rfile, self.headers.get("Content-Type", ""), length,
                    staged, studio.MAX_MB << 20)
                if not written:
                    raise studio.Refused("no file arrived")
                # Keep the real extension: decode.probe sniffs the container,
                # but ffmpeg does better with a truthful suffix.
                suffix = Path(filename or "").suffix.lower()[:6]
                if suffix and suffix != staged.suffix:
                    staged = staged.rename(staged.with_suffix(suffix))
                parts = [p for p in (fields.get("part", "")).split(",") if p]
                job = studio.adopt(jid, staged, filename or "upload.mp3", parts,
                                   __import__("earshot.separate", fromlist=["x"]).SIX)
                adopted = True
                self._json(200, {"ok": True, "id": job.id})
            except studio.Refused as e:
                self._json(400, {"ok": False, "why": str(e)})
            except Exception:                                 # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"ok": False, "why": "that broke on my side, sorry"})
            finally:
                # Staging happens BEFORE the bytes arrive, so every path that
                # does not reach a job -- malformed body, over the cap, no file
                # part, a raise in the middle -- has already made a directory.
                # Left behind, it has no job.json and reads as a job stuck in
                # "queued" for ever. `adopt` clears up after its own failures;
                # this clears up after everything before it.
                if not adopted:
                    studio.abandon(jid)
            return

        raw = self.rfile.read(min(length, 4096))
        if self.path != "/analyse":
            self._send(404, b"no", "text/plain")
            return

        who = self._who()
        now = time.time()
        if now - _day[1] > 86400:
            _day[0], _day[1] = 0, now
        if _day[0] >= DAILY_JOBS:
            self._json(429, {"ok": False, "why": "that is enough for one day; try tomorrow"})
            return
        if now - _seen.get(who, 0.0) < PER_IP_COOLDOWN:
            wait = int(PER_IP_COOLDOWN - (now - _seen[who]))
            self._json(429, {"ok": False, "why": f"one at a time, please. {wait}s"})
            return
        if not _job.acquire(blocking=False):
            self._json(429, {"ok": False,
                             "why": "something else is being measured right now; "
                                    "give it a minute"})
            return
        try:
            _seen[who] = now
            _day[0] += 1
            url = (json.loads(raw or b"{}").get("url") or "").strip()
            if not url:
                self._json(400, {"ok": False, "why": "no link"})
                return
            self._json(200, analyse_url(url))
        except Refused as e:
            self._json(400, {"ok": False, "why": str(e)})
        except decode.DecodeError:
            # ffmpeg's message names the temp path it was handed, which is
            # nobody's business and tells the visitor nothing.
            self._json(400, {"ok": False,
                             "why": "there is no audio track in that, or the "
                                    "format is one ffmpeg does not read"})
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"ok": False, "why": "that broke on my side, sorry"})
        finally:
            _job.release()


def _sweeper() -> None:
    """Old jobs go on a timer. Written as a loop rather than left to a cron I
    would have to remember: somebody's music should not sit on this machine."""
    while True:
        time.sleep(1800)
        try:
            gone = studio.sweep()
            if gone:
                print(f"swept {gone} finished job(s)", flush=True)
        except Exception:                                     # noqa: BLE001
            traceback.print_exc()


def main() -> None:
    threading.Thread(target=_sweeper, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"earshot on 127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
