"""Upload a song, tick what you want out of it, get the files.

Bruno's spec, and it is the whole product:

> *"I should be able to import an mp3. You should be able to give me options of
> vocals only / background noise or band / or anything in specific you can
> detect out."*

Everything here that is not the form is a limit, because separation is heavy and
this box is small:

  - one job at a time, and the rest queue. Two concurrent jobs is 2.8 GB on a
    1.8 GB machine, so the queue is not politeness, it is the thing standing
    between a second upload and an out-of-memory kill.
  - the worker runs OUT OF PROCESS, under a memory cap where systemd allows it,
    so a job that goes wrong dies alone.
  - a cap on file size and on duration, both stated on the page rather than
    discovered.
  - jobs are swept after a few hours. Nobody's music should sit on somebody
    else's machine indefinitely.

Progress is real, not a spinner: the worker writes how many seconds it has
finished, and the page shows that against the total. At two to four times
realtime, a spinner would be a lie for ten minutes at a stretch.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

MAX_MB = 25
MAX_MINUTES = 6.0
KEEP_HOURS = 6
JOBS = Path.home() / "ventures" / "earshot" / "work" / "jobs"
WORKER_MEMORY_MAX = "1500M"

_lock = threading.Lock()
_queue: list[str] = []
_current: str | None = None


class Refused(Exception):
    pass


@dataclass
class Job:
    id: str
    dir: Path

    @property
    def status(self) -> dict:
        f = self.dir / "status.json"
        if not f.exists():
            return {"state": "queued"}
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            # Being read mid-write is normal and not an error; the worker
            # replaces the file atomically, so the next poll will be fine.
            return {"state": "working"}


def staging_path(filename: str) -> tuple[str, Path]:
    """Where an upload should be streamed to, before it is a job.

    Handed out BEFORE the bytes arrive so the server can write straight to disk
    instead of holding the file in memory. Returns the job id it will become and
    the path to write. `adopt` finishes the job once the bytes are there;
    `abandon` cleans up if they never fully arrive.
    """
    jid = uuid.uuid4().hex[:12]
    d = JOBS / jid
    (d / "out").mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()[:6] or ".mp3"
    return jid, d / f"source{suffix}"


def write_status(jid: str, **fields) -> None:
    """Atomic merge into a job's status.json.

    The worker has its own copy of this taking a Path. This one takes a job id
    because the server needs to write a status BEFORE there is a job - a link
    that is still downloading has to be pollable, and "no status file yet" is
    indistinguishable from "queued" to the page.

    Same atomic replace, for the same reason: the web process reads this file
    while it is being written.
    """
    d = JOBS / jid
    d.mkdir(parents=True, exist_ok=True)
    path = d / "status.json"
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current))
    os.replace(tmp, path)


def abandon(jid: str) -> None:
    shutil.rmtree(JOBS / jid, ignore_errors=True)


def new_job(data: bytes, filename: str, parts: list[str], model: str) -> Job:
    """Byte-buffered entry point. Kept for callers that already hold the file.

    The web upload does NOT come through here any more -- it streams to
    `staging_path()` and calls `adopt()`. See `serve.read_multipart_to_disk`
    for the measurement that motivated it.
    """
    if len(data) > MAX_MB << 20:
        raise Refused(f"that file is over {MAX_MB} MB")
    jid, src = staging_path(filename)
    src.write_bytes(data)
    return adopt(jid, src, filename, parts, model)


def adopt(jid: str, src: Path, filename: str, parts: list[str], model: str,
          extra: dict | None = None) -> Job:
    """Turn a file already on disk into a queued job.

    `extra` is merged into job.json BEFORE the job is queued, and that ordering
    is the whole reason the argument exists rather than the caller patching the
    file afterwards. `adopt` starts the pump on the last line, so a caller that
    writes to job.json after it returns is racing the worker for the same file,
    and the losing side of that race is a job that quietly produces no video.
    """
    from . import decode, separate

    d = JOBS / jid
    if src.stat().st_size > MAX_MB << 20:
        shutil.rmtree(d, ignore_errors=True)
        raise Refused(f"that file is over {MAX_MB} MB")
    # `parts` is ignored on purpose and kept in the signature so old callers do
    # not break. Everything is produced every time; choosing happens afterwards,
    # when there is something to choose between.
    parts = separate.available(model)

    try:
        media = decode.probe(src)
    except Exception as e:                                    # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        raise Refused("that does not look like an audio file") from e
    if media.duration > MAX_MINUTES * 60:
        shutil.rmtree(d, ignore_errors=True)
        raise Refused(f"that is {media.duration/60:.1f} minutes; "
                      f"the limit here is {MAX_MINUTES:.0f}")

    spec = {"source": src.name, "parts": parts, "model": model,
            "filename": filename, "created": time.time()}
    spec.update(extra or {})
    (d / "job.json").write_text(json.dumps(spec))
    job = Job(jid, d)
    with _lock:
        _queue.append(jid)
    threading.Thread(target=_pump, daemon=True).start()
    return job


def place_take(jid: str, take: Path) -> None:
    """Run the liveroom placement for a finished job, out of process.

    Same shape as `_run` for the separator, and for the same reasons: the real
    interpreter rather than a name, a memory cap where systemd will give one,
    and a loud line saying which of those is actually in force.
    """
    d = JOBS / jid
    cmd = [sys.executable, "-m", "earshot.placer", str(d), take.name]
    scoped = ["systemd-run", "--user", "--scope", "--quiet",
              "-p", f"MemoryMax={WORKER_MEMORY_MAX}", "-p", "MemorySwapMax=0"] + cmd
    root = Path(__file__).resolve().parents[1]
    for attempt in (scoped, cmd):
        try:
            print(f"take {jid}: placing "
                  f"{'under a ' + WORKER_MEMORY_MAX + ' cap' if attempt is scoped else 'UNCAPPED'}",
                  flush=True)
            subprocess.run(attempt, cwd=root, capture_output=True, text=True,
                           timeout=int(MAX_MINUTES * 60 * 6))
            return
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            write_status(jid, liveroom={"state": "failed",
                                        "why": "the placement took too long"})
            return


def position(jid: str) -> int:
    """How many jobs are in front of this one. 0 means it is running or next."""
    with _lock:
        return _queue.index(jid) if jid in _queue else 0


def _pump() -> None:
    global _current
    while True:
        with _lock:
            if _current is not None or not _queue:
                return
            _current = _queue[0]
        try:
            _run(JOBS / _current)
        finally:
            with _lock:
                if _current in _queue:
                    _queue.remove(_current)
                _current = None


def _run(d: Path) -> None:
    from . import worker

    # sys.executable, not the name. A systemd user unit's PATH does not include
    # ~/.local/bin, so "python3.12" resolves in my shell and not in the service
    # -- the place I test is not the place it runs, which has cost this project
    # three separate outages already.
    cmd = [sys.executable, "-m", "earshot.worker", str(d)]
    # A memory cap where systemd will give us one. Proved to fire, with a
    # control, by tools/check_memory_cap.py: 500 MB under a 200 MB cap is killed
    # with SIGKILL, and the identical allocation succeeds uncapped. The control
    # is the point -- a capped run dying on its own proves nothing.
    # Without this, one bad job takes the talk server and the doorbell with it.
    scoped = ["systemd-run", "--user", "--scope", "--quiet",
              "-p", f"MemoryMax={WORKER_MEMORY_MAX}", "-p", "MemorySwapMax=0"] + cmd
    root = Path(__file__).resolve().parents[1]
    for attempt in (scoped, cmd):
        capped = attempt is scoped
        try:
            # Say which guard is actually in force. A fallback that happens
            # quietly is a cap everyone believes in and nobody has.
            print(f"job {d.name}: starting "
                  f"{'under a ' + WORKER_MEMORY_MAX + ' cap' if capped else 'UNCAPPED (systemd-run unavailable)'}",
                  flush=True)
            p = subprocess.run(attempt, cwd=root, capture_output=True, text=True,
                               timeout=int(MAX_MINUTES * 60 * 8))
            if p.returncode == 0 or (d / "status.json").exists():
                if p.returncode != 0:
                    st = json.loads((d / "status.json").read_text())
                    if st.get("state") != "failed":
                        # Killed rather than raised: no traceback ever ran, so
                        # the worker never got to record why.
                        worker.write_status(
                            d, state="failed",
                            why="the job was killed, most likely for memory")
                return
        except FileNotFoundError:
            continue                    # no systemd-run here; fall through
        except subprocess.TimeoutExpired:
            worker.write_status(d, state="failed", why="took too long and was stopped")
            return
    worker.write_status(d, state="failed", why="could not start the worker")


def sweep(now: float | None = None) -> int:
    """Delete the MEDIA of finished jobs older than KEEP_HOURS. Keep the index.

    THIS USED TO DELETE THE WHOLE DIRECTORY, and its own comment said that was
    safe because "the S3 copy is the point of having one and the download route
    pulls it back on demand". That sentence was false and it stayed false for
    two days.

    `job.json` and `status.json` ARE the index. `describe()` returns
    `{"state": "unknown"}` without them, `recent()` skips the job entirely, and
    the download route takes the allowed filenames FROM status.json -- so with
    the directory gone there is nothing to authorise a fetch and nothing to
    fetch it for. The files sat in S3, perfectly intact, and every link to them
    404'd.

    Bruno found it by opening the site the morning after I built the videos and
    seeing an empty projects list: "Hold up. What did you do new to earshot?
    Did you add the videos in the site how i asked?" I had verified that
    feature end to end and sent him the two files, and then the housekeeping
    quietly removed the evidence six hours later.

    So: the audio and video go, the two small json files stay. A swept job still
    opens, still lists, and still serves -- the first request for a file pulls
    it back from S3.
    """
    now = now or time.time()
    gone = 0
    if not JOBS.exists():
        return 0
    for d in JOBS.iterdir():
        spec = d / "job.json"
        if not spec.is_file():
            continue
        try:
            created = json.loads(spec.read_text()).get("created", 0)
        except json.JSONDecodeError:
            created = d.stat().st_mtime
        if now - created <= KEEP_HOURS * 3600:
            continue

        # By each file's OWN age, so a file just restored from S3 for somebody
        # who is listening to it right now does not get swept out from under
        # them on the next pass.
        freed = 0
        for f in d.rglob("*"):
            if not f.is_file() or f.name in ("job.json", "status.json"):
                continue
            if now - f.stat().st_mtime > KEEP_HOURS * 3600:
                f.unlink(missing_ok=True)
                freed += 1
        if freed:
            gone += 1
    return gone


def describe(job: Job) -> dict:
    """Status, plus what was actually found -- which is half the answer.

    The levels and the verdict are computed once by the worker, which has the
    wav files in front of it, rather than re-derived here from the mp3s. Two
    places measuring the same thing is two places to disagree.
    """
    st = dict(job.status)
    st["id"] = job.id
    if st.get("state") == "done":
        levels = st.get("levels") or {}
        verdict = st.get("verdict") or {}
        status = verdict.get("status", {})
        st["found"] = {
            name: {"file": fname,
                   "db": levels.get(name),
                   "status": status.get(name, "present")}
            for name, fname in (st.get("parts") or {}).items()}
        _check_take_survives(job, st)
    elif st.get("state") in (None, "queued"):
        st["ahead"] = position(job.id)
    return st


def _check_take_survives(job: Job, st: dict) -> None:
    """A take is the ONE artefact here that is not backed up, so say when it goes.

    Stems and videos are copied to S3 and the download route pulls them back on
    demand, which is why `sweep()` can delete the local media and every link
    keeps working. `placed.mp3` and `unplaced.mp3` are never uploaded, so six
    hours after somebody sings the files go and nothing notices: the status
    still says `state: done`, the page still offers two downloads, and "Watch
    it back" still builds an Audio pointing at a 404 and plays silence.

    Found by driving the new sing-along door in a real browser and reading the
    server log: two 404s for files the page was confidently advertising.

    This does not fix the loss. Whether takes should be kept, and for how long,
    is a product decision and Bruno's -- they are a recording of somebody's
    voice, which is not the same class of thing as the stems of a song they
    uploaded. What it stops is the page lying about it.
    """
    L = st.get("liveroom")
    if not isinstance(L, dict) or L.get("state") != "done":
        return
    out = job.dir / "out"
    names = [L[k] for k in ("placed", "unplaced") if L.get(k)]
    if names and all((out / n).is_file() for n in names):
        return
    st["liveroom"] = dict(L, state="gone",
                          why="the recording of your take was cleared with the "
                              "rest of the media after a few hours")


# ---------------------------------------------------------------------------
# Keeping the work. Bruno: "I should be able to come in and out of the app.
# Rhere should be a saved project in s3 linked in that site"
# ---------------------------------------------------------------------------

BUCKET = "soulful-iris-652539275920"
PREFIX = "earshot"
MAIL_TO = "brunodiaz@me.com"
MAIL_FROM = "iris@soulful-ai.dev"
SITE = "https://earshot.soulful-ai.dev"


def s3_key(jid: str, filename: str) -> str:
    return f"{PREFIX}/{jid}/{filename}"


def put_s3(jid: str, path: Path) -> bool:
    """Copy one finished stem to S3. Private; nothing here is world-readable.

    These are somebody's own music pulled apart. The bucket blocks public
    access and the only way to a file is through this site, which is a
    deliberate choice and not an oversight.
    """
    r = subprocess.run(
        ["aws", "s3", "cp", str(path), f"s3://{BUCKET}/{s3_key(jid, path.name)}",
         "--quiet"], capture_output=True, text=True, timeout=900)
    return r.returncode == 0


def fetch_s3(jid: str, filename: str, into: Path) -> Path | None:
    """Bring a stem back from S3 when the local copy has been swept.

    This is what makes a link in an email still work tomorrow. Presigned URLs
    would have been simpler and wrong: they are signed with the instance role's
    temporary credentials and stop working when those rotate, so an emailed
    link would quietly die in a few hours. A stable path on this site that
    fetches from S3 on demand does not have that problem.
    """
    into.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["aws", "s3", "cp", f"s3://{BUCKET}/{s3_key(jid, filename)}", str(into),
         "--quiet"], capture_output=True, text=True, timeout=900)
    return into if r.returncode == 0 and into.exists() else None


def in_s3(jid: str) -> dict:
    """What this job still has in S3, by filename."""
    r = subprocess.run(
        ["aws", "s3api", "list-objects-v2", "--bucket", BUCKET,
         "--prefix", f"{PREFIX}/{jid}/", "--query", "Contents[].{k:Key,s:Size}",
         "--output", "json"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout.strip() or r.stdout.strip() == "null":
        return {}
    try:
        return {Path(o["k"]).name: o["s"] for o in json.loads(r.stdout) or []}
    except (json.JSONDecodeError, TypeError, KeyError):
        return {}


def email_done(jid: str, spec: dict, status: dict) -> bool:
    """Tell him it is finished, with links that will still work tomorrow.

    Sent once, by the worker, at the moment the job completes -- because the
    whole point is that he does not have to be looking at the page. The links
    go to this site rather than to S3 directly, so they do not expire.
    """
    verdict = (status.get("verdict") or {}).get("kind", "")
    levels = status.get("levels") or {}
    name = spec.get("filename", "your file")
    order = ["vocals", "band", "drums", "bass", "guitar", "piano", "other"]
    lines = []
    for part in order:
        f = (status.get("parts") or {}).get(part)
        if not f:
            continue
        db = levels.get(part)
        gone = db is None or db < -30.0
        label = "voice" if part == "vocals" else part
        lines.append(f"  {label:<8} {'not in this recording' if gone else SITE + '/jobs/' + jid + '/' + f}")

    body = (f"{name}\n{verdict}\n\n"
            + "\n".join(lines)
            + f"\n\nAll of it, and the players: {SITE}/#{jid}\n"
            "Kept in S3, so these links keep working.\n")
    r = subprocess.run(
        ["aws", "ses", "send-email", "--region", "us-east-1",
         "--from", MAIL_FROM, "--destination", f"ToAddresses={MAIL_TO}",
         "--message", json.dumps({
             "Subject": {"Data": f"earshot: {name}"},
             "Body": {"Text": {"Data": body}}})],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"job {jid}: email FAILED: {r.stderr.strip()[:200]}", flush=True)
    return r.returncode == 0


def title_of(filename: str | None) -> str:
    """A song's name, not its file's name.

    Bruno, 2026-09-04: *"Why the whole link."* The projects list was printing
    `Metric - Black Sheep (Brie Larson Vocal Version) ft. Brie Larson.mp3`,
    extension and all, wrapping over two lines. That is a path, and it was
    sitting where a title goes.

    Only the extension is dropped here. The parentheticals and the featured
    artist stay, because they are part of what the song is called and throwing
    them away is a different decision from tidying a filename. The page clamps
    the line; the text is not shortened.
    """
    name = (filename or "").strip()
    if not name:
        return "(unnamed)"
    # REPEATEDLY, because one strip is not enough. A link job is saved as
    # f"{title}.mp3", and archive titles routinely end in ".mp4" themselves, so
    # the file on disk is "Beck - Ramona (Lyrics + HD).mp4.mp3" and a single
    # `Path.stem` leaves ".mp4" sitting in the title. Only known media
    # extensions are removed, so a song called "Blue Monday 88" keeps its name.
    media = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".opus", ".flac",
             ".ogg", ".aac", ".mkv", ".mov", ".avi"}
    for _ in range(4):
        stem = Path(name).stem
        if stem and Path(name).suffix.lower() in media:
            name = stem
        else:
            break
    return name or "(unnamed)"


def recent(limit: int = 12) -> list[dict]:
    """Finished jobs, newest first, for the list on the page."""
    out = []
    if not JOBS.exists():
        return out
    for d in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        spec_f, st_f = d / "job.json", d / "status.json"
        if not (spec_f.is_file() and st_f.is_file()):
            continue
        try:
            spec = json.loads(spec_f.read_text())
            st = json.loads(st_f.read_text())
        except json.JSONDecodeError:
            continue
        if st.get("state") != "done":
            continue
        # Name and kind ALONE are not enough to tell two jobs apart, and that
        # is not a hypothetical. Bruno ran the same link twice, once before a
        # fix and once after, and his projects list showed two rows reading
        # "Metric - Black Sheep" / "music with a voice in it". He opened the
        # older one, saw no subtitles, and asked whether he should redo the
        # link. Nothing on screen could have told him which was which.
        vid = st.get("videos") or {}
        out.append({"id": d.name, "name": title_of(spec.get("filename")),
                    "kind": (st.get("verdict") or {}).get("kind", ""),
                    "created": spec.get("created", 0),
                    "video": bool(vid.get("with_voice")),
                    "subtitles": bool(vid.get("subtitles"))})
        if len(out) >= limit:
            break
    return out
