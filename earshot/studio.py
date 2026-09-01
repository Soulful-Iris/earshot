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


def new_job(data: bytes, filename: str, parts: list[str], model: str) -> Job:
    from . import decode, separate

    if len(data) > MAX_MB << 20:
        raise Refused(f"that file is over {MAX_MB} MB")
    # `parts` is ignored on purpose and kept in the signature so old callers do
    # not break. Everything is produced every time; choosing happens afterwards,
    # when there is something to choose between.
    parts = separate.available(model)

    jid = uuid.uuid4().hex[:12]
    d = JOBS / jid
    (d / "out").mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()[:6] or ".mp3"
    src = d / f"source{suffix}"
    src.write_bytes(data)

    try:
        media = decode.probe(src)
    except Exception as e:                                    # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        raise Refused("that does not look like an audio file") from e
    if media.duration > MAX_MINUTES * 60:
        shutil.rmtree(d, ignore_errors=True)
        raise Refused(f"that is {media.duration/60:.1f} minutes; "
                      f"the limit here is {MAX_MINUTES:.0f}")

    (d / "job.json").write_text(json.dumps({
        "source": src.name, "parts": parts, "model": model,
        "filename": filename, "created": time.time()}))
    job = Job(jid, d)
    with _lock:
        _queue.append(jid)
    threading.Thread(target=_pump, daemon=True).start()
    return job


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
    """Delete finished jobs older than KEEP_HOURS. Returns how many went."""
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
        if now - created > KEEP_HOURS * 3600:
            # Local only. The S3 copy is the point of having one, and the
            # download route pulls it back on demand -- so sweeping here frees
            # the disk without breaking a link he was emailed yesterday.
            shutil.rmtree(d, ignore_errors=True)
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
    elif st.get("state") in (None, "queued"):
        st["ahead"] = position(job.id)
    return st


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
        out.append({"id": d.name, "name": spec.get("filename", "(unnamed)"),
                    "kind": (st.get("verdict") or {}).get("kind", ""),
                    "created": spec.get("created", 0)})
        if len(out) >= limit:
            break
    return out
