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
    known = separate.available(model)
    parts = [p for p in parts if p in known]
    if not parts:
        raise Refused("pick at least one thing to pull out")

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
            shutil.rmtree(d, ignore_errors=True)
            gone += 1
    return gone


def describe(job: Job) -> dict:
    """Status, plus what was actually found -- which is half the answer.

    A stem for an instrument the song does not contain comes out near-silent
    rather than missing, and handing somebody a silent file they have to open to
    understand is worse than telling them. Measured on a rock song with no
    piano: piano -67 dB, other -49, against vocals -14 and drums -13.
    """
    from . import decode
    from .loudness import lufs_of
    import numpy as np

    st = dict(job.status)
    st["id"] = job.id
    if st.get("state") == "done":
        found = {}
        for name, fname in (st.get("parts") or {}).items():
            path = job.dir / "out" / fname
            if not path.exists():
                continue
            try:
                media = decode.probe(path)
                sample = decode.read_all(media, channels=1)[:, 0]
                level = lufs_of(np.repeat(sample[:, None], 2, axis=1))
            except Exception:                                 # noqa: BLE001
                level = float("nan")
            found[name] = {
                "file": fname,
                "lufs": None if level != level or level == float("-inf") else round(level, 1),
                "present": bool(level == level and level > -45.0),
            }
        st["found"] = found
    elif st.get("state") in (None, "queued"):
        st["ahead"] = position(job.id)
    return st
