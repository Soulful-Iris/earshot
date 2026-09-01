"""One separation job, in its own process.

    python -m earshot.worker /path/to/job

Out of process on purpose. Separation peaks at about 1.4 GB on a box with 1.8 GB
that is also serving a talk page and a doorbell, so a job that goes wrong has to
be able to die without taking those with it. The parent runs this under a memory
cap where systemd allows it; either way a crash here leaves a job marked failed
and everything else still answering.

State is a file, not memory: `status.json` is rewritten as it goes, so the web
process can report progress without holding a handle on anything, and a worker
killed mid-run leaves a last known position rather than a lie.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def write_status(job: Path, **fields) -> None:
    """Atomic, because the web process reads this while it is being written."""
    path = job / "status.json"
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m earshot.worker JOB_DIR", file=sys.stderr)
        return 2
    job = Path(argv[0])
    spec = json.loads((job / "job.json").read_text())
    started = time.time()

    try:
        from . import decode, separate

        source = job / spec["source"]
        media = decode.probe(source)
        write_status(job, state="working", done_seconds=0.0,
                     total_seconds=media.duration, started=started)

        last = [0.0]

        def progress(done: float, total: float) -> None:
            # Rewriting a file ten times a second is pointless; the page polls
            # every couple of seconds.
            now = time.time()
            if now - last[0] < 1.0 and done < total:
                return
            last[0] = now
            write_status(job, done_seconds=round(done, 1),
                         total_seconds=round(total, 1),
                         elapsed=round(now - started, 1))

        model_name = spec.get("model", separate.SIX)
        # Everything, always. The model produces all its sources per window
        # regardless, so asking for one costs the same as asking for seven --
        # which means there is no reason to make somebody choose before they
        # know what is in the file. They choose afterwards, by downloading.
        stems = separate.split(
            source, job / "out",
            parts=separate.available(model_name),
            model_name=model_name,
            on_progress=progress)

        # Measure each stem against the mix BEFORE encoding, so the numbers
        # describe the separation and not the mp3 encoder.
        import numpy as np
        from .loudness import lufs_of

        def rel(path):
            x = decode.read_all(decode.probe(path), channels=1)[:, 0]
            v = lufs_of(np.repeat(x[:, None], 2, axis=1))
            return None if not np.isfinite(v) else round(v - mix_lufs, 1)

        mix_x = decode.read_all(media, channels=1)[:, 0]
        mix_lufs = lufs_of(np.repeat(mix_x[:, None], 2, axis=1))
        levels = {n: rel(p) for n, p in (stems.parts or {}).items()}
        del mix_x

        made = {}
        for name, path in (stems.parts or {}).items():
            mp3 = (job / "out" / f"{name}.mp3")
            # mp3 because he asked to import one and will want to play the
            # result on a phone. 192k is transparent enough for a stem and a
            # tenth the size of the wav.
            import subprocess
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                 "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                check=True, timeout=1800)
            path.unlink(missing_ok=True)
            made[name] = mp3.name

        write_status(job, state="done", parts=made, levels=levels,
                     verdict=separate.classify(levels),
                     elapsed=round(time.time() - started, 1))
        return 0

    except Exception as e:                                   # noqa: BLE001
        # A failure has to be recorded, not just logged. A job that stops
        # writing looks identical to a job that is being slow, and this project
        # has a long history of silence being read as progress.
        write_status(job, state="failed", why=f"{type(e).__name__}: {e}"[:300],
                     elapsed=round(time.time() - started, 1))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
