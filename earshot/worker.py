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

        # The model is finished with. Drop it before doing anything else: it is
        # about 1.3 GB resident and every megabyte held here is a megabyte the
        # measuring cannot have.
        import gc
        separate._models.clear()
        gc.collect()

        # Measure each stem against the mix BEFORE encoding, so the numbers
        # describe the separation and not the mp3 encoder.
        #
        # STREAMED, and that is the whole point of this block. The first version
        # read each whole file into memory with read_all, doubled it with
        # np.repeat to make it stereo, and filtered it in float64 -- roughly
        # 300 MB of transient for a three-minute song, in a process still
        # holding the model. It worked on every 30-second fixture I had and was
        # killed by the memory cap at 203.8 of 203.8 seconds on the first real
        # song anybody uploaded: separation complete, all seven stems on disk,
        # and the job reported as failed two seconds from the end.
        #
        # split() streams for exactly this reason. Writing a post-step that did
        # not is the same bug the streaming was built to prevent, one function
        # further down.
        import math

        def gated(path) -> float:
            return separate._gated(path)

        mix_lufs = gated(source)
        levels = {}
        for name, path in (stems.parts or {}).items():
            v = gated(path)
            levels[name] = None if not math.isfinite(v) else round(v - mix_lufs, 1)
            gc.collect()

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

        # --- the words, for singing along -----------------------------------
        # Read off the ISOLATED VOCAL, which is why no lyrics database is
        # needed and why the timing is the real timing rather than a guess at
        # where the lines fall.
        #
        # Wrapped, and never able to fail the job. Same rule as delivery below:
        # the stems are the product, the highlighter is the extra. Some singing
        # genuinely cannot be transcribed and that is an answer, not an error.
        #
        # It reads the finished mp3 (~5 MB for a four-minute vocal), not the wav
        # and not into an array. The last post-step added here read whole files
        # in beside a 1.3 GB model and got a job killed two seconds from the end.
        lyrics = None
        try:
            vocal = job / "out" / (made.get("vocals") or "")
            if vocal.is_file():
                from . import words as _words
                lyr = _words.read_vocal(vocal)
                _words.save(lyr, job / "out" / "words.json")
                lyrics = {"count": len(lyr.words), "language": lyr.language,
                          "confidence": round(lyr.mean_confidence, 2),
                          "unsure": lyr.unsure, "file": "words.json"}
        except Exception as e:                                    # noqa: BLE001
            lyrics = {"count": 0, "why": str(e)[:200]}
        gc.collect()

        verdict = separate.classify(levels)
        write_status(job, state="done", parts=made, levels=levels,
                     verdict=verdict, lyrics=lyrics,
                     elapsed=round(time.time() - started, 1))

        # Everything past this point is delivery, and NONE of it may turn a
        # finished job back into a failed one. The stems exist and the status
        # already says done; if S3 or SES is having a bad day that is worth
        # recording and not worth losing the work over. This is the exact shape
        # that ate the first real song anybody uploaded -- separation complete,
        # then killed in the step afterwards, reported as a failure.
        from . import studio
        try:
            kept = [n for n, f in made.items()
                    if studio.put_s3(job.name, job / "out" / f)]
            write_status(job, kept=sorted(kept))
            print(f"job {job.name}: {len(kept)}/{len(made)} stems in S3", flush=True)
        except Exception as e:                               # noqa: BLE001
            write_status(job, kept=[], keep_error=str(e)[:200])
            print(f"job {job.name}: S3 upload failed: {e}", flush=True)

        try:
            st = json.loads((job / "status.json").read_text())
            if studio.email_done(job.name, spec, st):
                write_status(job, emailed=True)
                print(f"job {job.name}: emailed", flush=True)
        except Exception as e:                               # noqa: BLE001
            print(f"job {job.name}: email failed: {e}", flush=True)

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
