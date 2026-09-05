"""One take, placed in the record's room, in its own process.

    python -m earshot.placer /path/to/job take.webm

Out of process for the same reason the separator is: `matchering` loads the
whole take and the whole reference into memory and ends in its own limiter, and
the web server it would otherwise share a process with is capped at 700 MB and
also answers the door. A take that goes wrong has to be able to die alone.

It writes into the SAME status.json the page already polls, under `liveroom`,
so nothing new had to learn how to report progress.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from .worker import write_status


def score_take(job: Path, take: Path, vocals: Path) -> None:
    """Compare the take's pitch line against the record's, and write the result.

    The reference track is cached beside the stems as `pitch.npz`. It never
    changes, it costs about 12 seconds for a three and a half minute song, and
    every take after the first reads it back in milliseconds.
    """
    from . import unison, video, words as _words

    write_status(job, unison={"state": "working",
                              "stage": "reading the melody off the record"})
    ref = unison.track(vocals, cache=job / "out" / "pitch.npz")

    write_status(job, unison={"state": "working", "stage": "listening to you"})
    yours = unison.track(take)

    shift = unison.offset(ref, yours)
    v = unison.score(ref, yours, shift)
    if v is None:
        write_status(job, unison={
            "state": "refused",
            "why": ("you and the singer only overlapped for a moment, so there "
                    "is nothing to compare. Sing a bit more of it.")})
        print("unison: not enough overlap", flush=True)
        return

    lines = []
    wjson = job / "out" / "words.json"
    if wjson.is_file():
        lines = video.cues(_words.load(wjson).words)
    scored = unison.per_line(ref, yours, shift, lines) if lines else []
    best, worst = unison.best_and_worst(scored)
    # Reported BESIDE the pitch verdict, never folded into it. Being flat and
    # being late are different mistakes with different fixes, and the reason
    # this exists is that they were arriving as one number.
    tim = unison.timing(scored)

    write_status(job, unison={
        "state": "done",
        "headline": unison.headline(v, scored),
        "timing_headline": unison.timing_headline(tim),
        "timing": tim,
        **v.as_dict(),
        "best": None if best is None else
                {"text": best.text, "cents": best.median_cents},
        "worst": None if worst is None else
                 {"text": worst.text, "cents": worst.median_cents,
                  "signed": worst.flat_or_sharp},
        "lines": [{"text": l.text, "start": round(l.start, 2),
                   "end": round(l.end, 2), "cents": l.median_cents,
                   "signed": l.flat_or_sharp, "late_ms": l.late_ms}
                  for l in scored],
        "contours": unison.contours(ref, yours, shift)})
    print(f"unison: {unison.headline(v, scored)}", flush=True)
    if tim:
        print(f"timing: {unison.timing_headline(tim)}", flush=True)


def make_encore(job: Path) -> None:
    """Build the take video, upload it, and say so in the status.

    Uploaded, unlike the take audio itself. Stems and videos go to S3 and the
    download route pulls them back, which is why `sweep()` can free the local
    media and every link keeps working; `placed.mp3` and `unplaced.mp3` never
    were, so they vanish after a few hours. This is the artefact somebody would
    actually send to a person, so it is the one that has to outlive the sweep.
    """
    from . import encore
    from . import studio

    write_status(job, encore={"state": "working",
                              "stage": "drawing the two pitch lines"})
    made = encore.render(job)
    if made is None:
        # Not a failure. A job with no take, no score or no source video has
        # nothing to build from, and the door says so in one line rather than
        # offering a player with nothing behind it.
        write_status(job, encore={"state": "none"})
        print("encore: nothing to build from", flush=True)
        return

    studio.put_s3(job.name, made)
    write_status(job, encore={"state": "done", "file": made.name,
                              "bytes": made.stat().st_size})
    print(f"encore: {made.name} ({made.stat().st_size // 1024} KB)", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print("usage: python -m earshot.placer JOB_DIR TAKE", file=sys.stderr)
        return 2
    job, take = Path(argv[0]), Path(argv[0]) / argv[1]
    started = time.time()

    try:
        from . import liveroom as lr

        st = json.loads((job / "status.json").read_text())
        parts = st.get("parts") or {}
        vocals = job / "out" / (parts.get("vocals") or "")
        band = job / "out" / (parts.get("band") or "")
        if not (vocals.is_file() and band.is_file()):
            raise RuntimeError("this job has no separated vocal and band to work from")

        write_status(job, liveroom={"state": "working",
                                    "stage": "measuring the room on the record"})
        d = lr.assess(take, vocals, band, job / "lr")

        if not d.ok:
            # A refusal is an answer and it is finished, not failed. The page
            # says the sentence; nothing is produced on purpose.
            write_status(job, liveroom={
                "state": "refused", "why": d.why,
                "record_tail": None if not d.record or d.record.tail is None
                               else round(d.record.tail, 3),
                "take_tail": None if not d.take or d.take.tail is None
                             else round(d.take.tail, 3),
                "elapsed": round(time.time() - started, 1)})
            print(f"refused: {d.why}", flush=True)
            return 0

        write_status(job, liveroom={"state": "working",
                                    "stage": "putting you in the room"})
        steps = lr.place(take, vocals, band, d.room, job / "out")

        write_status(job, liveroom={
            "state": "done",
            "record_tail": round(d.record.tail, 3),
            "take_tail": round(d.take.tail, 3),
            "room": {"rt60": d.room.rt60, "tilt": d.room.tilt,
                     "measured": None if d.room.measured is None
                                 else round(d.room.measured, 3)},
            "placed": steps.get("placed"), "unplaced": steps.get("unplaced"),
            "target_lufs": steps.get("target_lufs"),
            "placed_lufs": steps.get("placed_lufs"),
            "room_took_db": steps.get("room_took_db"),
            "colour": steps.get("colour"),
            "elapsed": round(time.time() - started, 1)})
        print(f"placed in {time.time() - started:.0f}s: {steps}", flush=True)

        # --- how close you sang it ------------------------------------------
        #
        # Separate from the placement on purpose. `liveroom` moves your voice
        # into the record's room; this has an opinion about whether you sang
        # the right notes, and neither should be able to break the other. It
        # runs last and is wrapped, because a scoring failure must never lose a
        # take that is already mixed and on disk.
        try:
            score_take(job, take, vocals)
        except Exception as e:                               # noqa: BLE001
            traceback.print_exc()
            write_status(job, unison={"state": "failed",
                                      "why": f"{type(e).__name__}: {e}"[:200]})

        # --- the thing you keep ---------------------------------------------
        #
        # Last, and wrapped like the scoring above, for the same reason: a
        # video that fails to build must never lose a take that is already
        # mixed and on disk. It needs both the placement and the score, so it
        # cannot run any earlier than this.
        try:
            make_encore(job)
        except Exception as e:                               # noqa: BLE001
            traceback.print_exc()
            write_status(job, encore={"state": "failed",
                                      "why": f"{type(e).__name__}: {e}"[:200]})
        return 0

    except Exception as e:                                   # noqa: BLE001
        # Recorded, never only logged. A process that stops writing looks
        # exactly like one that is being slow.
        write_status(job, liveroom={"state": "failed",
                                    "why": f"{type(e).__name__}: {e}"[:300],
                                    "elapsed": round(time.time() - started, 1)})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
