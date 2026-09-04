"""A take is the one artefact here that is not backed up, so the page must say
when it goes.

Stems and videos are copied to S3, and the download route pulls them back on
demand -- that is why `sweep()` can free the local media and every link keeps
working. `placed.mp3` and `unplaced.mp3` are never uploaded. So a few hours
after somebody sings, the two files go and nothing notices: `status.json` still
says `liveroom.state == "done"`, the page still offers two downloads, and the
"Watch it back" button still builds an Audio pointing at a 404 and plays
silence at you.

Found on 2026-09-04 by driving the new sing-along door in a real browser and
reading the server log, which had two 404s in it for files the page was
confidently advertising. No test could have found it, because every test here
builds its own files.

The pair below is the point. `describe()` reporting "gone" is worth nothing on
its own -- a function that always says gone would pass a one-sided test and
look exactly the same. The control is the first test.
"""
from __future__ import annotations

import json

import pytest

from earshot import studio


@pytest.fixture
def sung(tmp_path, monkeypatch):
    """A finished job that somebody has sung on, with both takes on disk."""
    monkeypatch.setattr(studio, "JOBS", tmp_path)
    jid = "abc123def456"
    d = tmp_path / jid
    (d / "out").mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"source": "audio.mp3"}))
    (d / "status.json").write_text(json.dumps(
        {"state": "done",
         "parts": {"vocals": "vocals.mp3", "band": "band.mp3"},
         "levels": {"vocals": -1.0, "band": -2.0},
         "verdict": {"kind": "music with a voice in it", "status": {}},
         "liveroom": {"state": "done", "placed": "placed.mp3",
                      "unplaced": "unplaced.mp3", "record_tail": 0.8,
                      "take_tail": 0.76, "room": {"rt60": 0.9}},
         "unison": {"state": "done", "headline": "40 cents close"}}))
    for name in ("vocals.mp3", "band.mp3", "placed.mp3", "unplaced.mp3"):
        (d / "out" / name).write_bytes(b"x" * 4096)
    return studio.Job(jid, d)


def test_a_take_still_on_disk_is_still_offered(sung):
    """The control. Without this, a check that only ever says 'gone' passes."""
    got = studio.describe(sung)
    assert got["liveroom"]["state"] == "done"
    assert "why" not in got["liveroom"]


def test_a_take_whose_audio_was_swept_is_reported_gone(sung):
    for name in ("placed.mp3", "unplaced.mp3"):
        (sung.dir / "out" / name).unlink()

    got = studio.describe(sung)
    assert got["liveroom"]["state"] == "gone"
    assert got["liveroom"]["why"]


def test_half_a_take_is_gone_too(sung):
    """Both files back one control each -- the raw take and the placed mix are
    different artefacts. One surviving is not a playable take."""
    (sung.dir / "out" / "placed.mp3").unlink()

    assert studio.describe(sung)["liveroom"]["state"] == "gone"


def test_the_score_survives_the_audio(sung):
    """What makes 'gone' the right word rather than 'failed': the numbers live
    in status.json, so how close you sang is still there when the recording of
    it is not. The page keeps showing the score and stops offering the audio."""
    (sung.dir / "out" / "placed.mp3").unlink()
    (sung.dir / "out" / "unplaced.mp3").unlink()

    got = studio.describe(sung)
    assert got["liveroom"]["state"] == "gone"
    assert got["unison"]["state"] == "done"
    assert got["unison"]["headline"] == "40 cents close"


def test_a_job_nobody_sang_on_is_untouched(sung):
    """No liveroom at all must not become a 'gone' take, which would put a
    sentence about a cleared recording on a job that never had one."""
    st = json.loads((sung.dir / "status.json").read_text())
    del st["liveroom"]
    (sung.dir / "status.json").write_text(json.dumps(st))

    assert "liveroom" not in studio.describe(sung)


def test_a_failed_take_is_left_as_failed(sung):
    """`failed` and `gone` are different sentences and only one of them is
    true. This fired for real: a job on the box has both files on disk and a
    liveroom that failed for an unrelated reason."""
    st = json.loads((sung.dir / "status.json").read_text())
    st["liveroom"] = {"state": "failed", "why": "the take was silent"}
    (sung.dir / "status.json").write_text(json.dumps(st))

    got = studio.describe(sung)
    assert got["liveroom"]["state"] == "failed"
    assert got["liveroom"]["why"] == "the take was silent"
