"""A swept job must still open, still list, and still be servable.

There was no test here at all, which is how the bug lived for two days: the
sweeper deleted the whole job directory, its own comment claimed that was safe
because S3 has a copy, and nothing anywhere checked whether a link to a swept
job still worked. It did not. `describe()` answered `{"state": "unknown"}`,
`recent()` skipped the job, and the download route had no allowlist to check a
filename against, so the files sat in S3 and every route to them 404'd.

Bruno found it by opening the site: "Did you add the videos in the site how i
asked?" The feature was built, verified end to end, and then quietly erased six
hours later by housekeeping.

These tests assert the property that actually matters -- the link survives --
rather than the mechanics of what gets deleted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from earshot import studio


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A finished job whose media is older than the keep window."""
    monkeypatch.setattr(studio, "JOBS", tmp_path)
    jid = "abc123def456"
    d = tmp_path / jid
    (d / "out").mkdir(parents=True)

    old = time.time() - (studio.KEEP_HOURS + 2) * 3600
    (d / "job.json").write_text(json.dumps(
        {"source": "audio.mp3", "filename": "a song.mp3", "created": old,
         "video": "vid.mp4"}))
    (d / "status.json").write_text(json.dumps(
        {"state": "done",
         "parts": {"vocals": "vocals.mp3", "band": "band.mp3"},
         "levels": {"vocals": -1.0, "band": -2.0},
         "verdict": {"kind": "music with a voice in it", "status": {}},
         "videos": {"with_voice": "with-voice.mp4", "no_voice": "no-voice.mp4",
                    "subtitles": True},
         "kept": ["vocals", "band", "with_voice", "no_voice"]}))

    for name in ("out/vocals.mp3", "out/band.mp3", "out/with-voice.mp4",
                 "out/no-voice.mp4", "audio.mp3", "vid.mp4"):
        f = d / name
        f.write_bytes(b"x" * 4096)
        import os
        os.utime(f, (old, old))
    return jid, d


def test_the_media_is_actually_freed(job):
    jid, d = job
    assert studio.sweep() == 1
    assert not (d / "out" / "vocals.mp3").exists()
    assert not (d / "out" / "with-voice.mp4").exists()
    assert not (d / "audio.mp3").exists()


def test_a_swept_job_still_opens(job):
    """The one that was missing. Before the fix this returned state=unknown,
    which the page renders as 'that project is gone'."""
    jid, d = job
    studio.sweep()
    described = studio.describe(studio.Job(jid, d))
    assert described["state"] == "done", described
    assert described["found"]["vocals"]["file"] == "vocals.mp3"
    assert described["videos"]["with_voice"] == "with-voice.mp4"


def test_a_swept_job_is_still_in_your_projects(job):
    jid, _ = job
    studio.sweep()
    assert [j["id"] for j in studio.recent()] == [jid]


def test_the_index_survives_but_nothing_else_does(job):
    jid, d = job
    studio.sweep()
    left = sorted(f.name for f in d.rglob("*") if f.is_file())
    assert left == ["job.json", "status.json"], left


def test_a_file_restored_from_storage_is_not_swept_out_from_under_the_listener(job):
    """Sweeping is by each file's OWN age. A stem pulled back from S3 thirty
    seconds ago belongs to somebody who is playing it right now."""
    jid, d = job
    studio.sweep()
    restored = d / "out" / "vocals.mp3"
    restored.write_bytes(b"y" * 4096)          # as fetch_s3 would leave it
    studio.sweep()
    assert restored.exists(), "swept a file that had just been restored"


def test_an_unfinished_job_directory_is_left_alone(tmp_path, monkeypatch):
    """No job.json means the upload never completed. `abandon` owns that path,
    not this one."""
    monkeypatch.setattr(studio, "JOBS", tmp_path)
    d = tmp_path / "partial00000"
    d.mkdir()
    (d / "source.mp3").write_bytes(b"x" * 10)
    assert studio.sweep() == 0
    assert (d / "source.mp3").exists()


def test_the_projects_list_can_tell_two_runs_of_the_same_link_apart(tmp_path, monkeypatch):
    """Name and kind alone are not enough, and that is not hypothetical.

    Bruno ran the same link twice, once before a fix and once after. His list
    showed two rows reading "Metric - Black Sheep" / "music with a voice in
    it". He opened the older one, saw no subtitles, and asked whether the
    feature was broken and whether he should redo the link. Nothing on screen
    could have told him which was which.
    """
    monkeypatch.setattr(studio, "JOBS", tmp_path)
    for jid, subs, created in (("aaaaaaaaaaaa", False, 1000.0),
                               ("bbbbbbbbbbbb", True, 2000.0)):
        d = tmp_path / jid
        (d / "out").mkdir(parents=True)
        (d / "job.json").write_text(json.dumps(
            {"source": "a.mp3", "filename": "same song.mp3", "created": created}))
        (d / "status.json").write_text(json.dumps(
            {"state": "done", "parts": {"vocals": "vocals.mp3"},
             "verdict": {"kind": "music with a voice in it", "status": {}},
             "videos": {"with_voice": "with-voice.mp4", "subtitles": subs}}))

    rows = {j["id"]: j for j in studio.recent()}
    assert rows["aaaaaaaaaaaa"]["name"] == rows["bbbbbbbbbbbb"]["name"]
    assert rows["aaaaaaaaaaaa"]["kind"] == rows["bbbbbbbbbbbb"]["kind"]
    # ...so something else has to differ, or the list is unusable.
    assert rows["bbbbbbbbbbbb"]["subtitles"] is True
    assert rows["aaaaaaaaaaaa"]["subtitles"] is False
    assert rows["aaaaaaaaaaaa"]["created"] != rows["bbbbbbbbbbbb"]["created"]
    assert all(r["video"] for r in rows.values())


def test_a_title_is_not_a_filename():
    """Bruno, 2026-09-04: "Why the whole link."

    The projects list was printing the raw filename, extension and all, wrapping
    over two lines. A link job is saved as f"{title}.mp3" and archive titles
    routinely end in ".mp4" themselves, so the file on disk is
    "Beck - Ramona (Lyrics + HD).mp4.mp3" and one strip is not enough.
    """
    t = studio.title_of
    assert t("Metric - Black Sheep.mp3") == "Metric - Black Sheep"
    assert t("Beck - Ramona (Lyrics + HD).mp4.mp3") == "Beck - Ramona (Lyrics + HD)"
    # a name that merely LOOKS like it has an extension keeps it
    assert t("Blue Monday 88") == "Blue Monday 88"
    assert t("a.b.c.mp3") == "a.b.c"
    assert t("") == "(unnamed)" and t(None) == "(unnamed)"
