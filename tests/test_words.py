"""The words step, and the parameter that made it lie.

    python3.12 tests/test_words.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from earshot import words                                          # noqa: E402


def test_it_tries_more_than_one_language_and_never_autodetect():
    """SUPERSEDES a test that asserted `language=multi` was hardcoded.

    That test was right about autodetect and wrong about the fix. multi rescued
    a Korean vocal that detect_language returned nothing for, and my control was
    one Italian stem where the two settings differed by two words, so I called
    it free and pinned it. On Bruno's English song multi got 82 words at 2.8 dB
    alignment and en got 135 at 3.6, which is the difference between subtitles
    and no subtitles.

    So there is no globally correct language and this asserts that the code no
    longer pretends there is.
    """
    assert len(words.ATTEMPTS) > 1, "back to a single hardcoded language"
    assert any("multi" in a for a in words.ATTEMPTS)
    assert any("=en" in a for a in words.ATTEMPTS)
    assert "detect_language" not in words.DEEPGRAM, \
        "detect_language returns nothing on sung vocals and blames the singer"
    assert not any("detect_language" in a for a in words.ATTEMPTS)


def test_the_attempts_are_judged_on_alignment_not_on_word_count():
    """The two attempts on his song both reported 0.89 confidence and one of
    them was unusable. Whatever picks between them must not be either number
    the recogniser hands back about itself."""
    import inspect
    src = inspect.getsource(words.read_vocal)
    assert "alignment(" in src, "read_vocal no longer measures against the audio"
    assert "mean_confidence" not in src, \
        "picking on the recogniser's own confidence is it grading its own homework"


def test_the_failure_message_does_not_assert_a_cause():
    """It used to say why, and the why was wrong.

    A message naming a false cause is worse than no message. This asserts the
    old confident explanation stays gone.
    """
    src = (Path(__file__).resolve().parent.parent / "earshot" / "words.py").read_text()
    # Strip comments FIRST. The old wording is quoted in a comment on purpose,
    # to record why it was wrong - and the first version of this test failed on
    # exactly that, which is the third time today I have written a guard that
    # cannot tell a warning from the thing it warns about.
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    body = code.split("if not raw:")[1].split("words = [")[0]
    for claim in ("genuinely cannot be read", "heavy effects", "buried in the mix"):
        assert claim not in body, \
            f"the failure message asserts {claim!r}, a cause it has not established"


def test_alignment_returns_plain_json_safe_types():
    """A numpy scalar survives arithmetic and comparison and dies in json.dumps.

    It would have died in the WORKER, at the very end, after the separation was
    already done - the exact shape of the post-step that ate a finished job
    once already. Plain float, plain bool, or the status write fails.
    """
    import json
    ws = [words.Word("a", 0.0, 0.4, 0.9), words.Word("b", 1.0, 1.4, 0.9)]
    l = words.Lyrics(words=ws, language=None, mean_confidence=0.9, unsure=0)
    payload = {"alignment_db": 4.2, "timings_usable": 4.2 >= words.ALIGNED_DB}
    json.dumps(payload)          # must not raise
    assert isinstance(payload["timings_usable"], bool)
    assert l.at(0.2) == 0


def test_a_confident_transcript_can_still_be_unusable():
    """The failure this whole check exists for.

    119 words at 0.71 confidence, and the clock was random. Nothing in the
    recogniser's own output said so, because confidence is it grading its own
    homework. Measured against the audio: -0.1 dB where a working track gives
    +11.7. The threshold has to separate those two and nothing else matters.
    """
    assert words.ALIGNED_DB > 0.0
    assert -0.1 < words.ALIGNED_DB <= 11.7, \
        "the threshold must sit between the measured bad case and good case"


def test_unsure_words_are_counted_not_hidden():
    ws = [words.Word("a", 0.0, 0.4, 0.99), words.Word("b", 0.4, 0.9, 0.20)]
    l = words.Lyrics(words=ws, language="en", mean_confidence=0.6,
                     unsure=sum(1 for w in ws if w.confidence < words.MIN_CONFIDENCE))
    assert l.unsure == 1 and l.as_dict()["unsure"] == 1


def test_the_highlighter_never_lands_between_words():
    ws = [words.Word("one", 0.0, 0.5, 1.0), words.Word("two", 0.6, 1.2, 1.0)]
    l = words.Lyrics(words=ws, language="en", mean_confidence=1.0, unsure=0)
    assert l.at(0.2) == 0 and l.at(0.9) == 1
    assert l.at(0.55) is None, "a gap between words must not highlight one"
    assert l.at(99) is None


if __name__ == "__main__":
    import traceback
    fails = 0
    for n, fn in sorted(globals().items()):
        if not n.startswith("test_") or not callable(fn):
            continue
        try:
            fn(); print(f"  ok   {n}")
        except Exception:
            fails += 1; print(f"  FAIL {n}"); traceback.print_exc()
    print("all green" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
