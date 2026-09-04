"""Word-by-word timings, read off the isolated vocal.

Bruno guessed this part would need a lyrics API, and asked whether DeepSeek
would help. It does not - that is a language model and it cannot hear. And no
lyrics database is needed either, which is the nice part: after separation we
are holding the singer's voice on its own, so we transcribe THAT. The timing
then comes out of the recording itself rather than out of somebody's guess at
where the lines fall.

MEASURED BEFORE BELIEVING IT, 2026-09-02. First test came back with zero words
and I nearly reported the whole idea impossible. The stem was at -22.1 LUFS
against -15.7 for the mix, so it was not silence - I had picked "Louie Louie",
which is famously the most unintelligible vocal ever recorded. A second, clearly
sung track returned 50 words with per-word start and end times at confidence
0.96 to 1.00.

TWO THINGS LEARNED SINCE, both by being wrong in public:

1. `detect_language=true` returns NOTHING on sung vocals. The failure message
   this file used to print blamed the singing, and that was false.
2. A transcript can come back full and CONFIDENT with timings that are noise.
   119 words at 0.71 confidence, and the clock was random. See `alignment()`
   at the bottom - the recogniser's own confidence is it grading its own
   homework, so the timings get checked against the audio instead.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

# SUPERSEDED, one day later, by ATTEMPTS below: there is no single right
# language setting and this file no longer picks one. The measurements are kept
# because they are why `detect_language=true` is not in the list at all.
#
# Measured 2026-09-03 on a real job that had just come back with zero words and
# the message "some singing genuinely cannot be read":
#
#     detect_language=true   ->    0 words
#     language=ko            ->  131 words
#     language=multi         ->  119 words
#
# The singing was perfectly readable. Deepgram's language AUTO-DETECTION is what
# fails on sung vocals, so my honest-sounding failure message was a FALSE
# EXPLANATION: it told somebody their song could not be read while the words sat
# behind a query parameter. A message naming a false cause is worse than no
# message, and this one read as humility, which is what made it convincing.
#
# Control, because swapping a parameter on the strength of one improvement is
# how you fix one case and break another. On the Italian stem that already
# worked: detect gave 50 words at 0.88 confidence, multi gives 48 at 0.88. Two
# words fewer where it worked, 119 recovered where it did not.
DEEPGRAM = ("https://api.deepgram.com/v1/listen"
            "?model=nova-3&punctuate=true&words=true")

# TRY BOTH AND KEEP WHICHEVER ACTUALLY LINES UP, rather than picking a language.
#
# This file used to hardcode `language=multi`. I chose that on 2026-09-03 to fix
# a Korean track that `detect_language=true` returned nothing for, and I did run
# a control before swapping -- an Italian stem, 50 words with detect against 48
# with multi. Two words apart, so I called it free.
#
# It was not free. Bruno put an English song through the next day and got no
# subtitles at all, because the gate correctly refused timings that did not
# track the singing. Measured on his vocal stem:
#
#     language=multi   82 words, alignment 2.8 dB   (below the 3.0 bar)
#     language=en     135 words, alignment 3.6 dB   (above it)
#
# Same audio, same confidence, 53 more words. My control was one song in a
# language neither setting struggles with, and it under-represented the cost by
# a mile. So there is no global right answer here and I should stop looking for
# one: run both, measure both against the audio with alignment(), keep the
# better. The measurement already exists and can already go red.
ATTEMPTS = ("language=multi", "language=en")

# Below this, the recogniser is guessing at a shape rather than hearing a word.
# A highlighter that lands on the wrong syllable is worse than no highlighter,
# because it is confidently wrong in front of somebody who is singing.
MIN_CONFIDENCE = 0.55


class Unreadable(Exception):
    """The vocal could not be read. Not a crash - an answer."""


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class Lyrics:
    words: list
    language: str | None
    mean_confidence: float
    unsure: int          # how many fell below MIN_CONFIDENCE
    tried: list | None = None   # what each attempt scored

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def as_dict(self) -> dict:
        return {"language": self.language,
                "mean_confidence": round(self.mean_confidence, 3),
                "unsure": self.unsure,
                "count": len(self.words),
                "words": [asdict(w) for w in self.words]}

    def at(self, t: float) -> int | None:
        """Index of the word being sung at time t, for the highlighter."""
        for i, w in enumerate(self.words):
            if w.start <= t <= w.end:
                return i
        return None


def _key() -> str:
    for cmd in (["aws", "secretsmanager", "get-secret-value", "--secret-id",
                 "soulful/iris/deepgram_key", "--region", "us-east-1",
                 "--query", "SecretString", "--output", "text"],
                ["aws", "ssm", "get-parameter", "--name",
                 "/soulful/iris/deepgram_key", "--with-decryption",
                 "--region", "us-east-1", "--query", "Parameter.Value",
                 "--output", "text"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise RuntimeError("no deepgram key in secretsmanager or ssm")


def _listen(path: Path, query: str, timeout: int) -> "Lyrics | None":
    """One transcription attempt. None when nothing usable came back."""
    req = urllib.request.Request(
        f"{DEEPGRAM}&{query}", data=Path(path).read_bytes(),
        headers={"Authorization": f"Token {_key()}", "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    try:
        ch = d["results"]["channels"][0]
        alt = ch["alternatives"][0]
    except (KeyError, IndexError):
        return None
    raw = alt.get("words") or []
    if not raw:
        return None
    ws = [Word(w["word"], float(w["start"]), float(w["end"]),
               float(w.get("confidence", 0.0))) for w in raw]
    mean = sum(w.confidence for w in ws) / len(ws)
    unsure = sum(1 for w in ws if w.confidence < MIN_CONFIDENCE)
    return Lyrics(words=ws, language=ch.get("detected_language"),
                  mean_confidence=mean, unsure=unsure)


def read_vocal(path: str | Path, timeout: int = 300) -> Lyrics:
    """Transcribe an isolated vocal stem into timed words.

    Runs every attempt in ATTEMPTS and keeps the one whose timings best track
    the audio. Judged by alignment() and NOT by word count or by the
    recogniser's own confidence, both of which said the two attempts on Bruno's
    song were equally good (0.89 either way) while one of them was unusable.
    """
    path = Path(path)
    best, best_al, tried = None, None, []
    for query in ATTEMPTS:
        try:
            lyr = _listen(path, query, timeout)
        except Exception as e:                                    # noqa: BLE001
            tried.append(f"{query}: {type(e).__name__}")
            continue
        if lyr is None:
            tried.append(f"{query}: nothing")
            continue
        al = alignment(path, lyr)
        tried.append(f"{query}: {len(lyr.words)}w align={al}")
        # None means not enough to compare on; keep it only if it is all we have.
        score = -99.0 if al is None else al
        if best is None or score > best_al:
            best, best_al = lyr, score
    if best is None:
        raise Unreadable(
            "no words came back from the vocal. The backing track is fine; "
            "there just will not be a highlighter for this one. "
            f"({'; '.join(tried)})")
    best.tried = tried
    return best


def save(lyrics: Lyrics, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(lyrics.as_dict(), indent=1))
    return p


def load(path: str | Path) -> Lyrics:
    """Read back what `save` wrote, so a video can be rebuilt without paying
    for the transcription again."""
    d = json.loads(Path(path).read_text())
    ws = [Word(w["word"], float(w["start"]), float(w["end"]),
               float(w.get("confidence", 0.0))) for w in d.get("words") or []]
    return Lyrics(words=ws, language=d.get("language"),
                  mean_confidence=float(d.get("mean_confidence") or 0.0),
                  unsure=int(d.get("unsure") or 0))


# --- does the transcript actually line up with the singing? ------------------
#
# The reason this exists, 2026-09-03. A job came back with 119 words at 0.71
# confidence and I reported it to Bruno as "119 words timed across 247 seconds".
# The count was real. The TIMINGS were noise - a highlighter driven by them
# would have landed randomly - and nothing in the recogniser's own output said
# so. Confidence is the recogniser grading its own homework.
#
# So this grades it against the audio instead: the vocal should be LOUD inside a
# word span and QUIET in the gaps between words. Measured:
#
#     a track that genuinely worked   +11.7 dB
#     the track I wrongly reported     -0.1 dB
#
# And the confound was checked before believing either: a stem with flat energy
# could not show alignment even if the timings were perfect. The failing stem
# has a 65 dB spread between its quiet and loud tenths of a second, so there was
# plenty to line up with. It simply did not.

ALIGNED_DB = 3.0   # below this the timings do not track the voice

_SR = 16000


def _mono(path) -> "object":
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
         "-ar", str(_SR), "-f", "f32le", "-"],
        capture_output=True, timeout=600).stdout
    return np.frombuffer(raw, dtype=np.float32)


def alignment(vocal_path, lyrics: Lyrics) -> float | None:
    """dB by which the vocal is louder inside words than between them.

    Positive and large means the timings track the singing. Near zero means the
    words may be right and the CLOCK is wrong, which is the failure that looks
    exactly like success from the transcript alone.

    None when there is not enough to compare - too few words, or no real gaps -
    because a number computed from two samples is worse than no number.
    """
    import numpy as np

    if not lyrics.words:
        return None

    def db(a):
        if a.size == 0:
            return None
        return 20 * np.log10(max(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))), 1e-9))

    v = _mono(vocal_path)
    if v.size < _SR:
        return None

    inside, gaps, prev = [], [], 0.0
    for w in lyrics.words:
        if w.start > prev + 0.15:
            gaps.append(v[int(prev * _SR):int(w.start * _SR)])
        inside.append(v[int(w.start * _SR):int(w.end * _SR)])
        prev = max(prev, w.end)

    ins = [x for x in inside if x.size]
    gap = [x for x in gaps if x.size]
    if not ins or not gap:
        return None
    a, b = db(np.concatenate(ins)), db(np.concatenate(gap))
    if a is None or b is None:
        return None
    # float(), not the numpy scalar. It survives arithmetic and comparison
    # perfectly and then dies in json.dumps as "Object of type bool is not JSON
    # serializable" - in the worker, at the very end, after the separation is
    # already done. Exactly the shape of the post-step that ate a finished job.
    return round(float(a) - float(b), 1)
