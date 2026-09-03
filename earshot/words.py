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

So it works, and it fails honestly: some singing genuinely cannot be read, and
this says so rather than inventing a line.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

# `language=multi`, NOT `detect_language=true`, and this is the most important
# line in the file.
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
            "?model=nova-3&punctuate=true&words=true&language=multi")

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


def read_vocal(path: str | Path, timeout: int = 300) -> Lyrics:
    """Transcribe an isolated vocal stem into timed words."""
    data = Path(path).read_bytes()
    req = urllib.request.Request(
        DEEPGRAM, data=data,
        headers={"Authorization": f"Token {_key()}",
                 "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())

    try:
        ch = d["results"]["channels"][0]
        alt = ch["alternatives"][0]
    except (KeyError, IndexError) as e:
        raise Unreadable(f"the recogniser returned nothing usable: {e}") from e

    raw = alt.get("words") or []
    if not raw:
        # Deliberately narrower than it used to be. The old message asserted a
        # cause - "some singing genuinely cannot be read, heavy effects, a vocal
        # buried in the mix" - and that cause was wrong: the real reason was a
        # query parameter, and the words came back the moment it changed. So
        # this now reports the OUTCOME and stops, because I do not know why.
        raise Unreadable(
            "no words came back from the vocal. The backing track is fine; there "
            "just will not be a highlighter for this one.")

    words = [Word(w["word"], float(w["start"]), float(w["end"]),
                  float(w.get("confidence", 0.0))) for w in raw]
    mean = sum(w.confidence for w in words) / len(words)
    unsure = sum(1 for w in words if w.confidence < MIN_CONFIDENCE)
    return Lyrics(words=words, language=ch.get("detected_language"),
                  mean_confidence=mean, unsure=unsure)


def save(lyrics: Lyrics, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(lyrics.as_dict(), indent=1))
    return p
