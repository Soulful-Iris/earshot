#!/usr/bin/env python3.12
"""Build test material whose answer is known before anything measures it.

Two kinds, and both are needed:

  synthetic  speech mixed under a bed at a ratio I chose. The ONLY way to ask
             "does the meter recover the number I put in", which is the check
             that can go red. Speech comes from Deepgram TTS so the dry signal
             is clean and I own the timing of every word.

  real       a public-domain human reading from LibriVox. Every synthetic
             fixture is speech I generated, and a corpus I built has a shape --
             last night's bug was exactly a corpus whose shape excluded the
             real case. This is the control on my own fixtures.

Usage:  tools/make_fixtures.py [outdir]
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earshot.loudness import SR, lufs_of  # noqa: E402

LINES = [
    "The thing about the harbour is that nobody ever agreed where it ended.",
    "She put the letter down and did not pick it up again for eleven years.",
    "We can be there before dark if we leave now and you stop arguing.",
    "There were four of them at the table and only one of them was telling the truth.",
    "It rained for nine days and on the tenth the river came up through the floor.",
]

# A public-domain human voice. LibriVox recordings are public domain worldwide.
LIBRIVOX_MP3 = ("https://www.archive.org/download/"
                "aesop_fables_volume_one_librivox/fables_01_00_aesop_64kb.mp3")


def deepgram_key() -> str:
    for cmd in (["aws", "ssm", "get-parameter", "--region", "us-east-1",
                 "--name", "/soulful/iris/deepgram_key", "--with-decryption",
                 "--query", "Parameter.Value", "--output", "text"],
                ["aws", "secretsmanager", "get-secret-value", "--region", "us-east-1",
                 "--secret-id", "soulful/iris/deepgram_key",
                 "--query", "SecretString", "--output", "text"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise SystemExit("no deepgram key")


HELDOUT_LINES = [
    "Nobody signs for the boxes any more, they just leave them by the gate.",
    "I told him the tide would turn at four and he went out anyway.",
    "Every window on that side of the building faces a wall.",
    "You can hear the trains from the kitchen if the door is open.",
]


def rough_bed(n: int, seed: int = 23) -> np.ndarray:
    """A different kind of background: broadband, percussive, no slow swell.

    The tuned fixtures all use a smooth pink-plus-chord bed, so a bed built the
    same way would inherit whatever the sweep learned about that shape.
    """
    from scipy import signal as sg
    rng = np.random.default_rng(seed)
    noise = sg.lfilter([0.5], [1.0, -0.5], rng.standard_normal(n))
    t = np.arange(n) / SR
    hits = np.zeros(n)
    for k in range(0, n, int(0.75 * SR)):                  # a beat every 750 ms
        env = np.exp(-np.arange(min(int(0.25 * SR), n - k)) / (0.04 * SR))
        hits[k:k + len(env)] += env * rng.uniform(0.5, 1.0)
    return noise / (np.abs(noise).max() + 1e-9) * 0.5 + hits * 0.5


def tts_voice(text: str, out: Path, key: str, model: str) -> None:
    req = urllib.request.Request(
        f"https://api.deepgram.com/v1/speak?model={model}"
        "&encoding=linear16&sample_rate=48000&container=wav",
        data=json.dumps({"text": text}).encode(),
        headers={"Authorization": f"Token {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out.write_bytes(r.read())


def read_mono(path: Path) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"],
        capture_output=True, timeout=300, check=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)


def write_wav(path: Path, mono: np.ndarray) -> None:
    stereo = np.repeat(mono[:, None], 2, axis=1).astype(np.float32)
    write_stereo(path, stereo)


def write_stereo(path: Path, stereo: np.ndarray) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR),
         "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", str(path)],
        input=np.ascontiguousarray(stereo, dtype=np.float32).tobytes(),
        check=True, timeout=300)


def widen(mono: np.ndarray, seed: int) -> np.ndarray:
    """Spread a signal across the stereo field the way a score is spread.

    Every other fixture in this file is one mono signal copied into both
    channels, which means the voice and the bed are equally centred and NOTHING
    that works on stereo position has anything to work with. That is not how a
    film is mixed: dialogue is anchored in the centre and the music is wide, and
    a centre-extraction filter exists precisely to exploit the difference.

    Measured consequence of not having this: ffmpeg's `dialoguenhance` scored
    +0.01 dB on the mono-in-stereo fixtures and I was about to write down that
    it does nothing. It had nothing to do.

    Decorrelated by short, different delays and opposed comb filtering per
    channel -- crude next to a real mix, and enough to make left and right
    genuinely different signals.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((len(mono), 2))
    for ch in range(2):
        d = int(rng.integers(120, 480))                  # 2.5-10 ms
        delayed = np.concatenate([np.zeros(d), mono])[:len(mono)]
        out[:, ch] = 0.7 * mono + 0.5 * (delayed if ch == 0 else -delayed)
    return out / max(np.abs(out).max(), 1e-9) * np.abs(mono).max()


def at_lufs(mono: np.ndarray, target: float) -> np.ndarray:
    """Scale a mono signal so a stereo copy of it measures `target` LUFS."""
    now = lufs_of(np.repeat(mono[:, None], 2, axis=1))
    if not np.isfinite(now):
        raise ValueError("signal is below the absolute gate; cannot be scaled to a target")
    return mono * 10.0 ** ((target - now) / 20.0)


def music_bed(n: int, seed: int = 7) -> np.ndarray:
    """Something with the spectral shape of a score, not white noise.

    Pink-ish noise plus a slow chord. White noise would be trivially separable
    from speech by any energy measure and would flatter every result.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    # one-pole lowpass twice: rolls off like a music bed rather than hiss
    b, a = [0.02], [1.0, -0.98]
    from scipy import signal as sg
    pink = sg.lfilter(b, a, sg.lfilter(b, a, white))
    t = np.arange(n) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (110.0, 164.81, 220.0)) / 3.0
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t)
    return (pink / (np.abs(pink).max() + 1e-9) * 0.6 + chord * 0.4) * swell


def build_speech_track(key: str, tmp: Path, gap_s: float = 1.2) -> np.ndarray:
    """Lines with silence between them, so 'background' is a measurable state."""
    parts = []
    gap = np.zeros(int(gap_s * SR))
    parts.append(gap)
    for i, line in enumerate(LINES):
        wav = tmp / f"line{i}.wav"
        if not wav.exists():
            tts_voice(line, wav, key, "aura-2-thalia-en")
        parts.extend([read_mono(wav), gap])
    return np.concatenate(parts)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               Path(__file__).resolve().parents[1] / "fixtures")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_raw"
    tmp.mkdir(exist_ok=True)

    speech = build_speech_track(deepgram_key(), tmp)
    n = len(speech)
    bed = music_bed(n)

    # Speech is pinned at -23 LUFS (the EBU R128 delivery target) and the bed
    # moves, so the recorded truth is one number per fixture.
    speech = at_lufs(speech, -23.0)
    manifest = {"speech_lufs_target": -23.0, "cases": []}

    # The stems are kept beside every mix. Without them the only truth available
    # is "the gap I asked for", and that turns out to be a different quantity
    # from the one the tool measures -- see tools/recover.py.
    stems = out / "stems"
    stems.mkdir(exist_ok=True)

    for gap in (25, 15, 10, 5, 0, -5):
        bed_target = -23.0 - gap
        scaled_bed = at_lufs(bed, bed_target)
        mixed = speech + scaled_bed
        peak = np.abs(mixed).max()
        trim = 1.0
        if peak > 0.99:                       # never let the fixture clip
            trim = 0.99 / peak
            mixed = mixed * trim
        name = f"gap{gap:+03d}.wav"
        write_wav(out / name, mixed)
        write_wav(stems / f"{name[:-4]}.voice.wav", speech * trim)
        write_wav(stems / f"{name[:-4]}.bed.wav", scaled_bed * trim)
        manifest["cases"].append(
            {"file": name, "speech_minus_bed_lu": gap, "bed_lufs": bed_target,
             "voice_stem": f"stems/{name[:-4]}.voice.wav",
             "bed_stem": f"stems/{name[:-4]}.bed.wav"})
        print(f"  {name}  speech -23.0 LUFS, bed {bed_target:.1f} LUFS, gap {gap:+d} LU")

    # Speech with nothing under it: the meter must report the bed as unmeasurable
    # rather than inventing a floor.
    write_wav(out / "dry.wav", speech)
    manifest["cases"].append({"file": "dry.wav", "speech_minus_bed_lu": None,
                              "bed_lufs": None})
    print("  dry.wav   speech only, no bed")

    # Real human voice, public domain, as the control on my own fixtures.
    real = out / "real_voice.mp3"
    if not real.exists():
        try:
            urllib.request.urlretrieve(LIBRIVOX_MP3, real)
            print(f"  real_voice.mp3  {real.stat().st_size // 1024} KB from LibriVox")
        except Exception as e:                # noqa: BLE001
            print(f"  real_voice.mp3  NOT fetched: {e}")

    # And the same known-ratio test built on that human voice. Every fixture
    # above is speech I generated, at a level I set, with silences I placed --
    # a corpus with a shape. Last night's undetectable bug was a corpus whose
    # shape excluded the real case, so the real case is now in the corpus.
    if real.exists():
        print("\n  real-voice mixes:")
        voice = read_mono(real)[: int(120 * SR)]        # two minutes is plenty
        voice = at_lufs(voice, -23.0)
        rbed = music_bed(len(voice), seed=11)
        for gap in (15, 8, 4):
            scaled = at_lufs(rbed, -23.0 - gap)
            mixed = voice + scaled
            peak = np.abs(mixed).max()
            trim = 1.0
            if peak > 0.99:
                trim = 0.99 / peak
                mixed = mixed * trim
            name = f"real_gap{gap:+03d}.wav"
            write_wav(out / name, mixed)
            write_wav(stems / f"{name[:-4]}.voice.wav", voice * trim)
            write_wav(stems / f"{name[:-4]}.bed.wav", scaled * trim)
            manifest["cases"].append(
                {"file": name, "speech_minus_bed_lu": gap,
                 "bed_lufs": -23.0 - gap, "voice": "human",
                 "voice_stem": f"stems/{name[:-4]}.voice.wav",
                 "bed_stem": f"stems/{name[:-4]}.bed.wav"})
            print(f"  {name}  human voice -23.0 LUFS, bed {-23.0-gap:.1f}, gap {gap:+d} LU")

    # HELD OUT. Three constants in measure.py (the guard, the window width, the
    # minimum gap sample) were chosen by sweeping against everything above, so
    # everything above is now a best case rather than a test. These are a
    # different voice, a different kind of bed and ratios that were not in the
    # sweep, generated once and never tuned against. The harness scores them
    # separately and that score is the one to believe.
    print("\n  held out:")
    held = []
    key = deepgram_key()
    for i, line in enumerate(HELDOUT_LINES):
        w = tmp / f"held{i}.wav"
        if not w.exists():
            tts_voice(line, w, key, "aura-2-orion-en")
        held.extend([read_mono(w), np.zeros(int(0.9 * SR))])
    hv = at_lufs(np.concatenate(held), -23.0)
    hbed = rough_bed(len(hv))
    for gap in (6.5, 12.0):
        scaled = at_lufs(hbed, -23.0 - gap)
        mixed = hv + scaled
        peak = np.abs(mixed).max()
        trim = 0.99 / peak if peak > 0.99 else 1.0
        mixed = mixed * trim
        name = f"held_gap{gap:+05.1f}.wav"
        write_wav(out / name, mixed)
        write_wav(stems / f"{name[:-4]}.voice.wav", hv * trim)
        write_wav(stems / f"{name[:-4]}.bed.wav", scaled * trim)
        manifest["cases"].append(
            {"file": name, "speech_minus_bed_lu": gap, "bed_lufs": -23.0 - gap,
             "held_out": True, "voice": "aura-2-orion",
             "voice_stem": f"stems/{name[:-4]}.voice.wav",
             "bed_stem": f"stems/{name[:-4]}.bed.wav"})
        print(f"  {name}  different voice, broadband bed, gap {gap:+.1f} LU")

    # STEREO, the way a film is actually mixed: voice anchored centre, bed wide.
    # Without these the corpus cannot tell a separator from a volume knob.
    print("\n  stereo (centred voice, wide bed):")
    for gap in (10, 4, 0):
        v = at_lufs(speech, -23.0)
        b = at_lufs(bed, -23.0 - gap)
        vs = np.repeat(v[:, None], 2, axis=1)             # dead centre
        bs = widen(b, seed=41 + gap)                      # spread
        mixed = vs + bs
        peak = float(np.abs(mixed).max())
        trim = 0.99 / peak if peak > 0.99 else 1.0
        name = f"wide_gap{gap:+03d}.wav"
        write_stereo(out / name, mixed * trim)
        write_stereo(stems / f"{name[:-4]}.voice.wav", vs * trim)
        write_stereo(stems / f"{name[:-4]}.bed.wav", bs * trim)
        manifest["cases"].append(
            {"file": name, "speech_minus_bed_lu": gap, "bed_lufs": -23.0 - gap,
             "stereo": "centred voice, wide bed",
             "voice_stem": f"stems/{name[:-4]}.voice.wav",
             "bed_stem": f"stems/{name[:-4]}.bed.wav"})
        print(f"  {name}  voice centre, bed wide, gap {gap:+d} LU")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
