# earshot

**Is the dialogue actually audible, or is it you?**

Point it at a film, an episode, a podcast or a lecture recording and it tells
you how far the talking sits above everything under it, where in the runtime it
goes wrong, and — if you want — writes a re-balanced copy.

```
earshot episode.mkv
earshot https://example.com/episode.mp3 --json
earshot episode.mkv --fix
```

---

## Why this exists

Of 1,200 Americans surveyed, **78% say they have difficulty hearing dialogue
because of loud background music**, and **55%** say it is harder than it used to
be. About **half of all viewers now watch with subtitles on**; among under-30s
it is 63%. Of the people who use subtitles, **nearly three in four name the
mix**, not their own hearing, as the reason.

That is not a niche accessibility problem. It is most of the audience reading
television instead of listening to it, and treating the workaround as normal.

The causes are documented and technical rather than personal: television
speakers fire downward or backward, cinema mixes are not re-balanced for a
living room, and prestige drama routinely places music and effects at the level
of whispered dialogue. The industry has had measurement standards for this for
over a decade — BS.1770 gated loudness, speech-to-background ratio, the AES work
on dialogue intelligibility. **Nobody outside a mix suite is ever shown the
number.**

So: here is the number.

---

## What it measures

You cannot measure the speech alone in a finished mix — the music is under it.
But you can measure two things:

```
S = loudness while somebody is talking     (speech + bed)
B = loudness while nobody is               (bed alone)
```

Power adds, so `S = P_speech + P_bed` and `B = P_bed`, and therefore

```
speech-to-background  =  10·log10( 10^((S−B)/10) − 1 )
```

That is exact rather than approximate wherever the bed is doing something
similar under the speech to what it does between it. Where that assumption
breaks, it breaks in the direction you would want: a mix that ducks its score
under the dialogue measures as clearer, because it is.

Three numbers come out:

| | what it means |
|---|---|
| **ratio** | how far the dialogue sits above the background. Under about **+4 LU** is where people reach for the subtitles. |
| **swing** | how far the loudest non-speech moments sit **above** the dialogue. This is the turn-it-up-for-the-talking, down-for-the-explosions number. |
| **spread** | how much the dialogue's own level wanders. A big spread means quiet lines vanish even where the average looks fine. |

Plus the timestamps of the hardest moments, so you can go and check.

## Fixing it: two different things

### `--enhance` — separate the voice out, then set the ratio

This is the one that changes the number. It pulls the voice away from
everything else with a source-separation model, then puts the background back
at a distance you choose:

```
earshot episode.mkv --enhance --to 12     # voice 12 LU above everything else
earshot episode.mkv --voice-only          # just the voice, nothing under it
```

`--to` is the whole point. All the measurement above produces a **verdict** —
here is how buried your dialogue is, sorry. Separation turns it into a **dial**.

Measured against the ground-truth stems, on 20-second excerpts:

| | SI-SDR of the mix | after separation | gain |
|---|---|---|---|
| voice level with bed (wide, +0 LU) | −1.28 | **+14.82** | +16.10 |
| wide, +4 LU | 1.94 | **+16.34** | +14.39 |
| wide, +10 LU | 8.58 | **+20.32** | +11.75 |
| human reading, +4 LU | 0.93 | **+10.41** | +9.47 |
| held-out voice and bed, +6.5 LU | 7.41 | **+16.65** | +9.24 |

**Mean gain +12.2 dB.** The voice survives at its original level (−24.7 dB
against −24.4 in the mix) while the background drops 17 to 33 dB — which is the
shape of a separation rather than of deleting everything, and the reason both
columns are printed.

For comparison, every filter ffmpeg already ships scores **+0.01 dB** on the
same test. `dialoguenhance` raises the voice by 9 dB and the bed by 9 dB; it is
a volume knob with a good name.

**Cost, measured on this machine** (2 cores, no GPU, 1.8 GB): about 4× realtime
and 1.3 GB peak, which is why separation streams in overlapping windows rather
than loading the programme.

### `--fix` — level only, no separation

Lifts quiet dialogue toward a consistent level and pulls loud non-speech
material down toward it, on a 100 ms grid, smoothed fast-down and slow-up so it
does not pump. Night mode done with a speech detector instead of a broadband
compressor. It improves `swing` and `spread` and **cannot** move `ratio`,
because it never separates anything. On a two-minute human reading under a bed
at +4 LU: spread 7.9 → 3.7, and a recogniser's word error rate against the
clean stem went **0.117 → 0.004**.

> **Superseded, and kept because being wrong about it was the interesting part.**
> This section used to read: *"It cannot turn the music down under the voice.
> They are one signal by the time anyone outside the mix suite hears them;
> separating them needs a source-separation model and a much bigger machine."*
>
> That was written without checking anything. Bruno read it and said he would
> have found it more impressive if the thing could actually remove the
> background and enhance the voice. Finding out took under an hour: the torch
> aarch64 wheel is 408 MB, demucs 4.0.1 installs, and separation runs on this
> box. A constraint I had asserted rather than measured was doing real work in a
> public document, which is exactly the failure the rest of this README is about.

---

## Whether to believe any of it

Every number above is produced by an instrument I wrote, which is exactly the
situation in which instruments agree with whoever built them. So:

**`tools/crosscheck_loudness.py`** — the BS.1770 implementation against
ffmpeg's `ebur128`, on five signals chosen to be able to disagree (a relative
gate that must discard a quiet half, true digital silence between bursts, a
level near the absolute gate). Worst disagreement **0.046 LU** across a 34 LU
spread, plus one case where both correctly report nothing measurable, plus a
1 kHz tone checked against arithmetic rather than against either implementation.

**`tools/recover.py`** — the estimator against mixes built at a ratio chosen
before anything measured them, with both stems kept so the truth is
recomputable. Worst error **2.1 LU**, mean under 1 LU. Three constants were
swept against those fixtures, so a **held-out** pair — different voice, a
percussive rather than a tonal bed, ratios never swept — is scored separately:
**0.91 LU**.

**`tools/intelligibility.py`** — the only check here with anything
listener-shaped in it. A speech recogniser transcribes the clean voice stem, the
mix, and the re-balanced mix, and scores word error rate. It is the one
instrument in the project that was not built by me and does not measure level.

**`tools/make_fixtures.py`** — builds all of the above, including a
public-domain human reading from LibriVox as a control on my own generated
speech. That control earned its place immediately: the synthetic corpus passed
while the human voice was out by 3 LU.

### Known limits, stated because they are the interesting part

- **A mix whose voice sits at or below its bed cannot be measured this way.**
  The inversion amplifies error by `10^(d/10)/(10^(d/10)−1)`, which is a factor
  of four once the two levels are within a dB. Below that separation the tool
  refuses rather than printing an unstable number.
- **Where the score rises in the gaps between lines — which is what scoring
  *is* — the background is over-estimated and the ratio reads worse than it
  sounds.** The report's quartile band widens when this is happening.
- **The recogniser is too good to be an ear.** It reads speech buried 5 LU under
  a bed with no errors at all, so most fixtures sit at a word error rate of zero
  before anything is done to them and can only stay there. It can prove the fix
  does no harm and it cannot grade the easy cases.
- Everything is measured on a stereo downmix, deliberately: that is what comes
  out of a television.

---

## Installing

Needs `ffmpeg`, and Python 3.12 with `numpy`, `scipy` and `onnxruntime`.

```
pip install numpy scipy onnxruntime
curl -Lo models/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
python -m earshot.cli FILE
```

Silero VAD is MIT-licensed. Note that the model wants the last 64 samples of the
previous window prepended to the current one — without that it returns about
0.001 for every window of clean loud speech and reports, with no error anywhere,
that nobody is talking.
