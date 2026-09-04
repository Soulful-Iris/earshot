#!/usr/bin/env python3.12
"""Drive the karaoke stage end to end in a real browser and look at it.

    tools/drive_stage.py [job_id]

Not a pytest, and the reason is the whole point of the file: THIS BOX HAS NO
MICROPHONE. `navigator.mediaDevices.enumerateDevices()` returns zero devices of
any kind in playwright's chromium, and `--use-fake-device-for-media-capture`
plus `--use-file-for-fake-audio-capture` do not conjure one. So the hardware is
the single thing that cannot be exercised here.

Everything downstream of the stream object CAN be. The init script below builds
a real MediaStream in the page from an oscillator with a slow LFO on its gain,
and hands it over where getUserMedia would answer. The analyser, the meter, the
countdown, MediaRecorder, the lyric wipe and the upload are all the real code
path, fed by a real stream that nobody spoke into.

What it prints is the evidence: how many meter bars the level lit, how many
lyric lines loaded, and at two different moments how many words are sung, which
one is mid-wipe and how far through it is. Screenshots land in /tmp/kt.
"""
import sys, time; sys.path.insert(0,'tools')
import look; look.ensure_libs()
from playwright.sync_api import sync_playwright

FAKE_MIC = """
// A real MediaStream, generated in the page, handed over where the microphone
// would be. This box's chromium reports ZERO media devices, so the hardware is
// the one thing that cannot be exercised here; everything downstream of the
// stream object is the real code path.
(() => {
  const ctx = new AudioContext();
  const dst = ctx.createMediaStreamDestination();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const lfo = ctx.createOscillator(), lg = ctx.createGain();
  osc.frequency.value = 220; lfo.frequency.value = 0.7; lg.gain.value = 0.45;
  gain.gain.value = 0.5;
  lfo.connect(lg); lg.connect(gain.gain);
  osc.connect(gain); gain.connect(dst);
  osc.start(); lfo.start();
  navigator.mediaDevices.getUserMedia = async () => dst.stream;
})();
"""
JOB = sys.argv[1] if len(sys.argv) > 1 else "f615b24d9e43"

with sync_playwright() as pw:
    b = pw.chromium.launch(chromium_sandbox=False,
                           args=["--autoplay-policy=no-user-gesture-required"])
    ctx = b.new_context(viewport={"width":390,"height":844}, device_scale_factor=2)
    ctx.add_init_script(FAKE_MIC)
    pg = ctx.new_page()
    errs=[]
    pg.on("pageerror", lambda e: errs.append("pageerror: "+str(e)))
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type=="error" else None)
    pg.goto(f"http://localhost:8781/#{JOB}", wait_until="networkidle", timeout=60000)
    pg.wait_for_selector("#singgo", timeout=30000)
    pg.click("#singgo")
    pg.wait_for_selector("#meter i", timeout=20000)
    time.sleep(1.6)
    lit = pg.eval_on_selector_all("#meter i.lit, #meter i.hot", "e => e.length")
    print(f"  READY: meter lit {lit}/28  |  {pg.text_content('#mtxt')}")
    print(f"  words loaded: {pg.evaluate('() => stageWords ? stageWords.length : null')} lines")
    pg.screenshot(path="/tmp/kt/s_ready.png")

    pg.click("#stagestart")
    time.sleep(0.5)
    print("  COUNTDOWN:", (pg.text_content(".cd") or "").strip())
    pg.screenshot(path="/tmp/kt/s_count.png")

    pg.wait_for_selector("#lyr", timeout=20000)
    time.sleep(3.0)
    for label in ("A","B"):
        st = pg.evaluate("""() => ({
          t:+(bandEl?bandEl.currentTime:0).toFixed(2),
          cur:(document.querySelector('.lyr .cur')||{}).textContent,
          done:document.querySelectorAll('.lyr w.done').length,
          now:document.querySelectorAll('.lyr w.now').length,
          todo:document.querySelectorAll('.lyr w:not(.done):not(.now)').length,
          p:(document.querySelector('.lyr w.now')||{style:{getPropertyValue:()=>''}}).style.getPropertyValue('--p'),
          clock:(document.getElementById('pclock')||{}).textContent,
          rec: mediaRec?mediaRec.state:null})""")
        print(f"  SINGING {label}: t={st['t']}s rec={st['rec']} clock={st['clock']}")
        print(f"     line: {st['cur']!r}")
        print(f"     done={st['done']} now={st['now']} todo={st['todo']} wipe={st['p']}")
        pg.screenshot(path=f"/tmp/kt/s_sing{label}.png")
        time.sleep(2.2)
    print("  errors:", errs or "none")
    b.close()
