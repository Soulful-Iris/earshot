#!/usr/bin/env python3.12
"""Render a page and look at it, at the size it will actually be read at.

    tools/look.py https://earshot.soulful-ai.dev/ out.png [width] [height]

This exists because of a specific run of failures on 2026-09-01. I shipped a
project list to a page, verified it with curl, and shipped it to the one person
it was for -- who was signed in and therefore served a completely different
file. Then he reported the studio's UI was broken on his phone and I could not
say whether it was, because in a whole day of building web pages I had never
once seen one.

There is no browser on this box and no sudo. Chromium comes from playwright;
the twelve X11 and GBM libraries it needs are unpacked from RPMs into
~/sysroot with rpm2cpio and reached through LD_LIBRARY_PATH, which needs no
root at all. `sudo playwright install-deps` is what the error message tells you
to run and it is not the only way.

Default size is 390x844: an iPhone, which is where he reads everything.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SYSROOT = Path.home() / "sysroot"
LIBS = f"{SYSROOT}/usr/lib64:{SYSROOT}/lib64"


def ensure_libs() -> None:
    """Put the unpacked libraries on the path before chromium is started.

    Set here rather than left to the caller's shell: a tool that only works
    when you remember to export something is a tool that will be reported
    broken by the next person, who will be me.
    """
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if LIBS not in current:
        os.environ["LD_LIBRARY_PATH"] = f"{LIBS}:{current}" if current else LIBS


def shot(url: str, out: Path, width: int = 390, height: int = 844,
         full: bool = True, wait_for: str | None = None) -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(chromium_sandbox=False)
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)
        problems: list[str] = []
        page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.goto(url, wait_until="networkidle", timeout=60000)
        if wait_for:
            page.wait_for_selector(wait_for, timeout=120000)
        page.screenshot(path=str(out), full_page=full)

        # Anything wider than the viewport is the commonest way a page is
        # "messed up" on a phone, and it is invisible in a screenshot of the
        # top of the page.
        overflow = page.evaluate("""() => {
            const bad = [];
            for (const el of document.querySelectorAll('*')) {
                const r = el.getBoundingClientRect();
                if (r.width > window.innerWidth + 1 && r.width > 0)
                    bad.push(el.tagName.toLowerCase() +
                             (el.id ? '#' + el.id : '') +
                             (el.className && typeof el.className === 'string'
                              ? '.' + el.className.trim().split(/\\s+/).join('.') : '') +
                             ' ' + Math.round(r.width) + 'px');
            }
            return [...new Set(bad)].slice(0, 12);
        }""")
        browser.close()

    if overflow:
        print(f"  WIDER THAN THE {width}px SCREEN:")
        for o in overflow:
            print(f"    {o}")
    for pr in problems[:8]:
        print(f"  {pr}")
    return out


def main() -> int:
    ensure_libs()
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 2
    url, out = sys.argv[1], Path(sys.argv[2])
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 390
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 844
    shot(url, out, w, h)
    print(f"  wrote {out} ({out.stat().st_size // 1024} KB) at {w}x{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
