"""Turning the measurement into sentences a person would say.

The numbers are LU and LUFS, which mean nothing outside a mix suite, and the
audience for this is somebody who has just turned the subtitles on and wants to
know whether it is them or the programme. So every number gets a sentence, and
the verdict never appears without the band around it -- the measured worst-case
error is about 2 LU, which is enough to move a borderline programme across a
line, and hiding that would make the tool exactly the kind of confident
instrument this was built to be the opposite of.
"""
from __future__ import annotations

from .measure import BIG_SPREAD_LU, BIG_SWING_LU, CLEAR_LU, OK_LU, Report

BAR = "▁▂▃▄▅▆▇█"


def strip(report: Report, width: int = 56) -> str:
    """A one-line picture of where in the runtime the dialogue is buried."""
    if not report.worst:
        return ""
    # The worst list is sparse by design; use the moments we have and leave the
    # rest blank rather than interpolating a shape that was never measured.
    cells = [" "] * width
    for m in report.worst:
        i = min(width - 1, int(m.at / max(report.duration, 1e-9) * width))
        if m.sbr_lu is None:
            cells[i] = "█"
        else:
            level = max(0, min(7, int((CLEAR_LU - m.sbr_lu) / CLEAR_LU * 7)))
            cells[i] = BAR[level]
    return "".join(cells)


def lines(report: Report) -> list[str]:
    out: list[str] = []
    mins = int(report.duration // 60)
    secs = int(report.duration % 60)
    out.append(f"{report.path}  ({mins}:{secs:02d}, {report.speech_fraction*100:.0f}% speech)")
    out.append("")

    if report.sbr_lu is None:
        out.append("VERDICT: the talking is not measurably above what is under it.")
        out.append("  Either nobody stops speaking long enough to hear the "
                   "background on its own,")
        out.append("  or the background really is as loud as the voice.")
    else:
        band = ""
        if report.sbr_range:
            lo, hi = report.sbr_range
            band = f"   (half the programme falls between {lo:+.1f} and {hi:+.1f})"
        out.append(f"VERDICT: {report.verdict()}.")
        out.append(f"  The dialogue sits {report.sbr_lu:+.1f} LU above the "
                   f"background.{band}")
        out.append(f"  Give or take about 2 LU -- see tools/recover.py for where "
                   f"that figure comes from.")
        out.append("")
        if report.sbr_lu < OK_LU:
            out.append("  Under about +4 is where people reach for the subtitles. "
                       "This is that.")
        elif report.sbr_lu < CLEAR_LU:
            out.append("  Followable in a quiet room, hard in a kitchen with the "
                       "tap running.")
        else:
            out.append("  Comfortably above the background.")

    if report.swing_lu is not None and report.swing_lu > 0:
        out.append("")
        out.append(f"  The loudest non-speech moments are {report.swing_lu:.1f} LU "
                   f"ABOVE the dialogue.")
        if report.swing_lu > BIG_SWING_LU:
            out.append("  That is the turn-it-up-for-the-talking, "
                       "down-for-the-bangs problem, and it is large here.")

    if report.spread_lu is not None and report.spread_lu > BIG_SPREAD_LU:
        out.append("")
        out.append(f"  The dialogue's own level moves by {report.spread_lu:.1f} LU "
                   f"across the programme,")
        out.append("  so quiet lines will disappear even where the average is fine.")

    if report.worst:
        out.append("")
        out.append("  Hardest moments:")
        for m in report.worst:
            if m.sbr_lu is None:
                out.append(f"    {m.clock():>8}   voice not above the background at all")
            else:
                out.append(f"    {m.clock():>8}   {m.sbr_lu:+5.1f} LU")
        s = strip(report)
        if s.strip():
            out.append("")
            out.append(f"  0:00 |{s}| {int(report.duration//60)}:{int(report.duration%60):02d}")

    for n in report.notes:
        out.append(f"\n  note: {n}")

    out.append("")
    out.append(f"  programme loudness {report.programme_lufs:.1f} LUFS"
               f"   dialogue {report.speech_lufs:.1f}"
               f"   background {report.background_lufs:.1f}")
    return out


def text(report: Report) -> str:
    return "\n".join(lines(report))
