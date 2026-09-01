#!/usr/bin/env python3.12
"""Does the memory cap on a separation job actually fire?

`studio._run` claims to run the worker under `systemd-run --user --scope` with a
MemoryMax, so that a job which goes wrong dies alone instead of taking the talk
server and the doorbell with it. That claim was a comment in a file before it
was a fact, which is the shape this whole project keeps catching itself in, so:

  capped     allocate well past the cap under the scope   -> must be KILLED
  uncapped   allocate exactly the same amount, no scope   -> must SUCCEED

The second one is the control and it is the point. A test where the capped run
dies proves nothing on its own -- the allocation might fail for its own reasons,
or systemd-run might not exist, or the command might be wrong. Only the pair
says the cap is what killed it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CAP_MB = 200
GRAB_MB = 500          # comfortably past the cap, comfortably inside this box

EATER = (
    "import sys;"
    f"b = bytearray({GRAB_MB} * 1024 * 1024);"
    # touch every page, or the kernel never actually hands the memory over and
    # the cgroup has nothing to object to
    "  \n"
    "for i in range(0, len(b), 4096): b[i] = 1\n"
    "print('allocated fine')"
)


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()[-200:]


def main() -> int:
    py = sys.executable
    eater = [py, "-c", EATER]

    if run(["systemd-run", "--user", "--scope", "--quiet",
            "-p", "MemoryMax=100M", "/bin/true"])[0] != 0:
        print("systemd-run --user --scope does not work here.")
        print("The cap CANNOT be relied on; studio._run falls back to an "
              "uncapped subprocess and the queue-of-one is the only protection.")
        return 1

    capped = ["systemd-run", "--user", "--scope", "--quiet",
              "-p", f"MemoryMax={CAP_MB}M", "-p", "MemorySwapMax=0"] + eater
    code_capped, out_capped = run(capped)
    code_plain, out_plain = run(eater)

    print(f"grabbing {GRAB_MB} MB")
    print(f"  under a {CAP_MB} MB cap : exit {code_capped}   {out_capped[:90]}")
    print(f"  with no cap at all     : exit {code_plain}   {out_plain[:90]}")

    killed = code_capped != 0
    survived = code_plain == 0
    print()
    if killed and survived:
        print("PASS: the cap killed it, and the same allocation is fine without "
              "the cap.\n      So it is the cap doing the killing, not the "
              "allocation failing on its own.")
        return 0
    if not survived:
        print("INCONCLUSIVE: the uncapped control also failed, so this machine "
              "could not\n              spare the memory either way and the "
              "capped result proves nothing.")
        return 1
    print("FAIL: the cap did not stop a job from taking more than it was "
          "allowed.\n      studio.WORKER_MEMORY_MAX is decoration.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
