"""Tests for the streaming multipart reader.

Byte-level parsing is the kind of code that is correct for every fixture you
think to write and wrong on the first real file, so the cases here are chosen
to be the ones that would NOT be caught by a happy path:

  - a boundary landing exactly across a read-chunk edge, at every offset
  - binary payloads containing bytes that look like the start of a boundary
  - the size cap firing DURING the read rather than after
  - a body that ends early

Run: python3.12 -m pytest tests/ -q   (or python3.12 tests/test_upload.py)
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earshot import serve, studio                                  # noqa: E402

BOUND = "----earshotTESTboundary9f2c"
CT = f"multipart/form-data; boundary={BOUND}"


def body(payload: bytes, fields=(), filename="song.mp3") -> bytes:
    sep = f"--{BOUND}\r\n".encode()
    out = b""
    for name, value in fields:
        out += sep
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += value.encode() + b"\r\n"
    if filename is not None:
        out += sep
        out += (f'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
                f'Content-Type: audio/mpeg\r\n\r\n').encode()
        out += payload + b"\r\n"
    out += f"--{BOUND}--\r\n".encode()
    return out


def run(payload: bytes, tmp: Path, fields=(), filename="song.mp3",
        max_bytes=1 << 30, chunk=None):
    raw = body(payload, fields, filename)
    if chunk is not None:
        old, serve.UPLOAD_CHUNK = serve.UPLOAD_CHUNK, chunk
    try:
        dest = tmp / "staged.mp3"
        f, written, name = serve.read_multipart_to_disk(
            io.BytesIO(raw), CT, len(raw), dest, max_bytes)
        got = dest.read_bytes() if dest.exists() else b""
        return f, written, name, got
    finally:
        if chunk is not None:
            serve.UPLOAD_CHUNK = old


def test_roundtrip_small(tmp):
    payload = b"\x00\x01\x02hello there\xff\xfe"
    f, written, name, got = run(payload, tmp, fields=[("part", "vocals")])
    assert got == payload, "payload came back changed"
    assert written == len(payload)
    assert name == "song.mp3"
    assert f["part"] == "vocals"


def test_binary_payload_survives(tmp):
    payload = bytes(range(256)) * 40
    _, written, _, got = run(payload, tmp)
    assert got == payload
    assert written == len(payload)


def test_boundary_across_every_chunk_edge(tmp):
    """The case that a happy path cannot see.

    Hold back too few bytes and the reader writes half a boundary into the file
    and then fails to find the other half. It only misbehaves when the boundary
    straddles a chunk edge, so a single fixture has roughly a 1-in-chunk-size
    chance of noticing. Walk the payload length across a whole chunk instead.
    """
    chunk = 64
    for n in range(0, 3 * chunk):
        payload = bytes([(i * 7) % 251 for i in range(n)])
        _, written, _, got = run(payload, tmp, chunk=chunk)
        assert got == payload, f"corrupted at length {n}"
        assert written == n, f"wrong count at length {n}"


def test_payload_containing_boundary_prefix(tmp):
    """Bytes that look like the start of a boundary but are not one."""
    fake = f"--{BOUND[:-3]}".encode()
    payload = b"before" + fake + b"after" + fake + b"end"
    _, _, _, got = run(payload, tmp, chunk=17)
    assert got == payload


def test_cap_refuses_during_the_read(tmp):
    """Oversize must be refused as it arrives, not after it is all accepted."""
    payload = b"x" * 5000
    try:
        run(payload, tmp, max_bytes=1000, chunk=128)
    except studio.Refused as e:
        assert "too big" in str(e)
        staged = tmp / "staged.mp3"
        # The point of streaming: the file on disk never grew far past the cap.
        if staged.exists():
            assert staged.stat().st_size <= 1000 + 128 + len(BOUND) + 8, \
                "the cap did not stop the write, it only reported afterwards"
        return
    raise AssertionError("an oversize upload was accepted")


def test_no_file_part(tmp):
    f, written, name, _ = run(b"", tmp, fields=[("part", "drums")], filename=None)
    assert written == 0 and name == ""
    assert f["part"] == "drums"


def test_fields_before_and_after_the_file(tmp):
    raw = body(b"AUDIO", fields=[("part", "vocals")])
    extra = (f"--{BOUND}\r\n"
             f'Content-Disposition: form-data; name="model"\r\n\r\nsix\r\n').encode()
    raw = raw.replace(f"--{BOUND}--\r\n".encode(), extra + f"--{BOUND}--\r\n".encode())
    dest = tmp / "staged.mp3"
    f, written, _ = serve.read_multipart_to_disk(
        io.BytesIO(raw), CT, len(raw), dest, 1 << 30)
    assert dest.read_bytes() == b"AUDIO"
    assert f["part"] == "vocals" and f["model"] == "six"


def test_truncated_body_does_not_hang_or_raise(tmp):
    raw = body(b"z" * 500)[:-40]
    dest = tmp / "staged.mp3"
    _, written, _ = serve.read_multipart_to_disk(
        io.BytesIO(raw), CT, len(raw), dest, 1 << 30)
    assert written > 0


def test_socket_is_drained_on_refusal(tmp):
    """Leaving bytes in the socket breaks the NEXT request under keep-alive.

    The chunk size matters and the first version of this test got it wrong: at
    the default 1 MB the whole 5 KB body is swallowed by the first read, so
    `remaining` is 0 by the time the cap fires and the drain has nothing to do.
    The test passed with the drain deleted. Mutation testing is the only reason
    I know that.
    """
    raw = body(b"x" * 5000)
    stream = io.BytesIO(raw + b"GET /health HTTP/1.1\r\n\r\n")
    old, serve.UPLOAD_CHUNK = serve.UPLOAD_CHUNK, 256
    try:
        serve.read_multipart_to_disk(stream, CT, len(raw), tmp / "s.mp3", 1000)
    except studio.Refused:
        pass
    finally:
        serve.UPLOAD_CHUNK = old
    assert stream.read() == b"GET /health HTTP/1.1\r\n\r\n", \
        "the refused body was left in the socket"


def test_memory_is_bounded_not_proportional(tmp):
    """The whole reason this exists. 8 MB in must not cost 8 MB of buffer."""
    import tracemalloc
    payload = os.urandom(8 << 20)
    raw = body(payload)
    dest = tmp / "big.mp3"
    tracemalloc.start()
    serve.read_multipart_to_disk(io.BytesIO(raw), CT, len(raw), dest, 1 << 30)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert dest.stat().st_size == len(payload)
    # Generous: the point is that it is a small multiple of the CHUNK, not of
    # the FILE. The old path would sit around 3x the file here.
    assert peak < 8 * serve.UPLOAD_CHUNK, \
        f"peak {peak/1e6:.1f} MB for an 8 MB upload — still proportional to the file"


if __name__ == "__main__":
    import tempfile, traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"  ok   {name}")
            except Exception:
                fails += 1
                print(f"  FAIL {name}")
                traceback.print_exc()
    print("all green" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
