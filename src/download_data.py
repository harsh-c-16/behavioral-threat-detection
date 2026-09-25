"""Fetch the LANL 2015 auth + red-team files into data/raw/.

LANL hands out the files only after a short web form at
https://csr.lanl.gov/data/cyber1/ (email + intended use). Pass the links you get
there, or paths to files you already downloaded.

auth.txt.gz is 7.2 GB for 58 days. The file is time-sorted, so by default only
the first --days days are streamed and the download stops there (~125 MB of
compressed data per day). prepare_data.py reads the truncated gzip.
"""
import argparse
import time
import urllib.request
import zlib
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DAY = 86400


class LogClock:
    """Feed gzip bytes in order; tracks the time of the last complete decompressed line."""

    def __init__(self) -> None:
        self.inflate, self.tail, self.last_time = zlib.decompressobj(zlib.MAX_WBITS | 16), b"", -1

    def feed(self, chunk: bytes) -> int:
        buf = self.tail + self.inflate.decompress(chunk)
        end = buf.rfind(b"\n")
        if end > 0:
            self.last_time = int(buf[buf.rfind(b"\n", 0, end) + 1:end].split(b",", 1)[0])
            buf = buf[end + 1:]
        self.tail = buf
        return self.last_time


def fetch_prefix(url: str, out: Path, max_time: int) -> None:
    """Copy gzip bytes from url until the decompressed log passes max_time seconds.

    If `out` already holds the start of the file (an interrupted run), the existing
    bytes are replayed through the decompressor and the download resumes from there.
    """
    clock, size = LogClock(), 0
    if out.exists():
        with open(out, "rb") as f:
            while chunk := f.read(1 << 20):
                clock.feed(chunk)
                size += len(chunk)
        print(f"  resuming after {size >> 20} MB already on disk (log time {clock.last_time / DAY:.2f} days)")
        if clock.last_time >= max_time:
            return
    failures = 0
    while clock.last_time < max_time:  # the LANL server drops long connections
        before = size
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={size}-"} if size else {})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if size and resp.status != 206:
                    raise SystemExit("server ignored the resume request; delete data/raw/auth.txt.gz and rerun")
                with open(out, "ab" if size else "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        size += len(chunk)
                        if clock.feed(chunk) >= max_time:
                            break
                        if size % (50 << 20) < len(chunk):
                            print(f"  {size >> 20} MB, log time {clock.last_time / DAY:.2f} days")
        except OSError as e:
            print(f"  connection error: {e}")
        failures = 0 if size > before else failures + 1
        if failures >= 5:
            raise SystemExit(f"no progress after 5 attempts; rerun later to resume from {size >> 20} MB")
        if failures:
            time.sleep(10)
    print(f"  saved {size >> 20} MB to {out}")


def get(src: str, out: Path, max_time: int | None = None) -> None:
    """Link a local file, download a small file whole, or stream a prefix of a large one."""
    if Path(src).exists():
        if Path(src).resolve() == out.resolve():
            return
        out.unlink(missing_ok=True)
        out.symlink_to(Path(src).resolve())
        print(f"  linked {out} -> {src}")
    elif max_time is None:
        urllib.request.urlretrieve(src, out)
    else:
        fetch_prefix(src, out, max_time)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", required=True, help="URL or local path of auth.txt.gz")
    ap.add_argument("--redteam", required=True, help="URL or local path of redteam.txt.gz")
    ap.add_argument("--days", type=float, default=10.0, help="days of auth data to stream")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    print("red-team labels")
    get(args.redteam, RAW / "redteam.txt.gz")
    rt = pd.read_csv(RAW / "redteam.txt.gz", names=["time", "user", "src", "dst"])
    print(f"  {len(rt)} red-team events; per day (pick --calib-end / --test-end in prepare_data.py from this):")
    print((rt.time // DAY).value_counts().sort_index().rename("events").to_string())

    print(f"auth events, first {args.days:g} days")
    get(args.auth, RAW / "auth.txt.gz", int(args.days * DAY))


if __name__ == "__main__":
    main()
