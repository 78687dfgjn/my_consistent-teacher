"""Resume a COCO archive with concurrent HTTP range requests.

Usage: python tools/dataset/download_coco_parallel.py train2017.zip
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from urllib.request import Request, urlopen


def remote_size(url):
    with urlopen(Request(url, method="HEAD"), timeout=30) as response:
        return int(response.headers["Content-Length"])


def fetch_part(url, path, start, end):
    expected = end - start + 1
    while True:
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            break
        if have > expected:
            raise ValueError(f"Part exceeds its assigned range: {path}")
        first = start + have
        request = Request(url, headers={"Range": f"bytes={first}-{end}"})
        try:
            with urlopen(request, timeout=60) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {first}-"):
                    raise ValueError(f"Unexpected range response: {content_range}")
                with path.open("ab") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
        except OSError as exc:
            print(f"{path.name}: retrying after {exc}", flush=True)
            time.sleep(5)
    print(f"{path.name}: complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--connections", type=int, default=12)
    args = parser.parse_args()
    archive = args.archive
    url = f"http://images.cocodataset.org/zips/{archive.name}"
    size = remote_size(url)
    prefix = archive.stat().st_size if archive.exists() else 0
    if prefix >= size:
        print(f"Already downloaded: {archive}")
        return
    step = (size - prefix + args.connections - 1) // args.connections
    ranges = [(prefix + i * step, min(size - 1, prefix + (i + 1) * step - 1))
              for i in range(args.connections)]
    parts = [archive.with_name(f"{archive.name}.part{i:02d}")
             for i in range(args.connections)]
    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        futures = [pool.submit(fetch_part, url, part, start, end)
                   for part, (start, end) in zip(parts, ranges) if start <= end]
        for future in futures:
            future.result()
    with archive.open("ab") as output:
        for part in parts:
            if not part.exists():
                continue
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
            part.unlink()
    if archive.stat().st_size != size:
        raise ValueError(f"Unexpected archive size: {archive.stat().st_size} != {size}")
    print(f"Download complete: {archive}", flush=True)


if __name__ == "__main__":
    main()
