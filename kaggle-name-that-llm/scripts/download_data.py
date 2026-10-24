#!/usr/bin/env python3
"""Download Name That LLM tables. Prefers the public Hugging Face mirror."""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HF_TRAIN = "https://huggingface.co/datasets/loeclos/name-that-llm/resolve/main/train.csv"
HF_TEST = "https://huggingface.co/datasets/loeclos/name-that-llm/resolve/main/test.csv"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)", flush=True)


def try_kaggle(dest_dir: Path) -> bool:
    try:
        subprocess.check_call(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "name-that-llm",
                "-p",
                str(dest_dir),
            ]
        )
        return True
    except Exception as exc:
        print(f"kaggle download skipped: {exc}", flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        download(HF_TRAIN, args.out / "train.csv")
        download(HF_TEST, args.out / "test.csv")
        return 0
    except Exception as exc:
        print(f"Hugging Face download failed: {exc}", flush=True)
    if try_kaggle(args.out):
        return 0
    print("No data downloaded. Place train.csv and test.csv under data/.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
