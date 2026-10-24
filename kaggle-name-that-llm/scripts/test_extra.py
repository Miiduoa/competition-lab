#!/usr/bin/env python3
"""Smoke the extra-passage cleaner without GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ntl.extra import clean_extra, subsample_extra
from ntl.pipeline import make_synthetic


def main() -> int:
    train, _ = make_synthetic(n_prompts=4, seed=1)
    extra = pd.DataFrame(
        [
            {
                "id": "e1",
                "prompt_id": "x",
                "text": "A " + ("word " * 50) + "with an em dash — here.",
                "model": "gemini-3.6-flash",
            },
            {
                "id": "e2",
                "prompt_id": "x",
                "text": "A " + ("word " * 50) + "clean gemini passage without that mark.",
                "model": "gemini-3.6-flash",
            },
            {
                "id": "e3",
                "prompt_id": "x",
                "text": "# heading\n" + ("word " * 50),
                "model": "gpt-5.5",
            },
            {
                "id": "e4",
                "prompt_id": "x",
                "text": "short",
                "model": "gpt-5.5",
            },
        ]
    )
    cleaned = clean_extra(extra, train)
    assert "e1" not in set(cleaned["id"]), cleaned
    assert "e2" in set(cleaned["id"]), cleaned
    assert "e3" not in set(cleaned["id"]), cleaned
    assert "e4" not in set(cleaned["id"]), cleaned
    sub = subsample_extra(pd.concat([cleaned, cleaned], ignore_index=True), max_rows=1)
    assert len(sub) == 1
    print("extra cleaner ok", len(cleaned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
