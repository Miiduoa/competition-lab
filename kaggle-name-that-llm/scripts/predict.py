#!/usr/bin/env python3
"""Fit on all local training rows and write submission.parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ntl.data_io import load_competition
from ntl.pipeline import NTLClassifier, make_synthetic, write_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "submission.parquet")
    parser.add_argument("--extra-weight", type=float, default=0.55)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    extra = None
    if args.smoke:
        train, test = make_synthetic()
    else:
        train, test, extra = load_competition(args.data_dir)

    texts = train["text"].astype(str).tolist()
    labels = train["model"].astype(str).tolist()
    weights = [1.0] * len(texts)
    if extra is not None and len(extra):
        texts.extend(extra["text"].astype(str).tolist())
        labels.extend(extra["model"].astype(str).tolist())
        weights.extend([args.extra_weight] * len(extra))

    clf = NTLClassifier(extra_weight=args.extra_weight)
    clf.fit(texts, labels, sample_weight=np.asarray(weights, dtype=float))
    pred = clf.predict(test["text"].astype(str).tolist())
    path = write_submission(test["id"].astype(str), pred, args.out)
    print(f"wrote {path} rows={len(pred)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
