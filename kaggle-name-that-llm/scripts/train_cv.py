#!/usr/bin/env python3
"""Prompt-grouped cross-validation and local submission file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ntl.data_io import load_competition
from ntl.pipeline import (
    NTLClassifier,
    grouped_cv,
    make_synthetic,
    write_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--extra-weight", type=float, default=0.55)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    used_real = False
    extra = None
    if args.smoke:
        train, test = make_synthetic(n_prompts=8, seed=0)
        print("[smoke] synthetic passages only; scores are not a leaderboard estimate.", flush=True)
    else:
        try:
            train, test, extra = load_competition(args.data_dir)
            used_real = True
            print(f"loaded train={len(train)} test={len(test)} extra={len(extra)}", flush=True)
        except FileNotFoundError:
            train, test = make_synthetic(n_prompts=8, seed=0)
            print("[warn] competition files missing; falling back to synthetic smoke.", flush=True)

    result = {
        "used_real_data": used_real,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_extra": int(len(extra) if extra is not None else 0),
        "metric": "balanced_accuracy",
        "cv": "StratifiedGroupKFold by prompt_id",
        "smoke": bool(args.smoke),
    }
    if not args.skip_cv:
        cv = grouped_cv(
            train,
            extra=extra,
            n_splits=min(args.folds, max(2, train["prompt_id"].nunique())),
            extra_weight=args.extra_weight,
        )
        result["overall"] = cv["overall"]
        result["fold_scores"] = cv["fold_scores"]
        result["per_class"] = cv["per_class"]
        print(f"grouped CV balanced_accuracy={cv['overall']:.4f}", flush=True)

    texts = train["text"].astype(str).tolist()
    labels = train["model"].astype(str).tolist()
    weights = [1.0] * len(texts)
    if extra is not None and len(extra):
        texts.extend(extra["text"].astype(str).tolist())
        labels.extend(extra["model"].astype(str).tolist())
        weights.extend([args.extra_weight] * len(extra))
    clf = NTLClassifier(extra_weight=args.extra_weight)
    clf.fit(texts, labels, sample_weight=__import__("numpy").asarray(weights, dtype=float))
    pred = clf.predict(test["text"].astype(str).tolist())
    sub_path = write_submission(test["id"].astype(str), pred, args.out_dir / "submission.parquet")
    result["submission"] = str(sub_path)
    result_name = "cv_results_smoke.json" if (args.smoke or not used_real) else "cv_results.json"
    result_path = args.out_dir / result_name
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {sub_path}", flush=True)
    print(f"wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
