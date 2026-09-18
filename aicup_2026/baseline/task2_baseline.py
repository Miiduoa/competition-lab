#!/usr/bin/env python3
"""Task 2 baseline: Predicting Rank (local validation only).

1) Majority-rank baseline (global mode of training labels).
2) Lightweight sklearn LogisticRegression on query-set aggregated features
   (mean/std game length, length-bin hist, opening hash buckets, color rate).

Metric: exact=1, ±1 level=1/e, else 0.

Writes:
  ../outputs/task2_baseline_metrics.json
  ../outputs/task2_baseline_preds_sample.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import (
    DEFAULT_OUTPUT_ROOT,
    RANK_ORDER,
    RANK_TO_IDX,
    data_dir,
    ensure_dir,
    length_hist,
    load_sampled_features,
    rank_distance_score,
)


def opening_bucket(s: str, n_buckets: int = 32) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_buckets


def aggregate_query_features(
    g: pd.DataFrame, n_open_buckets: int = 32
) -> np.ndarray:
    """Aggregate a set of games (one query / one player sample) → feature vector."""
    mean_moves = float(g["n_moves"].mean())
    std_moves = float(g["n_moves"].std(ddof=0))
    med_moves = float(g["n_moves"].median())
    color_b = sum(1 for c in g["color"] if c == "B") / max(len(g), 1)
    lh = length_hist(g["length_bin"].tolist())
    buckets = np.zeros(n_open_buckets, dtype=np.float64)
    for op in g["opening"]:
        buckets[opening_bucket(op, n_open_buckets)] += 1.0
    if buckets.sum() > 0:
        buckets /= buckets.sum()
    first_buckets = np.zeros(16, dtype=np.float64)
    for fc in g["first_coord"]:
        if not fc:
            continue
        first_buckets[opening_bucket(fc, 16)] += 1.0
    if first_buckets.sum() > 0:
        first_buckets /= first_buckets.sum()
    return np.concatenate(
        [
            [mean_moves, std_moves, med_moves, color_b, float(len(g))],
            lh,
            buckets,
            first_buckets,
        ]
    )


def build_player_examples(
    df: pd.DataFrame,
    games_per_example: int = 5,
    max_examples_per_player: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Each example = a small set of games from one player (simulates query set)."""
    rng = np.random.default_rng(seed)
    Xs, ys, pids, ranks = [], [], [], []
    for pid, g in df.groupby("player_id", sort=False):
        if len(g) < games_per_example:
            continue
        rank = str(g["rank"].iloc[0])
        if rank not in RANK_TO_IDX:
            continue
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_ex = min(max_examples_per_player, len(idx) // games_per_example)
        for i in range(n_ex):
            sl = idx[i * games_per_example : (i + 1) * games_per_example]
            feat = aggregate_query_features(df.loc[sl])
            Xs.append(feat)
            ys.append(RANK_TO_IDX[rank])
            pids.append(pid)
            ranks.append(rank)
    return np.vstack(Xs), np.array(ys), pids, ranks


def evaluate_preds(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> dict:
    scores = []
    exact = 0
    within1 = 0
    for t, p in zip(y_true_idx, y_pred_idx):
        tr = RANK_ORDER[int(t)]
        pr = RANK_ORDER[int(p)]
        sc = rank_distance_score(tr, pr)
        scores.append(sc)
        d = abs(int(t) - int(p))
        if d == 0:
            exact += 1
        if d <= 1:
            within1 += 1
    n = len(scores) or 1
    return {
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "exact_acc": exact / n,
        "within1_acc": within1 / n,
        "n": len(scores),
    }


def run(
    task_dir: Path,
    out_dir: Path,
    max_games_per_rank: int = 4000,
    games_per_example: int = 5,
    max_examples_per_player: int = 2,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"[task2] loading sampled features from {task_dir} …")
    # Prefer task2 copy; fall back to task1 if identical
    if not any(task_dir.glob("train_*.csv")):
        alt = data_dir("task1")
        print(f"[task2] {task_dir} empty; using {alt}")
        task_dir = alt

    df = load_sampled_features(
        task_dir, max_games_per_rank=max_games_per_rank, seed=seed
    )
    print(
        f"[task2] loaded {len(df)} games, "
        f"{df['player_id'].nunique()} players"
    )

    X, y, pids, ranks = build_player_examples(
        df,
        games_per_example=games_per_example,
        max_examples_per_player=max_examples_per_player,
        seed=seed,
    )
    print(f"[task2] built {len(y)} query-set examples")

    # Group-aware split by player to reduce leakage
    unique_pids = np.array(sorted(set(pids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n_train = int(0.8 * len(unique_pids))
    train_pids = set(unique_pids[:n_train])
    train_mask = np.array([p in train_pids for p in pids])
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # --- Majority baseline (global mode on train labels) ---
    mode_idx = int(Counter(y_train.tolist()).most_common(1)[0][0])
    maj_pred = np.full_like(y_test, mode_idx)
    maj_metrics = evaluate_preds(y_test, maj_pred)

    # --- Logistic regression ---
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        C=1.0,
        random_state=seed,
    )
    clf.fit(Xtr, y_train)
    lr_pred = clf.predict(Xte)
    lr_metrics = evaluate_preds(y_test, lr_pred)

    # Sample preds CSV
    pred_rows = []
    test_pids = [p for p, m in zip(pids, test_mask) if m]
    test_ranks = [r for r, m in zip(ranks, test_mask) if m]
    for i in range(min(200, len(y_test))):
        pred_rows.append(
            {
                "player_id": test_pids[i],
                "true_rank": test_ranks[i],
                "pred_majority": RANK_ORDER[mode_idx],
                "pred_logistic": RANK_ORDER[int(lr_pred[i])],
                "score_majority": rank_distance_score(test_ranks[i], RANK_ORDER[mode_idx]),
                "score_logistic": rank_distance_score(
                    test_ranks[i], RANK_ORDER[int(lr_pred[i])]
                ),
            }
        )

    metrics = {
        "task": "task2_predicting_rank",
        "metric": "exact=1_pm1=1/e_else=0",
        "majority_baseline": maj_metrics,
        "logistic_baseline": lr_metrics,
        "majority_rank": RANK_ORDER[mode_idx],
        "sample": {
            "max_games_per_rank": max_games_per_rank,
            "games_per_example": games_per_example,
            "max_examples_per_player": max_examples_per_player,
            "seed": seed,
            "n_games_loaded": int(len(df)),
            "n_examples": int(len(y)),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_train_players": len(train_pids),
            "n_test_players": int(len(unique_pids) - len(train_pids)),
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local holdout validation only — not an official leaderboard score.",
    }

    ensure_dir(out_dir)
    metrics_path = out_dir / "task2_baseline_metrics.json"
    preds_path = out_dir / "task2_baseline_preds_sample.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).to_csv(preds_path, index=False)

    print(
        f"[task2] majority mean={maj_metrics['mean_score']:.4f} "
        f"(exact={maj_metrics['exact_acc']:.3f})"
    )
    print(
        f"[task2] logistic mean={lr_metrics['mean_score']:.4f} "
        f"(exact={lr_metrics['exact_acc']:.3f}, "
        f"±1={lr_metrics['within1_acc']:.3f})"
    )
    print(f"[task2] wrote {metrics_path}")
    print(f"[task2] wrote {preds_path}")
    print(f"[task2] elapsed {metrics['elapsed_sec']}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AI CUP 2026 Task2 baseline")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=4000)
    ap.add_argument("--games-per-example", type=int, default=5)
    ap.add_argument("--max-examples-per-player", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    task_dir = data_dir("task2", args.data_root)
    run(
        task_dir=task_dir,
        out_dir=args.out_dir,
        max_games_per_rank=args.max_games_per_rank,
        games_per_example=args.games_per_example,
        max_examples_per_player=args.max_examples_per_player,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
