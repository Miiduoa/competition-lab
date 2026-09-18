#!/usr/bin/env python3
"""Task2 v2: rich set aggregation + LightGBM / HistGB (local holdout).

Metric: exact=1, ±1=1/e, else 0.
Player-grouped split. Softmax probs → pick class maximizing expected score.

Writes:
  ../outputs/task2_v2_metrics.json
  ../outputs/task2_v2_preds_sample.csv
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from features_v2 import (
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    RANK_ORDER,
    RANK_TO_IDX,
    data_dir,
    dense_matrix,
    ensure_dir,
    load_sampled_features_v2,
    rank_distance_score,
)

INV_E = 1.0 / np.e
N_RANKS = len(RANK_ORDER)


def aggregate_set(g: pd.DataFrame) -> np.ndarray:
    """Mean / std / q25 / q75 of dense cols + a few extras."""
    dens = dense_matrix(g)
    mean = dens.mean(axis=0)
    std = dens.std(axis=0)
    q25 = np.percentile(dens, 25, axis=0)
    q75 = np.percentile(dens, 75, axis=0)
    # Extra: set size, color rate, move quantiles already in mean of n_moves
    extra = np.array(
        [
            float(len(g)),
            float(g["color_b"].mean()),
            float(g["n_moves"].median()),
            float(g["pass_rate"].mean()) if "pass_rate" in g.columns else float(mean[1]),
            float(g["corner_rate"].mean()) if "corner_rate" in g.columns else float(mean[5]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([mean, std, q25, q75, extra])


def build_examples(
    df: pd.DataFrame,
    games_per_example: int = 5,
    max_examples_per_player: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
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
            Xs.append(aggregate_set(df.loc[sl]))
            ys.append(RANK_TO_IDX[rank])
            pids.append(pid)
            ranks.append(rank)
    return np.vstack(Xs), np.array(ys), pids, ranks


def expected_score_predict(proba: np.ndarray) -> np.ndarray:
    """For each row, pick class maximizing E[score] under ±1 metric."""
    # E_k = P(k)*1 + P(k-1)*1/e + P(k+1)*1/e
    n, c = proba.shape
    exp = np.zeros_like(proba)
    for k in range(c):
        exp[:, k] = proba[:, k]
        if k - 1 >= 0:
            exp[:, k] += INV_E * proba[:, k - 1]
        if k + 1 < c:
            exp[:, k] += INV_E * proba[:, k + 1]
    return exp.argmax(axis=1)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    scores, exact, within1 = [], 0, 0
    for t, p in zip(y_true, y_pred):
        tr, pr = RANK_ORDER[int(t)], RANK_ORDER[int(p)]
        scores.append(rank_distance_score(tr, pr))
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


def player_split(pids: List[str], seed: int = 42, train_frac: float = 0.8):
    unique = np.array(sorted(set(pids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(train_frac * len(unique))
    train_pids = set(unique[:n_train])
    train_mask = np.array([p in train_pids for p in pids])
    return train_mask, train_pids, unique


def run(
    task_dir: Path,
    out_dir: Path,
    max_games_per_rank: int = 7000,
    games_per_example: int = 5,
    max_examples_per_player: int = 3,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"[task2_v2] loading richer features from {task_dir} …")
    if not any(task_dir.glob("train_*.csv")):
        task_dir = data_dir("task1")
        print(f"[task2_v2] fallback to {task_dir}")

    df = load_sampled_features_v2(
        task_dir, max_games_per_rank=max_games_per_rank, seed=seed
    )
    print(f"[task2_v2] loaded {len(df)} games, {df['player_id'].nunique()} players")

    X, y, pids, ranks = build_examples(
        df,
        games_per_example=games_per_example,
        max_examples_per_player=max_examples_per_player,
        seed=seed,
    )
    print(f"[task2_v2] built {len(y)} examples, dim={X.shape[1]}")

    train_mask, train_pids, unique_pids = player_split(pids, seed=seed)
    test_mask = ~train_mask
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Majority
    mode_idx = int(Counter(y_train.tolist()).most_common(1)[0][0])
    maj_metrics = evaluate(y_test, np.full_like(y_test, mode_idx))

    # LightGBM multiclass
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "multiclass",
        "num_class": N_RANKS,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
        "n_jobs": -1,
    }
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=400,
        valid_sets=[lgb_train],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    lgb_proba = booster.predict(X_test)
    lgb_pred_raw = lgb_proba.argmax(axis=1)
    lgb_pred_cal = expected_score_predict(lgb_proba)
    lgb_raw_m = evaluate(y_test, lgb_pred_raw)
    lgb_cal_m = evaluate(y_test, lgb_pred_cal)

    # HistGradientBoosting fallback / comparison
    hgb = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.08,
        max_iter=250,
        random_state=seed,
    )
    hgb.fit(X_train, y_train)
    hgb_proba = hgb.predict_proba(X_test)
    # Align columns if some classes missing
    full_proba = np.zeros((len(y_test), N_RANKS), dtype=np.float64)
    for i, cls in enumerate(hgb.classes_):
        full_proba[:, int(cls)] = hgb_proba[:, i]
    hgb_pred = expected_score_predict(full_proba)
    hgb_m = evaluate(y_test, hgb_pred)

    # Pick best of LGB cal / HGB for "primary"
    primary_name = "lightgbm_expected_score"
    primary = lgb_cal_m
    primary_pred = lgb_pred_cal
    if hgb_m["mean_score"] > primary["mean_score"]:
        primary_name = "histgb_expected_score"
        primary = hgb_m
        primary_pred = hgb_pred

    test_pids = [p for p, m in zip(pids, test_mask) if m]
    test_ranks = [r for r, m in zip(ranks, test_mask) if m]
    pred_rows = []
    for i in range(min(200, len(y_test))):
        pred_rows.append(
            {
                "player_id": test_pids[i],
                "true_rank": test_ranks[i],
                "pred_majority": RANK_ORDER[mode_idx],
                "pred_lgb_raw": RANK_ORDER[int(lgb_pred_raw[i])],
                "pred_lgb_cal": RANK_ORDER[int(lgb_pred_cal[i])],
                "pred_hgb_cal": RANK_ORDER[int(hgb_pred[i])],
                "pred_primary": RANK_ORDER[int(primary_pred[i])],
                "score_primary": rank_distance_score(
                    test_ranks[i], RANK_ORDER[int(primary_pred[i])]
                ),
            }
        )

    metrics = {
        "task": "task2_predicting_rank_v2",
        "metric": "exact=1_pm1=1/e_else=0",
        "majority": maj_metrics,
        "lightgbm_argmax": lgb_raw_m,
        "lightgbm_expected_score": lgb_cal_m,
        "histgb_expected_score": hgb_m,
        "primary_model": primary_name,
        "primary": primary,
        "majority_rank": RANK_ORDER[mode_idx],
        "feature_dim": int(X.shape[1]),
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
        "baseline_ref_local": {
            "logistic_mean": 0.2213,
            "note": "baseline task2 logistic local holdout; not leaderboard",
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local holdout validation only — NOT an official AIdea leaderboard score.",
    }

    ensure_dir(out_dir)
    metrics_path = out_dir / "task2_v2_metrics.json"
    preds_path = out_dir / "task2_v2_preds_sample.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).to_csv(preds_path, index=False)

    # Optionally persist LightGBM model for later
    model_dir = Path(__file__).resolve().parent / "artifacts"
    ensure_dir(model_dir)
    booster.save_model(str(model_dir / "task2_lgbm.txt"))

    print(
        f"[task2_v2] majority mean={maj_metrics['mean_score']:.4f} "
        f"(exact={maj_metrics['exact_acc']:.3f})"
    )
    print(
        f"[task2_v2] LGB argmax mean={lgb_raw_m['mean_score']:.4f} "
        f"cal={lgb_cal_m['mean_score']:.4f} "
        f"(exact={lgb_cal_m['exact_acc']:.3f}, ±1={lgb_cal_m['within1_acc']:.3f})"
    )
    print(
        f"[task2_v2] HistGB cal mean={hgb_m['mean_score']:.4f} "
        f"(exact={hgb_m['exact_acc']:.3f}, ±1={hgb_m['within1_acc']:.3f})"
    )
    print(f"[task2_v2] primary={primary_name} mean={primary['mean_score']:.4f}")
    print(f"[task2_v2] wrote {metrics_path}")
    print(f"[task2_v2] wrote {preds_path}")
    print(f"[task2_v2] elapsed {metrics['elapsed_sec']}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AI CUP 2026 Task2 v2")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=7000)
    ap.add_argument("--games-per-example", type=int, default=5)
    ap.add_argument("--max-examples-per-player", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(
        task_dir=data_dir("task2", args.data_root),
        out_dir=args.out_dir,
        max_games_per_rank=args.max_games_per_rank,
        games_per_example=args.games_per_example,
        max_examples_per_player=args.max_examples_per_player,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
