#!/usr/bin/env python3
"""Task 1 baseline: Identifying Players (local validation only).

Samples games, builds per-player profiles (move-length hist + opening
fingerprint frequency), holds out K games per player as query sets,
and scores Top-5 retrieval with exponential decay.

Writes:
  ../outputs/task1_baseline_metrics.json
  ../outputs/task1_baseline_preds_sample.csv
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from features import (
    DEFAULT_OUTPUT_ROOT,
    cosine_sim,
    data_dir,
    ensure_dir,
    jaccard_from_counters,
    length_hist,
    load_sampled_features,
    top5_exp_decay_score,
)


def build_profiles(
    df: pd.DataFrame,
) -> Dict[str, dict]:
    """Aggregate gallery games into per-player profiles."""
    profiles: Dict[str, dict] = {}
    grouped = df.groupby("player_id", sort=False)
    for pid, g in grouped:
        bins = g["length_bin"].tolist()
        openings = g["opening"].tolist()
        colors = g["color"].tolist()
        firsts = g["first_coord"].tolist()
        profiles[pid] = {
            "n_games": len(g),
            "length_hist": length_hist(bins),
            "opening_counter": Counter(openings),
            "color_b_rate": sum(1 for c in colors if c == "B") / max(len(colors), 1),
            "first_counter": Counter(firsts),
            "mean_moves": float(g["n_moves"].mean()),
            "std_moves": float(g["n_moves"].std(ddof=0)),
            "rank": str(g["rank"].iloc[0]),
        }
    return profiles


def query_vector(g: pd.DataFrame) -> dict:
    return {
        "length_hist": length_hist(g["length_bin"].tolist()),
        "opening_counter": Counter(g["opening"].tolist()),
        "color_b_rate": sum(1 for c in g["color"] if c == "B") / max(len(g), 1),
        "first_counter": Counter(g["first_coord"].tolist()),
        "mean_moves": float(g["n_moves"].mean()),
        "std_moves": float(g["n_moves"].std(ddof=0)),
    }


def score_pair(q: dict, p: dict) -> float:
    """Weighted similarity between query set and player profile."""
    s_len = cosine_sim(q["length_hist"], p["length_hist"])
    s_open = jaccard_from_counters(q["opening_counter"], p["opening_counter"])
    s_first = jaccard_from_counters(q["first_counter"], p["first_counter"])
    s_color = 1.0 - abs(q["color_b_rate"] - p["color_b_rate"])
    # mean moves closeness (normalized by ~400 stone game)
    dm = abs(q["mean_moves"] - p["mean_moves"]) / 400.0
    s_moves = float(np.exp(-dm))
    return 0.30 * s_len + 0.35 * s_open + 0.15 * s_first + 0.10 * s_color + 0.10 * s_moves


def holdout_split(
    df: pd.DataFrame,
    min_games: int = 5,
    query_k: int = 3,
    max_players: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Select players with enough games; hold out query_k games each."""
    rng = np.random.default_rng(seed)
    counts = df.groupby("player_id").size()
    eligible = counts[counts >= min_games].index.tolist()
    if len(eligible) > max_players:
        eligible = list(rng.choice(eligible, size=max_players, replace=False))
    eligible_set = set(eligible)
    gallery_rows = []
    query_rows = []
    kept = []
    for pid, g in df.groupby("player_id", sort=False):
        if pid not in eligible_set:
            continue
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        q_idx = idx[:query_k]
        gal_idx = idx[query_k:]
        if len(gal_idx) == 0:
            continue
        query_rows.append(df.loc[q_idx])
        gallery_rows.append(df.loc[gal_idx])
        kept.append(pid)
    if not query_rows:
        raise RuntimeError("No eligible players for holdout; lower min_games or sample more.")
    return (
        pd.concat(gallery_rows, ignore_index=True),
        pd.concat(query_rows, ignore_index=True),
        kept,
    )



def run(
    task_dir: Path,
    out_dir: Path,
    max_games_per_rank: int = 3000,
    min_games: int = 5,
    query_k: int = 3,
    max_players: int = 1500,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"[task1] loading sampled features from {task_dir} …")
    df = load_sampled_features(
        task_dir, max_games_per_rank=max_games_per_rank, seed=seed
    )
    print(
        f"[task1] loaded {len(df)} games, "
        f"{df['player_id'].nunique()} players, ranks={sorted(df['rank'].unique())}"
    )

    gallery_df, query_df, players = holdout_split(
        df,
        min_games=min_games,
        query_k=query_k,
        max_players=max_players,
        seed=seed,
    )
    print(
        f"[task1] holdout: {len(players)} players, "
        f"gallery={len(gallery_df)}, query_games={len(query_df)} (K={query_k})"
    )

    profiles = build_profiles(gallery_df)
    # Only score against players that still have a gallery profile
    cand_ids = list(profiles.keys())

    scores = []
    pred_rows = []
    for pid, qg in query_df.groupby("player_id", sort=False):
        if pid not in profiles:
            continue
        q = query_vector(qg)
        sims = [(cid, score_pair(q, profiles[cid])) for cid in cand_ids]
        sims.sort(key=lambda x: -x[1])
        top5 = [c for c, _ in sims[:5]]
        sc = top5_exp_decay_score(pid, top5)
        scores.append(sc)
        pred_rows.append(
            {
                "true_player_id": pid,
                "pred_1": top5[0] if len(top5) > 0 else "",
                "pred_2": top5[1] if len(top5) > 1 else "",
                "pred_3": top5[2] if len(top5) > 2 else "",
                "pred_4": top5[3] if len(top5) > 3 else "",
                "pred_5": top5[4] if len(top5) > 4 else "",
                "score": sc,
                "n_query_games": len(qg),
                "true_rank": profiles[pid]["rank"],
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    hit_at_1 = float(np.mean([1.0 if r["pred_1"] == r["true_player_id"] else 0.0 for r in pred_rows])) if pred_rows else 0.0
    hit_at_5 = float(np.mean([1.0 if r["score"] > 0 else 0.0 for r in pred_rows])) if pred_rows else 0.0

    metrics = {
        "task": "task1_identifying_players",
        "metric": "top5_exponential_decay",
        "mean_score": mean_score,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "n_queries": len(scores),
        "n_candidates": len(cand_ids),
        "sample": {
            "max_games_per_rank": max_games_per_rank,
            "min_games_per_player": min_games,
            "query_k": query_k,
            "max_players": max_players,
            "seed": seed,
            "n_games_loaded": int(len(df)),
            "n_players_loaded": int(df["player_id"].nunique()),
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local holdout validation only — not an official leaderboard score.",
    }

    ensure_dir(out_dir)
    metrics_path = out_dir / "task1_baseline_metrics.json"
    preds_path = out_dir / "task1_baseline_preds_sample.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).head(200).to_csv(preds_path, index=False)

    print(f"[task1] mean Top-5 exp-decay score = {mean_score:.4f}")
    print(f"[task1] hit@1={hit_at_1:.4f}  hit@5={hit_at_5:.4f}")
    print(f"[task1] wrote {metrics_path}")
    print(f"[task1] wrote {preds_path}")
    print(f"[task1] elapsed {metrics['elapsed_sec']}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AI CUP 2026 Task1 baseline")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=3000)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--query-k", type=int, default=3)
    ap.add_argument("--max-players", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    task_dir = data_dir("task1", args.data_root)
    run(
        task_dir=task_dir,
        out_dir=args.out_dir,
        max_games_per_rank=args.max_games_per_rank,
        min_games=args.min_games,
        query_k=args.query_k,
        max_players=args.max_players,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
