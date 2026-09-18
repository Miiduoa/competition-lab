#!/usr/bin/env python3
"""Task1 v2: richer profiles + TF-IDF openings + dense cosine (local holdout).

Writes:
  ../outputs/task1_v2_metrics.json
  ../outputs/task1_v2_preds_sample.csv
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from features_v2 import (
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    cosine_sim,
    data_dir,
    dense_matrix,
    ensure_dir,
    jaccard_from_counters,
    load_sampled_features_v2,
    top5_exp_decay_score,
)


def holdout_split(
    df: pd.DataFrame,
    min_games: int = 6,
    query_k: int = 3,
    max_players: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    rng = np.random.default_rng(seed)
    counts = df.groupby("player_id").size()
    eligible = counts[counts >= min_games].index.tolist()
    if len(eligible) > max_players:
        eligible = list(rng.choice(eligible, size=max_players, replace=False))
    eligible_set = set(eligible)
    gallery_rows, query_rows, kept = [], [], []
    for pid, g in df.groupby("player_id", sort=False):
        if pid not in eligible_set:
            continue
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        q_idx, gal_idx = idx[:query_k], idx[query_k:]
        if len(gal_idx) == 0:
            continue
        query_rows.append(df.loc[q_idx])
        gallery_rows.append(df.loc[gal_idx])
        kept.append(pid)
    if not query_rows:
        raise RuntimeError("No eligible players for holdout")
    return (
        pd.concat(gallery_rows, ignore_index=True),
        pd.concat(query_rows, ignore_index=True),
        kept,
    )


def build_profiles(gallery_df: pd.DataFrame) -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    for pid, g in gallery_df.groupby("player_id", sort=False):
        dens = dense_matrix(g)
        mean_v = dens.mean(axis=0)
        std_v = dens.std(axis=0)
        profiles[pid] = {
            "n_games": len(g),
            "mean_vec": mean_v,
            "std_vec": std_v,
            "opening_counter": Counter(g["opening"].tolist()),
            "opening8_counter": Counter(g["opening8"].tolist()),
            "first_counter": Counter(g["first_coord"].tolist()),
            "color_b_rate": float(g["color_b"].mean()),
            "mean_moves": float(g["n_moves"].mean()),
            "std_moves": float(g["n_moves"].std(ddof=0)),
            "doc": " ".join(g["opening_tokens"].tolist()),
            "rank": str(g["rank"].iloc[0]),
        }
    return profiles


def query_pack(qg: pd.DataFrame) -> dict:
    dens = dense_matrix(qg)
    return {
        "mean_vec": dens.mean(axis=0),
        "std_vec": dens.std(axis=0),
        "opening_counter": Counter(qg["opening"].tolist()),
        "opening8_counter": Counter(qg["opening8"].tolist()),
        "first_counter": Counter(qg["first_coord"].tolist()),
        "color_b_rate": float(qg["color_b"].mean()),
        "mean_moves": float(qg["n_moves"].mean()),
        "doc": " ".join(qg["opening_tokens"].tolist()),
    }


def run(
    task_dir: Path,
    out_dir: Path,
    max_games_per_rank: int = 7000,
    min_games: int = 6,
    query_k: int = 3,
    max_players: int = 2000,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"[task1_v2] loading richer features from {task_dir} …")
    df = load_sampled_features_v2(
        task_dir, max_games_per_rank=max_games_per_rank, seed=seed
    )
    print(
        f"[task1_v2] loaded {len(df)} games, "
        f"{df['player_id'].nunique()} players"
    )

    gallery_df, query_df, players = holdout_split(
        df,
        min_games=min_games,
        query_k=query_k,
        max_players=max_players,
        seed=seed,
    )
    print(
        f"[task1_v2] holdout: {len(players)} players, "
        f"gallery={len(gallery_df)}, query_games={len(query_df)}"
    )

    profiles = build_profiles(gallery_df)
    cand_ids = list(profiles.keys())
    n_cand = len(cand_ids)

    # TF-IDF on opening token documents
    docs = [profiles[pid]["doc"] for pid in cand_ids]
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )
    gal_tfidf = vectorizer.fit_transform(docs)
    gal_tfidf = normalize(gal_tfidf)

    # Dense gallery matrix (L2-normalized mean vectors)
    gal_dense = np.vstack([profiles[pid]["mean_vec"] for pid in cand_ids])
    # Scale important dims lightly: leave as-is then L2 norm
    gal_dense_n = gal_dense / (np.linalg.norm(gal_dense, axis=1, keepdims=True) + 1e-12)

    scores = []
    pred_rows = []
    for pid, qg in query_df.groupby("player_id", sort=False):
        if pid not in profiles:
            continue
        q = query_pack(qg)
        q_vec = q["mean_vec"]
        q_vec_n = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        dense_sims = gal_dense_n @ q_vec_n  # (n_cand,)

        q_tf = vectorizer.transform([q["doc"]])
        q_tf = normalize(q_tf)
        tfidf_sims = (gal_tfidf @ q_tf.T).toarray().ravel()

        # Classical Jaccard signals
        open_sims = np.array(
            [
                jaccard_from_counters(q["opening_counter"], profiles[c]["opening_counter"])
                for c in cand_ids
            ],
            dtype=np.float64,
        )
        open8_sims = np.array(
            [
                jaccard_from_counters(
                    q["opening8_counter"], profiles[c]["opening8_counter"]
                )
                for c in cand_ids
            ],
            dtype=np.float64,
        )
        first_sims = np.array(
            [
                jaccard_from_counters(q["first_counter"], profiles[c]["first_counter"])
                for c in cand_ids
            ],
            dtype=np.float64,
        )
        color_sims = np.array(
            [
                1.0 - abs(q["color_b_rate"] - profiles[c]["color_b_rate"])
                for c in cand_ids
            ],
            dtype=np.float64,
        )
        move_sims = np.array(
            [
                float(np.exp(-abs(q["mean_moves"] - profiles[c]["mean_moves"]) / 400.0))
                for c in cand_ids
            ],
            dtype=np.float64,
        )

        # Blend (weights tuned for distinctive openings + style denseness)
        blend = (
            0.28 * dense_sims
            + 0.32 * tfidf_sims
            + 0.18 * open_sims
            + 0.08 * open8_sims
            + 0.07 * first_sims
            + 0.04 * color_sims
            + 0.03 * move_sims
        )
        top_idx = np.argpartition(-blend, min(5, n_cand - 1))[:5]
        top_idx = top_idx[np.argsort(-blend[top_idx])]
        top5 = [cand_ids[i] for i in top_idx]
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
                "best_sim": float(blend[top_idx[0]]) if len(top_idx) else 0.0,
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    hit_at_1 = (
        float(np.mean([1.0 if r["pred_1"] == r["true_player_id"] else 0.0 for r in pred_rows]))
        if pred_rows
        else 0.0
    )
    hit_at_5 = (
        float(np.mean([1.0 if r["score"] > 0 else 0.0 for r in pred_rows]))
        if pred_rows
        else 0.0
    )

    metrics = {
        "task": "task1_identifying_players_v2",
        "metric": "top5_exponential_decay",
        "mean_score": mean_score,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "n_queries": len(scores),
        "n_candidates": n_cand,
        "model": {
            "dense_features": FEATURE_NAMES,
            "tfidf_max_features": 8000,
            "blend_weights": {
                "dense": 0.28,
                "tfidf": 0.32,
                "opening12_jaccard": 0.18,
                "opening8_jaccard": 0.08,
                "first": 0.07,
                "color": 0.04,
                "mean_moves": 0.03,
            },
        },
        "sample": {
            "max_games_per_rank": max_games_per_rank,
            "min_games_per_player": min_games,
            "query_k": query_k,
            "max_players": max_players,
            "seed": seed,
            "n_games_loaded": int(len(df)),
            "n_players_loaded": int(df["player_id"].nunique()),
        },
        "baseline_ref_local": {
            "mean_score": 0.02197,
            "note": "baseline task1 local holdout (~0.022); not leaderboard",
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local holdout validation only — NOT an official AIdea leaderboard score.",
    }

    ensure_dir(out_dir)
    metrics_path = out_dir / "task1_v2_metrics.json"
    preds_path = out_dir / "task1_v2_preds_sample.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).head(200).to_csv(preds_path, index=False)

    print(f"[task1_v2] mean Top-5 exp-decay = {mean_score:.4f}")
    print(f"[task1_v2] hit@1={hit_at_1:.4f}  hit@5={hit_at_5:.4f}")
    print(f"[task1_v2] wrote {metrics_path}")
    print(f"[task1_v2] wrote {preds_path}")
    print(f"[task1_v2] elapsed {metrics['elapsed_sec']}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AI CUP 2026 Task1 v2")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=7000)
    ap.add_argument("--min-games", type=int, default=6)
    ap.add_argument("--query-k", type=int, default=3)
    ap.add_argument("--max-players", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(
        task_dir=data_dir("task1", args.data_root),
        out_dir=args.out_dir,
        max_games_per_rank=args.max_games_per_rank,
        min_games=args.min_games,
        query_k=args.query_k,
        max_players=args.max_players,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
