#!/usr/bin/env python3
"""Task1 v3: stronger features + dual TF-IDF + ANN shortlist + weight tuning.

Local holdout only — NOT AIdea leaderboard.
Writes:
  ../outputs/task1_v3_metrics.json
  ../outputs/task1_v3_preds_sample.csv
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from features_v3 import (
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    data_dir,
    dense_matrix,
    ensure_dir,
    jaccard_from_counters,
    load_or_cache_features_v3,
    top5_exp_decay_score,
)


def holdout_split(
    df: pd.DataFrame,
    min_games: int = 8,
    query_k: int = 3,
    max_players: int = 2500,
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


def clean_gallery_group(g: pd.DataFrame, keep_frac: float = 0.85) -> pd.DataFrame:
    """Drop outlier games farthest from player dense mean (gallery cleaning)."""
    if len(g) < 6:
        return g
    dens = dense_matrix(g)
    mean = dens.mean(axis=0)
    dist = np.linalg.norm(dens - mean, axis=1)
    k = max(3, int(np.ceil(len(g) * keep_frac)))
    keep_idx = np.argsort(dist)[:k]
    return g.iloc[keep_idx]


def build_profiles(gallery_df: pd.DataFrame, clean: bool = True) -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    for pid, g in gallery_df.groupby("player_id", sort=False):
        if clean:
            g = clean_gallery_group(g)
        dens = dense_matrix(g)
        mean_v = dens.mean(axis=0)
        # color-conditioned means
        gb = g[g["color"] == "B"]
        gw = g[g["color"] == "W"]
        mean_b = dense_matrix(gb).mean(axis=0) if len(gb) else mean_v
        mean_w = dense_matrix(gw).mean(axis=0) if len(gw) else mean_v
        profiles[pid] = {
            "n_games": len(g),
            "mean_vec": mean_v,
            "mean_b": mean_b,
            "mean_w": mean_w,
            "std_vec": dens.std(axis=0),
            "opening_counter": Counter(g["opening"].tolist()),
            "opening12_counter": Counter(g["opening12"].tolist()),
            "opening8_counter": Counter(g["opening8"].tolist()),
            "first_counter": Counter(g["first_coord"].tolist()),
            "color_b_rate": float(np.asarray(g["color_b"]).mean()),
            "mean_moves": float(np.asarray(g["n_moves"]).mean()),
            "doc": " ".join(g["opening_tokens"].tolist()),
            "pat_doc": " ".join(g["pattern_tokens"].tolist()),
            "rank": str(g["rank"].iloc[0]),
        }
    return profiles


def query_pack(qg: pd.DataFrame) -> dict:
    dens = dense_matrix(qg)
    mean_v = dens.mean(axis=0)
    gb = qg[qg["color"] == "B"]
    gw = qg[qg["color"] == "W"]
    mean_b = dense_matrix(gb).mean(axis=0) if len(gb) else mean_v
    mean_w = dense_matrix(gw).mean(axis=0) if len(gw) else mean_v
    return {
        "mean_vec": mean_v,
        "mean_b": mean_b,
        "mean_w": mean_w,
        "opening_counter": Counter(qg["opening"].tolist()),
        "opening12_counter": Counter(qg["opening12"].tolist()),
        "opening8_counter": Counter(qg["opening8"].tolist()),
        "first_counter": Counter(qg["first_coord"].tolist()),
        "color_b_rate": float(np.asarray(qg["color_b"]).mean()),
        "mean_moves": float(np.asarray(qg["n_moves"]).mean()),
        "doc": " ".join(qg["opening_tokens"].tolist()),
        "pat_doc": " ".join(qg["pattern_tokens"].tolist()),
    }


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


DEFAULT_WEIGHTS = {
    "dense": 0.22,
    "dense_color": 0.10,
    "tfidf": 0.26,
    "pat_tfidf": 0.14,
    "opening16_jaccard": 0.12,
    "opening12_jaccard": 0.06,
    "opening8_jaccard": 0.04,
    "first": 0.03,
    "color": 0.02,
    "mean_moves": 0.01,
}


def blend_scores(
    components: Dict[str, np.ndarray], weights: Dict[str, float]
) -> np.ndarray:
    out = None
    for k, w in weights.items():
        if k not in components:
            continue
        term = w * components[k]
        out = term if out is None else out + term
    return out


def score_queries(
    query_df: pd.DataFrame,
    profiles: Dict[str, dict],
    cand_ids: List[str],
    gal_dense_n: np.ndarray,
    gal_b_n: np.ndarray,
    gal_w_n: np.ndarray,
    gal_tfidf,
    gal_pat,
    vectorizer,
    pat_vectorizer,
    nn_index: NearestNeighbors,
    shortlist: int,
    weights: Dict[str, float],
) -> Tuple[List[float], List[dict]]:
    n_cand = len(cand_ids)
    # Precompute opening counters arrays lazily via lists
    open_cs = [profiles[c]["opening_counter"] for c in cand_ids]
    open12_cs = [profiles[c]["opening12_counter"] for c in cand_ids]
    open8_cs = [profiles[c]["opening8_counter"] for c in cand_ids]
    first_cs = [profiles[c]["first_counter"] for c in cand_ids]
    color_rates = np.array([profiles[c]["color_b_rate"] for c in cand_ids])
    mean_moves = np.array([profiles[c]["mean_moves"] for c in cand_ids])

    scores, pred_rows = [], []
    for pid, qg in query_df.groupby("player_id", sort=False):
        if pid not in profiles:
            continue
        q = query_pack(qg)
        q_vec_n = _l2(q["mean_vec"][None, :])[0]
        q_b_n = _l2(q["mean_b"][None, :])[0]
        q_w_n = _l2(q["mean_w"][None, :])[0]

        # ANN shortlist on dense
        k_nn = min(shortlist, n_cand)
        _, nn_idx = nn_index.kneighbors(q_vec_n.reshape(1, -1), n_neighbors=k_nn)
        cand_idx = nn_idx[0]

        # Also pull top from tfidf quickly
        q_tf = normalize(vectorizer.transform([q["doc"]]))
        tf_full = (gal_tfidf @ q_tf.T).toarray().ravel()
        top_tf = np.argpartition(-tf_full, min(shortlist, n_cand) - 1)[
            : min(shortlist, n_cand)
        ]
        cand_idx = np.unique(np.concatenate([cand_idx, top_tf]))

        dense_sims = gal_dense_n[cand_idx] @ q_vec_n
        # color-conditioned: average of B-B and W-W sims
        color_dense = 0.5 * (gal_b_n[cand_idx] @ q_b_n + gal_w_n[cand_idx] @ q_w_n)
        tfidf_sims = tf_full[cand_idx]
        q_pat = normalize(pat_vectorizer.transform([q["pat_doc"]]))
        pat_sims = (gal_pat[cand_idx] @ q_pat.T).toarray().ravel()

        open_sims = np.array(
            [jaccard_from_counters(q["opening_counter"], open_cs[i]) for i in cand_idx],
            dtype=np.float64,
        )
        open12_sims = np.array(
            [
                jaccard_from_counters(q["opening12_counter"], open12_cs[i])
                for i in cand_idx
            ],
            dtype=np.float64,
        )
        open8_sims = np.array(
            [
                jaccard_from_counters(q["opening8_counter"], open8_cs[i])
                for i in cand_idx
            ],
            dtype=np.float64,
        )
        first_sims = np.array(
            [jaccard_from_counters(q["first_counter"], first_cs[i]) for i in cand_idx],
            dtype=np.float64,
        )
        color_sims = 1.0 - np.abs(q["color_b_rate"] - color_rates[cand_idx])
        move_sims = np.exp(-np.abs(q["mean_moves"] - mean_moves[cand_idx]) / 400.0)

        components = {
            "dense": dense_sims,
            "dense_color": color_dense,
            "tfidf": tfidf_sims,
            "pat_tfidf": pat_sims,
            "opening16_jaccard": open_sims,
            "opening12_jaccard": open12_sims,
            "opening8_jaccard": open8_sims,
            "first": first_sims,
            "color": color_sims,
            "mean_moves": move_sims,
        }
        blend = blend_scores(components, weights)
        top_local = np.argpartition(-blend, min(5, len(blend) - 1))[:5]
        top_local = top_local[np.argsort(-blend[top_local])]
        top5 = [cand_ids[cand_idx[i]] for i in top_local]
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
                "best_sim": float(blend[top_local[0]]) if len(top_local) else 0.0,
                "shortlist": int(len(cand_idx)),
            }
        )
    return scores, pred_rows


def tune_weights(
    query_df: pd.DataFrame,
    profiles,
    cand_ids,
    gal_dense_n,
    gal_b_n,
    gal_w_n,
    gal_tfidf,
    gal_pat,
    vectorizer,
    pat_vectorizer,
    nn_index,
    shortlist: int,
    seed: int = 42,
    n_trials: int = 40,
) -> Dict[str, float]:
    """Random search on a subset of query players (no leakage into final if split)."""
    rng = np.random.default_rng(seed)
    pids = query_df["player_id"].unique().tolist()
    if len(pids) < 80:
        return dict(DEFAULT_WEIGHTS)
    tune_pids = set(rng.choice(pids, size=min(400, len(pids) // 3), replace=False))
    tune_df = query_df[query_df["player_id"].isin(tune_pids)]
    keys = list(DEFAULT_WEIGHTS.keys())
    best_w, best_s = dict(DEFAULT_WEIGHTS), -1.0
    for t in range(n_trials):
        raw = rng.dirichlet(np.ones(len(keys)) * 1.5)
        w = {k: float(raw[i]) for i, k in enumerate(keys)}
        # slight bias toward known good channels
        w["tfidf"] = max(w["tfidf"], 0.15)
        w["dense"] = max(w["dense"], 0.12)
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
        scores, _ = score_queries(
            tune_df,
            profiles,
            cand_ids,
            gal_dense_n,
            gal_b_n,
            gal_w_n,
            gal_tfidf,
            gal_pat,
            vectorizer,
            pat_vectorizer,
            nn_index,
            shortlist,
            w,
        )
        mean_s = float(np.mean(scores)) if scores else 0.0
        if mean_s > best_s:
            best_s, best_w = mean_s, w
        if (t + 1) % 10 == 0:
            print(f"  [tune] trial {t+1}/{n_trials} best={best_s:.4f}")
    print(f"  [tune] selected mean on tune-set ≈ {best_s:.4f}")
    return best_w


def run(
    task_dir: Path,
    out_dir: Path,
    max_games_per_rank: int = 15000,
    min_games: int = 8,
    query_k: int = 3,
    max_players: int = 2500,
    shortlist: int = 250,
    tune: bool = True,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"[task1_v3] loading features from {task_dir} …")
    cache = Path(__file__).resolve().parent / "artifacts" / f"features_v3_{max_games_per_rank}_{seed}.parquet"
    df = load_or_cache_features_v3(
        task_dir, cache, max_games_per_rank=max_games_per_rank, seed=seed
    )
    print(
        f"[task1_v3] loaded {len(df)} games, {df['player_id'].nunique()} players"
    )

    gallery_df, query_df, players = holdout_split(
        df,
        min_games=min_games,
        query_k=query_k,
        max_players=max_players,
        seed=seed,
    )
    print(
        f"[task1_v3] holdout: {len(players)} players, "
        f"gallery={len(gallery_df)}, query_games={len(query_df)}"
    )

    profiles = build_profiles(gallery_df, clean=True)
    cand_ids = list(profiles.keys())
    n_cand = len(cand_ids)

    docs = [profiles[pid]["doc"] for pid in cand_ids]
    pat_docs = [profiles[pid]["pat_doc"] for pid in cand_ids]
    vectorizer = TfidfVectorizer(
        max_features=16000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    pat_vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    gal_tfidf = normalize(vectorizer.fit_transform(docs))
    gal_pat = normalize(pat_vectorizer.fit_transform(pat_docs))

    gal_dense = np.vstack([profiles[pid]["mean_vec"] for pid in cand_ids])
    gal_b = np.vstack([profiles[pid]["mean_b"] for pid in cand_ids])
    gal_w = np.vstack([profiles[pid]["mean_w"] for pid in cand_ids])
    gal_dense_n = _l2(gal_dense)
    gal_b_n = _l2(gal_b)
    gal_w_n = _l2(gal_w)

    nn_index = NearestNeighbors(n_neighbors=min(shortlist, n_cand), metric="cosine")
    nn_index.fit(gal_dense_n)

    # Split query players: tune on 30%, evaluate on remaining (honest)
    rng = np.random.default_rng(seed + 7)
    all_q_pids = query_df["player_id"].unique().tolist()
    rng.shuffle(all_q_pids)
    n_tune = int(0.3 * len(all_q_pids)) if tune else 0
    tune_pids = set(all_q_pids[:n_tune])
    eval_pids = set(all_q_pids[n_tune:]) if n_tune else set(all_q_pids)
    tune_df = query_df[query_df["player_id"].isin(tune_pids)]
    eval_df = query_df[query_df["player_id"].isin(eval_pids)]

    if tune and len(tune_pids) >= 50:
        print(f"[task1_v3] tuning weights on {len(tune_pids)} players …")
        weights = tune_weights(
            tune_df,
            profiles,
            cand_ids,
            gal_dense_n,
            gal_b_n,
            gal_w_n,
            gal_tfidf,
            gal_pat,
            vectorizer,
            pat_vectorizer,
            nn_index,
            shortlist,
            seed=seed,
            n_trials=36,
        )
    else:
        weights = dict(DEFAULT_WEIGHTS)

    print(f"[task1_v3] evaluating on {len(eval_pids)} players …")
    scores, pred_rows = score_queries(
        eval_df,
        profiles,
        cand_ids,
        gal_dense_n,
        gal_b_n,
        gal_w_n,
        gal_tfidf,
        gal_pat,
        vectorizer,
        pat_vectorizer,
        nn_index,
        shortlist,
        weights,
    )

    mean_score = float(np.mean(scores)) if scores else 0.0
    hit_at_1 = (
        float(
            np.mean(
                [1.0 if r["pred_1"] == r["true_player_id"] else 0.0 for r in pred_rows]
            )
        )
        if pred_rows
        else 0.0
    )
    hit_at_5 = (
        float(np.mean([1.0 if r["score"] > 0 else 0.0 for r in pred_rows]))
        if pred_rows
        else 0.0
    )

    metrics = {
        "task": "task1_identifying_players_v3",
        "metric": "top5_exponential_decay",
        "mean_score": mean_score,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "n_queries": len(scores),
        "n_candidates": n_cand,
        "model": {
            "dense_dim": len(FEATURE_NAMES),
            "tfidf_max_features": 16000,
            "pat_tfidf_max_features": 12000,
            "shortlist": shortlist,
            "gallery_cleaning": True,
            "color_conditioned_dense": True,
            "blend_weights": weights,
            "weight_tuning": bool(tune and len(tune_pids) >= 50),
            "n_tune_players": len(tune_pids),
            "n_eval_players": len(eval_pids),
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
        "v2_ref_local": {
            "mean_score_main": 0.0868,
            "mean_score_matched_protocol": 0.1108,
            "note": "v2 local holdout; not leaderboard",
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local holdout validation only — NOT an official AIdea leaderboard score.",
    }

    ensure_dir(out_dir)
    metrics_path = out_dir / "task1_v3_metrics.json"
    preds_path = out_dir / "task1_v3_preds_sample.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).head(200).to_csv(preds_path, index=False)

    print(f"[task1_v3] mean Top-5 exp-decay = {mean_score:.4f}")
    print(f"[task1_v3] hit@1={hit_at_1:.4f}  hit@5={hit_at_5:.4f}")
    print(f"[task1_v3] wrote {metrics_path}")
    print(f"[task1_v3] elapsed {metrics['elapsed_sec']}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AI CUP 2026 Task1 v3")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=15000)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--query-k", type=int, default=3)
    ap.add_argument("--max-players", type=int, default=2500)
    ap.add_argument("--shortlist", type=int, default=250)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(
        task_dir=data_dir("task1", args.data_root),
        out_dir=args.out_dir,
        max_games_per_rank=args.max_games_per_rank,
        min_games=args.min_games,
        query_k=args.query_k,
        max_players=args.max_players,
        shortlist=args.shortlist,
        tune=not args.no_tune,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
