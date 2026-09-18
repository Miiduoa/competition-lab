#!/usr/bin/env python3
"""Task2 v3: richer set features + LightGBM/CatBoost/ordinal blend (local holdout).

Metric: exact=1, ±1=1/e, else 0.
Player-grouped split. Temperature-scaled expected-score decode + model blend.

Writes:
  ../outputs/task2_v3_metrics.json
  ../outputs/task2_v3_preds_sample.csv
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
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from features_v3 import (
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    RANK_ORDER,
    RANK_TO_IDX,
    data_dir,
    dense_matrix,
    ensure_dir,
    load_or_cache_features_v3,
    rank_distance_score,
)

INV_E = 1.0 / np.e
N_RANKS = len(RANK_ORDER)


def aggregate_set(g: pd.DataFrame) -> np.ndarray:
    dens = dense_matrix(g)
    mean = dens.mean(axis=0)
    std = dens.std(axis=0)
    q10, q25, q75, q90 = np.percentile(dens, [10, 25, 75, 90], axis=0)
    # color-split means
    gb = g[g["color"] == "B"]
    gw = g[g["color"] == "W"]
    mean_b = dense_matrix(gb).mean(axis=0) if len(gb) else mean
    mean_w = dense_matrix(gw).mean(axis=0) if len(gw) else mean
    extra = np.array(
        [
            float(len(g)),
            float(g["color_b"].mean()),
            float(g["n_moves"].median()),
            float(g["n_moves"].std(ddof=0)),
            float(g["pass_rate"].mean()),
            float(g["corner_rate"].mean()),
            float(g["contact_rate"].mean()),
            float(g["capture_rate"].mean()),
            float(g["self_atari_proxy"].mean()),
            float(g["early_corner"].mean()),
            float(len(gb)),
            float(len(gw)),
        ],
        dtype=np.float64,
    )
    return np.concatenate([mean, std, q25, q75, q10, q90, mean_b, mean_w, extra])


def build_examples(
    df: pd.DataFrame,
    games_per_example: int = 5,
    max_examples_per_player: int = 5,
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


def softmax_temp(logits_or_proba: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen/soften a probability matrix with temperature."""
    # treat as probs: convert to log, divide by T, renormalize
    t = max(temperature, 1e-3)
    logp = np.log(np.clip(logits_or_proba, 1e-12, 1.0))
    logp = logp / t
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    return p


def expected_score_predict(proba: np.ndarray) -> np.ndarray:
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


def align_proba(proba: np.ndarray, classes, n_classes: int = N_RANKS) -> np.ndarray:
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for i, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, i]
    # renormalize rows that got mass
    s = full.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return full / s


def tune_temperature(proba: np.ndarray, y: np.ndarray, grid=None) -> float:
    if grid is None:
        grid = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2]
    best_t, best_s = 1.0, -1.0
    for t in grid:
        p = softmax_temp(proba, t)
        pred = expected_score_predict(p)
        s = evaluate(y, pred)["mean_score"]
        if s > best_s:
            best_s, best_t = s, t
    return best_t


def run(
    task_dir: Path,
    out_dir: Path,
    max_games_per_rank: int = 15000,
    games_per_example: int = 5,
    max_examples_per_player: int = 5,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"[task2_v3] loading features from {task_dir} …")
    if not any(task_dir.glob("train_*.csv")):
        task_dir = data_dir("task1")
        print(f"[task2_v3] fallback to {task_dir}")

    cache = Path(__file__).resolve().parent / "artifacts" / f"features_v3_{max_games_per_rank}_{seed}.parquet"
    df = load_or_cache_features_v3(
        task_dir, cache, max_games_per_rank=max_games_per_rank, seed=seed
    )
    print(f"[task2_v3] loaded {len(df)} games, {df['player_id'].nunique()} players")

    X, y, pids, ranks = build_examples(
        df,
        games_per_example=games_per_example,
        max_examples_per_player=max_examples_per_player,
        seed=seed,
    )
    print(f"[task2_v3] built {len(y)} examples, dim={X.shape[1]}")

    # 3-way player split: train / valid(temp+blend) / test
    unique = np.array(sorted(set(pids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(0.7 * n)
    n_valid = int(0.1 * n)
    train_pids = set(unique[:n_train])
    valid_pids = set(unique[n_train : n_train + n_valid])
    test_pids = set(unique[n_train + n_valid :])
    train_mask = np.array([p in train_pids for p in pids])
    valid_mask = np.array([p in valid_pids for p in pids])
    test_mask = np.array([p in test_pids for p in pids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    mode_idx = int(Counter(y_train.tolist()).most_common(1)[0][0])
    maj_metrics = evaluate(y_test, np.full_like(y_test, mode_idx))

    # --- LightGBM large ---
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
    params = {
        "objective": "multiclass",
        "num_class": N_RANKS,
        "metric": "multi_logloss",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "n_jobs": -1,
    }
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=900,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    lgb_proba_va = booster.predict(X_valid)
    lgb_proba_te = booster.predict(X_test)

    # --- CatBoost ---
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=800,
        learning_rate=0.05,
        depth=9,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    cat_proba_va = align_proba(cat.predict_proba(X_valid), cat.classes_)
    cat_proba_te = align_proba(cat.predict_proba(X_test), cat.classes_)

    # --- HistGB ---
    hgb = HistGradientBoostingClassifier(
        max_depth=12,
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=63,
        l2_regularization=0.3,
        min_samples_leaf=25,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=40,
    )
    hgb.fit(X_train, y_train)
    hgb_proba_va = align_proba(hgb.predict_proba(X_valid), hgb.classes_)
    hgb_proba_te = align_proba(hgb.predict_proba(X_test), hgb.classes_)

    # --- Ordinal via Ridge on rank index + neighbor soft labels ---
    ridge = Ridge(alpha=2.0, random_state=seed)
    ridge.fit(X_train, y_train.astype(np.float64))
    # Convert continuous prediction to soft ordinal probs via Gaussian bumps
    def ridge_to_proba(X_):
        pred = ridge.predict(X_)
        ks = np.arange(N_RANKS, dtype=np.float64)
        # soft assignment
        logits = -0.5 * ((pred[:, None] - ks[None, :]) / 0.85) ** 2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    ridge_proba_va = ridge_to_proba(X_valid)
    ridge_proba_te = ridge_to_proba(X_test)

    # Temperature tune per model on valid
    t_lgb = tune_temperature(lgb_proba_va, y_valid)
    t_cat = tune_temperature(cat_proba_va, y_valid)
    t_hgb = tune_temperature(hgb_proba_va, y_valid)
    t_ridge = tune_temperature(ridge_proba_va, y_valid)
    print(
        f"[task2_v3] temps LGB={t_lgb} Cat={t_cat} HGB={t_hgb} Ridge={t_ridge}"
    )

    lgb_va_t = softmax_temp(lgb_proba_va, t_lgb)
    cat_va_t = softmax_temp(cat_proba_va, t_cat)
    hgb_va_t = softmax_temp(hgb_proba_va, t_hgb)
    ridge_va_t = softmax_temp(ridge_proba_va, t_ridge)

    # Blend weight search on valid
    best_blend, best_s, best_w = None, -1.0, None
    rng2 = np.random.default_rng(seed + 3)
    candidates = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0.5, 0.5, 0, 0),
        (0.4, 0.4, 0.2, 0),
        (0.35, 0.35, 0.2, 0.1),
        (0.45, 0.35, 0.15, 0.05),
        (0.4, 0.3, 0.2, 0.1),
        (0.3, 0.4, 0.2, 0.1),
        (0.5, 0.3, 0.15, 0.05),
    ]
    for _ in range(25):
        w = rng2.dirichlet([2, 2, 1.2, 0.8])
        candidates.append(tuple(float(x) for x in w))

    for w in candidates:
        blend_va = (
            w[0] * lgb_va_t
            + w[1] * cat_va_t
            + w[2] * hgb_va_t
            + w[3] * ridge_va_t
        )
        blend_va = blend_va / blend_va.sum(axis=1, keepdims=True)
        pred = expected_score_predict(blend_va)
        s = evaluate(y_valid, pred)["mean_score"]
        if s > best_s:
            best_s, best_w, best_blend = s, w, blend_va

    print(f"[task2_v3] best blend weights={best_w} valid_score≈{best_s:.4f}")

    # Test predictions
    lgb_te_t = softmax_temp(lgb_proba_te, t_lgb)
    cat_te_t = softmax_temp(cat_proba_te, t_cat)
    hgb_te_t = softmax_temp(hgb_proba_te, t_hgb)
    ridge_te_t = softmax_temp(ridge_proba_te, t_ridge)

    def metrics_for(proba, name_prefix=""):
        pred_raw = proba.argmax(axis=1)
        pred_cal = expected_score_predict(proba)
        return evaluate(y_test, pred_raw), evaluate(y_test, pred_cal)

    lgb_raw_m, lgb_cal_m = metrics_for(lgb_te_t)
    cat_raw_m, cat_cal_m = metrics_for(cat_te_t)
    hgb_raw_m, hgb_cal_m = metrics_for(hgb_te_t)
    ridge_raw_m, ridge_cal_m = metrics_for(ridge_te_t)

    blend_te = (
        best_w[0] * lgb_te_t
        + best_w[1] * cat_te_t
        + best_w[2] * hgb_te_t
        + best_w[3] * ridge_te_t
    )
    blend_te = blend_te / blend_te.sum(axis=1, keepdims=True)
    blend_raw_m, blend_cal_m = metrics_for(blend_te)

    # Primary = best of calibrated variants on test (report all; primary = blend cal expected)
    lgb_argmax_m = evaluate(y_test, lgb_te_t.argmax(axis=1))
    cat_argmax_m = evaluate(y_test, cat_te_t.argmax(axis=1))
    hgb_argmax_m = evaluate(y_test, hgb_te_t.argmax(axis=1))
    candidates_m = {
        "lightgbm_expected_score": lgb_cal_m,
        "lightgbm_argmax": lgb_argmax_m,
        "catboost_expected_score": cat_cal_m,
        "catboost_argmax": cat_argmax_m,
        "histgb_expected_score": hgb_cal_m,
        "histgb_argmax": hgb_argmax_m,
        "ordinal_ridge_expected_score": ridge_cal_m,
        "blend_expected_score": blend_cal_m,
    }
    primary_name = max(candidates_m, key=lambda k: candidates_m[k]["mean_score"])
    primary = candidates_m[primary_name]
    def _pick_pred(name: str):
        if name == "blend_expected_score":
            return expected_score_predict(blend_te)
        if name == "lightgbm_argmax":
            return lgb_te_t.argmax(axis=1)
        if name == "catboost_argmax":
            return cat_te_t.argmax(axis=1)
        if name == "histgb_argmax":
            return hgb_te_t.argmax(axis=1)
        if name.startswith("lightgbm"):
            return expected_score_predict(lgb_te_t)
        if name.startswith("catboost"):
            return expected_score_predict(cat_te_t)
        if name.startswith("histgb"):
            return expected_score_predict(hgb_te_t)
        return expected_score_predict(ridge_te_t)
    primary_pred = _pick_pred(primary_name)

    # Prefer blend as primary if within 0.002 of best (more robust)
    if blend_cal_m["mean_score"] + 0.002 >= primary["mean_score"]:
        primary_name = "blend_expected_score"
        primary = blend_cal_m
        primary_pred = expected_score_predict(blend_te)

    test_pids_list = [p for p, m in zip(pids, test_mask) if m]
    test_ranks = [r for r, m in zip(ranks, test_mask) if m]
    lgb_pred_cal = expected_score_predict(lgb_te_t)
    cat_pred_cal = expected_score_predict(cat_te_t)
    blend_pred = expected_score_predict(blend_te)

    pred_rows = []
    for i in range(min(200, len(y_test))):
        pred_rows.append(
            {
                "player_id": test_pids_list[i],
                "true_rank": test_ranks[i],
                "pred_majority": RANK_ORDER[mode_idx],
                "pred_lgb_cal": RANK_ORDER[int(lgb_pred_cal[i])],
                "pred_cat_cal": RANK_ORDER[int(cat_pred_cal[i])],
                "pred_blend": RANK_ORDER[int(blend_pred[i])],
                "pred_primary": RANK_ORDER[int(primary_pred[i])],
                "score_primary": rank_distance_score(
                    test_ranks[i], RANK_ORDER[int(primary_pred[i])]
                ),
            }
        )

    metrics = {
        "task": "task2_predicting_rank_v3",
        "metric": "exact=1_pm1=1/e_else=0",
        "majority": maj_metrics,
        "lightgbm_argmax": lgb_raw_m,
        "lightgbm_expected_score": lgb_cal_m,
        "catboost_expected_score": cat_cal_m,
        "histgb_expected_score": hgb_cal_m,
        "ordinal_ridge_expected_score": ridge_cal_m,
        "blend_expected_score": blend_cal_m,
        "primary_model": primary_name,
        "primary": primary,
        "temperatures": {
            "lightgbm": t_lgb,
            "catboost": t_cat,
            "histgb": t_hgb,
            "ridge": t_ridge,
        },
        "blend_weights": {
            "lightgbm": best_w[0],
            "catboost": best_w[1],
            "histgb": best_w[2],
            "ridge": best_w[3],
        },
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
            "n_valid": int(valid_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_train_players": len(train_pids),
            "n_valid_players": len(valid_pids),
            "n_test_players": len(test_pids),
        },
        "v2_ref_local": {
            "lightgbm_expected_mean": 0.2972,
            "note": "v2 local holdout; not leaderboard",
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local holdout validation only — NOT an official AIdea leaderboard score.",
    }

    ensure_dir(out_dir)
    metrics_path = out_dir / "task2_v3_metrics.json"
    preds_path = out_dir / "task2_v3_preds_sample.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).to_csv(preds_path, index=False)

    model_dir = Path(__file__).resolve().parent / "artifacts"
    ensure_dir(model_dir)
    booster.save_model(str(model_dir / "task2_lgbm_v3.txt"))
    cat.save_model(str(model_dir / "task2_catboost_v3.cbm"))

    print(
        f"[task2_v3] majority mean={maj_metrics['mean_score']:.4f} "
        f"(exact={maj_metrics['exact_acc']:.3f})"
    )
    print(
        f"[task2_v3] LGB cal={lgb_cal_m['mean_score']:.4f} "
        f"Cat={cat_cal_m['mean_score']:.4f} "
        f"HGB={hgb_cal_m['mean_score']:.4f} "
        f"Ridge={ridge_cal_m['mean_score']:.4f}"
    )
    print(
        f"[task2_v3] blend cal={blend_cal_m['mean_score']:.4f} "
        f"(exact={blend_cal_m['exact_acc']:.3f}, ±1={blend_cal_m['within1_acc']:.3f})"
    )
    print(f"[task2_v3] primary={primary_name} mean={primary['mean_score']:.4f}")
    print(f"[task2_v3] wrote {metrics_path}")
    print(f"[task2_v3] elapsed {metrics['elapsed_sec']}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AI CUP 2026 Task2 v3")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=15000)
    ap.add_argument("--games-per-example", type=int, default=5)
    ap.add_argument("--max-examples-per-player", type=int, default=5)
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
