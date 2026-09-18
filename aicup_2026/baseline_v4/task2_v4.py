#!/usr/bin/env python3
"""Task2 v4b: TF-IDF set features + ordinal/soft heads + refit train+valid.

Honest player-grouped holdout. Metric exact=1, ±1=1/e.
NOT AIdea leaderboard.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from features_v4 import (
    DEFAULT_OUTPUT_ROOT,
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


def aggregate_set_from_dense(
    dens, color_b, n_moves, pass_rate, corner_rate, contact_rate,
    capture_rate, self_atari, early_corner, mean_manh, std_manh,
    side_rate, center_rate,
) -> np.ndarray:
    mean = dens.mean(axis=0)
    std = dens.std(axis=0)
    q10, q25, q75, q90 = np.percentile(dens, [10, 25, 75, 90], axis=0)
    is_b = color_b >= 0.5
    mean_b = dens[is_b].mean(axis=0) if is_b.any() else mean
    mean_w = dens[~is_b].mean(axis=0) if (~is_b).any() else mean
    norms = np.linalg.norm(dens, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    dn = dens / norms
    gram = dn @ dn.T
    n = dens.shape[0]
    pair_cos = ((gram.sum() - np.trace(gram)) / (n * (n - 1))) if n > 1 else 1.0
    n_b = float(is_b.sum())
    extra = np.array(
        [
            float(n), float(color_b.mean()), float(np.median(n_moves)),
            float(n_moves.std(ddof=0)), float(pass_rate.mean()),
            float(corner_rate.mean()), float(contact_rate.mean()),
            float(capture_rate.mean()), float(self_atari.mean()),
            float(early_corner.mean()), n_b, float(n - n_b), float(pair_cos),
            float(n_moves.mean()), float(mean_manh.mean()), float(std_manh.mean()),
            float(side_rate.mean()), float(center_rate.mean()),
        ],
        dtype=np.float64,
    )
    return np.concatenate([mean, std, q25, q75, q10, q90, mean_b, mean_w, extra])


def build_examples(df, games_per_example=5, max_examples_per_player=6, seed=42):
    rng = np.random.default_rng(seed)
    print("[task2_v4] precomputing dense…", flush=True)
    dens_all = dense_matrix(df)
    cols = {
        k: df[k].to_numpy(dtype=np.float64)
        for k in [
            "color_b", "n_moves", "pass_rate", "corner_rate", "contact_rate",
            "capture_rate", "self_atari_proxy", "early_corner", "mean_manh",
            "std_manh", "side_rate", "center_rate",
        ]
    }
    open_tok = df["opening_tokens"].to_numpy()
    pat_tok = df["pattern_tokens"].to_numpy()

    Xs, ys, pids, ranks, texts = [], [], [], [], []
    groups = df.groupby("player_id", sort=False).indices
    for pid, ilocs in groups.items():
        if len(ilocs) < games_per_example:
            continue
        rank = str(df["rank"].iloc[ilocs[0]])
        if rank not in RANK_TO_IDX:
            continue
        idx = np.asarray(ilocs, dtype=np.int64).copy()
        rng.shuffle(idx)
        n_ex = min(max_examples_per_player, len(idx) // games_per_example)
        for i in range(n_ex):
            sl = idx[i * games_per_example : (i + 1) * games_per_example]
            feat = aggregate_set_from_dense(
                dens_all[sl], cols["color_b"][sl], cols["n_moves"][sl],
                cols["pass_rate"][sl], cols["corner_rate"][sl],
                cols["contact_rate"][sl], cols["capture_rate"][sl],
                cols["self_atari_proxy"][sl], cols["early_corner"][sl],
                cols["mean_manh"][sl], cols["std_manh"][sl],
                cols["side_rate"][sl], cols["center_rate"][sl],
            )
            Xs.append(feat)
            ys.append(RANK_TO_IDX[rank])
            pids.append(pid)
            ranks.append(rank)
            # join tokens from games in the set
            parts = []
            for j in sl:
                parts.append(str(open_tok[j]))
                parts.append(str(pat_tok[j]))
            texts.append(" ".join(parts))
    return np.vstack(Xs), np.array(ys, dtype=np.int64), pids, ranks, texts


def softmax_temp(proba, temperature):
    t = max(temperature, 1e-3)
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / t
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def expected_score_predict(proba, neighbor_w=INV_E):
    n, c = proba.shape
    exp = np.zeros_like(proba)
    for k in range(c):
        exp[:, k] = proba[:, k]
        if k - 1 >= 0:
            exp[:, k] += neighbor_w * proba[:, k - 1]
        if k + 1 < c:
            exp[:, k] += neighbor_w * proba[:, k + 1]
    return exp.argmax(axis=1)


def evaluate(y_true, y_pred):
    scores, exact, within1 = [], 0, 0
    for t, p in zip(y_true, y_pred):
        scores.append(rank_distance_score(RANK_ORDER[int(t)], RANK_ORDER[int(p)]))
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


def align_proba(proba, classes, n_classes=N_RANKS):
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for i, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, i]
    s = full.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return full / s


def tune_temperature(proba, y, grid=None):
    if grid is None:
        grid = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2]
    best_t, best_s = 1.0, -1.0
    for t in grid:
        s = evaluate(y, expected_score_predict(softmax_temp(proba, t)))["mean_score"]
        if s > best_s:
            best_s, best_t = s, t
    return best_t


def continuous_to_proba(pred, sigma):
    ks = np.arange(N_RANKS, dtype=np.float64)
    logits = -0.5 * ((pred[:, None] - ks[None, :]) / max(sigma, 0.15)) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def tune_sigma(pred, y):
    best_s, best = -1.0, 0.85
    for sig in [0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.55, 1.8]:
        sc = evaluate(y, expected_score_predict(continuous_to_proba(pred, sig)))["mean_score"]
        if sc > best_s:
            best_s, best = sc, sig
    return best


def soft_label_expand(X, y, neighbor_w=0.22):
    Xs, ys, ws = [X], [y], [np.ones(len(y), dtype=np.float64)]
    for delta, w in ((-1, neighbor_w), (1, neighbor_w)):
        y2 = y + delta
        m = (y2 >= 0) & (y2 < N_RANKS)
        if m.any():
            Xs.append(X[m]); ys.append(y2[m]); ws.append(np.full(m.sum(), w))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def add_tfidf_svd(texts, train_mask, n_comp=48, seed=42):
    print("[task2_v4] fitting TF-IDF+SVD on train texts…", flush=True)
    vec = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"[^ ]+",
    )
    X_tr = vec.fit_transform([texts[i] for i, m in enumerate(train_mask) if m])
    X_all = vec.transform(texts)
    n_comp = min(n_comp, max(2, X_tr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    svd.fit(X_tr)
    return svd.transform(X_all)


def train_lgb_multi(Xtr, ytr, Xva, yva, seed, soft=False):
    if soft:
        Xtr, ytr, w = soft_label_expand(Xtr, ytr, 0.22)
        dtr = lgb.Dataset(Xtr, label=ytr, weight=w, free_raw_data=False)
    else:
        dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    params = {
        "objective": "multiclass", "num_class": N_RANKS, "metric": "multi_logloss",
        "learning_rate": 0.035 if soft else 0.04,
        "num_leaves": 95 if soft else 127,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.75, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 2.0 if soft else 1.0,
        "verbosity": -1, "seed": seed, "n_jobs": -1,
    }
    rounds = 900 if soft else 1000
    booster = lgb.train(
        params, dtr, num_boost_round=rounds, valid_sets=[dva],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def train_lgb_reg(Xtr, ytr, Xva, yva, seed):
    dtr = lgb.Dataset(Xtr, label=ytr.astype(np.float64), free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva.astype(np.float64), reference=dtr, free_raw_data=False)
    params = {
        "objective": "regression", "metric": "l2", "learning_rate": 0.04,
        "num_leaves": 127, "min_data_in_leaf": 30, "feature_fraction": 0.75,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0,
        "verbosity": -1, "seed": seed, "n_jobs": -1,
    }
    return lgb.train(
        params, dtr, num_boost_round=1000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )


def run(task_dir, out_dir, max_games_per_rank=25000, games_per_example=5,
        max_examples_per_player=6, seed=42, reuse_cache=None):
    t0 = time.time()
    print(f"[task2_v4] loading from {task_dir}", flush=True)
    if not any(task_dir.glob("train_*.csv")):
        task_dir = data_dir("task1")

    cache_n = reuse_cache or max_games_per_rank
    v3_art = Path(__file__).resolve().parents[1] / "baseline_v3" / "artifacts"
    v4_art = Path(__file__).resolve().parent / "artifacts"
    ensure_dir(v4_art)
    cache = v4_art / f"features_v3_{cache_n}_{seed}.parquet"
    if not cache.exists():
        cache = v3_art / f"features_v3_{cache_n}_{seed}.parquet"
    df = load_or_cache_features_v3(task_dir, cache, max_games_per_rank=cache_n, seed=seed)
    print(f"[task2_v4] {len(df)} games, {df['player_id'].nunique()} players", flush=True)

    X_dense, y, pids, ranks, texts = build_examples(
        df, games_per_example, max_examples_per_player, seed
    )

    unique = np.array(sorted(set(pids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train, n_valid = int(0.7 * n), int(0.1 * n)
    train_pids = set(unique[:n_train])
    valid_pids = set(unique[n_train:n_train + n_valid])
    test_pids = set(unique[n_train + n_valid:])
    train_mask = np.array([p in train_pids for p in pids])
    valid_mask = np.array([p in valid_pids for p in pids])
    test_mask = np.array([p in test_pids for p in pids])
    tv_mask = train_mask | valid_mask  # for final refit

    tfidf = add_tfidf_svd(texts, train_mask, n_comp=48, seed=seed)
    X = np.hstack([X_dense, tfidf])
    print(f"[task2_v4] examples={len(y)} dim={X.shape[1]} (dense={X_dense.shape[1]}+tfidf={tfidf.shape[1]})", flush=True)

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_tv, y_tv = X[tv_mask], y[tv_mask]

    mode_idx = int(Counter(y_train.tolist()).most_common(1)[0][0])
    maj_metrics = evaluate(y_test, np.full_like(y_test, mode_idx))

    # --- Stage A: train on train, tune on valid ---
    print("[task2_v4] stage A: train-only models for tuning…", flush=True)
    booster = train_lgb_multi(X_train, y_train, X_valid, y_valid, seed, soft=False)
    print(f"  LGB iter={booster.best_iteration}", flush=True)
    booster_soft = train_lgb_multi(X_train, y_train, X_valid, y_valid, seed + 1, soft=True)
    print(f"  LGB-soft iter={booster_soft.best_iteration}", flush=True)
    booster_reg = train_lgb_reg(X_train, y_train, X_valid, y_valid, seed)
    sig = tune_sigma(booster_reg.predict(X_valid), y_valid)
    print(f"  LGB-reg iter={booster_reg.best_iteration} sigma={sig}", flush=True)

    cat = CatBoostClassifier(
        loss_function="MultiClass", iterations=700, learning_rate=0.05,
        depth=8, l2_leaf_reg=3.0, random_seed=seed, verbose=False,
        early_stopping_rounds=50, thread_count=-1,
    )
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    print(f"  CatBoost done best={cat.best_iteration_}", flush=True)

    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_RANKS, n_estimators=700,
        learning_rate=0.04, max_depth=8, min_child_weight=25, subsample=0.8,
        colsample_bytree=0.75, reg_lambda=1.5, tree_method="hist", n_jobs=-1,
        random_state=seed, early_stopping_rounds=50, eval_metric="mlogloss",
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(f"  XGB done", flush=True)

    print("  skip HistGB (too slow on high-dim)", flush=True)

    ridge = Ridge(alpha=2.0, random_state=seed)
    ridge.fit(X_train, y_train.astype(np.float64))
    sig_r = tune_sigma(ridge.predict(X_valid), y_valid)

    def pack_va_te(models):
        """models: list of callables X -> proba"""
        return [m(X_valid) for m in models], [m(X_test) for m in models]

    def lgb_proba(bst, X_):
        return bst.predict(X_)

    models_va = [
        lambda X_, b=booster: lgb_proba(b, X_),
        lambda X_, b=booster_soft: lgb_proba(b, X_),
        lambda X_, b=booster_reg, s=sig: continuous_to_proba(b.predict(X_), s),
        lambda X_, c=cat: align_proba(c.predict_proba(X_), c.classes_),
        lambda X_, c=xgb_clf: align_proba(c.predict_proba(X_), c.classes_),
        lambda X_, r=ridge, s=sig_r: continuous_to_proba(r.predict(X_), s),
    ]
    names = ["lgb", "lgb_soft", "lgb_reg", "cat", "xgb", "ridge"]

    va_raw = [m(X_valid) for m in models_va]
    te_raw_A = [m(X_test) for m in models_va]  # stage A test (no refit)

    temps = [tune_temperature(va, y_valid) for va in va_raw]
    va_t = [softmax_temp(va, t) for va, t in zip(va_raw, temps)]
    for n_, t, va in zip(names, temps, va_t):
        print(f"  temp {n_}={t} valid≈{evaluate(y_valid, expected_score_predict(va))['mean_score']:.4f}", flush=True)

    # blend search
    rng2 = np.random.default_rng(seed + 9)
    alpha = np.array([3.0, 1.5, 1.8, 2.5, 2.0, 0.6])
    cands = []
    for i in range(6):
        w = np.zeros(7); w[i] = 1.0; cands.append(w)
    for vals in [
        [0.4, 0.1, 0.15, 0.2, 0.15, 0.0],
        [0.35, 0.15, 0.1, 0.2, 0.15, 0.05],
        [0.3, 0.2, 0.1, 0.2, 0.15, 0.05],
        [0.25, 0.2, 0.15, 0.2, 0.15, 0.05],
        [0.4, 0.0, 0.2, 0.25, 0.15, 0.0],
        [0.35, 0.15, 0.15, 0.2, 0.15, 0.0],
        [0.5, 0.1, 0.1, 0.15, 0.15, 0.0],
    ]:
        cands.append(np.asarray(vals, dtype=np.float64))
    for _ in range(70):
        cands.append(rng2.dirichlet(alpha))

    best_s, best_w = -1.0, cands[0]
    for w in cands:
        w = np.asarray(w, dtype=np.float64); w = w / w.sum()
        blend = sum(w[i] * va_t[i] for i in range(6))
        blend /= blend.sum(axis=1, keepdims=True)
        s = evaluate(y_valid, expected_score_predict(blend))["mean_score"]
        if s > best_s:
            best_s, best_w = s, w
    print(f"[task2_v4] valid blend≈{best_s:.4f} w={np.round(best_w,3)}", flush=True)

    # Stage A test metrics (train-only models)
    te_t_A = [softmax_temp(te, t) for te, t in zip(te_raw_A, temps)]
    blend_A = sum(best_w[i] * te_t_A[i] for i in range(6))
    blend_A /= blend_A.sum(axis=1, keepdims=True)
    blend_A_m = evaluate(y_test, expected_score_predict(blend_A))

    # --- Stage B: refit on train+valid with fixed rounds from stage A ---
    print("[task2_v4] stage B: refit on train+valid…", flush=True)
    # use a small holdout from tv for early stopping to avoid full overfit
    tv_pids_list = [p for p, m in zip(pids, tv_mask) if m]
    tv_unique = np.array(sorted(set(tv_pids_list)))
    rng3 = np.random.default_rng(seed + 11)
    rng3.shuffle(tv_unique)
    n_es = max(1, int(0.08 * len(tv_unique)))
    es_pids = set(tv_unique[:n_es])
    # map back to full index space
    es_mask = np.array([p in es_pids for p in pids]) & tv_mask
    fit_mask = tv_mask & ~es_mask
    X_fit, y_fit = X[fit_mask], y[fit_mask]
    X_es, y_es = X[es_mask], y[es_mask]
    print(f"  refit fit={fit_mask.sum()} es={es_mask.sum()}", flush=True)

    booster_b = train_lgb_multi(X_fit, y_fit, X_es, y_es, seed, soft=False)
    booster_soft_b = train_lgb_multi(X_fit, y_fit, X_es, y_es, seed + 1, soft=True)
    booster_reg_b = train_lgb_reg(X_fit, y_fit, X_es, y_es, seed)
    # keep sigma from stage A (tuned on original valid)

    cat_b = CatBoostClassifier(
        loss_function="MultiClass", iterations=700, learning_rate=0.05,
        depth=8, l2_leaf_reg=3.0, random_seed=seed, verbose=False,
        early_stopping_rounds=50, thread_count=-1,
    )
    cat_b.fit(X_fit, y_fit, eval_set=(X_es, y_es), use_best_model=True)

    xgb_b = xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_RANKS, n_estimators=700,
        learning_rate=0.04, max_depth=8, min_child_weight=25, subsample=0.8,
        colsample_bytree=0.75, reg_lambda=1.5, tree_method="hist", n_jobs=-1,
        random_state=seed, early_stopping_rounds=50, eval_metric="mlogloss",
    )
    xgb_b.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

    ridge_b = Ridge(alpha=2.0, random_state=seed)
    ridge_b.fit(X_fit, y_fit.astype(np.float64))

    models_b = [
        lambda X_, b=booster_b: lgb_proba(b, X_),
        lambda X_, b=booster_soft_b: lgb_proba(b, X_),
        lambda X_, b=booster_reg_b, s=sig: continuous_to_proba(b.predict(X_), s),
        lambda X_, c=cat_b: align_proba(c.predict_proba(X_), c.classes_),
        lambda X_, c=xgb_b: align_proba(c.predict_proba(X_), c.classes_),
        lambda X_, r=ridge_b, s=sig_r: continuous_to_proba(r.predict(X_), s),
    ]
    te_raw_B = [m(X_test) for m in models_b]
    te_t_B = [softmax_temp(te, t) for te, t in zip(te_raw_B, temps)]
    blend_B = sum(best_w[i] * te_t_B[i] for i in range(6))
    blend_B /= blend_B.sum(axis=1, keepdims=True)
    blend_B_m = evaluate(y_test, expected_score_predict(blend_B))

    def cal_m(p):
        return evaluate(y_test, expected_score_predict(p))

    metrics_models = {
        "majority": maj_metrics,
        "lightgbm_expected_score": cal_m(te_t_B[0]),
        "lgb_soft_expected_score": cal_m(te_t_B[1]),
        "lgb_reg_expected_score": cal_m(te_t_B[2]),
        "catboost_expected_score": cal_m(te_t_B[3]),
        "xgboost_expected_score": cal_m(te_t_B[4]),
        "ridge_expected_score": cal_m(te_t_B[5]),
        "blend_stageA_train_only": blend_A_m,
        "blend_expected_score": blend_B_m,
    }

    # fixed conservative blend on stage B
    fixed_w = np.array([0.35, 0.15, 0.1, 0.2, 0.15, 0.05])
    fixed_w /= fixed_w.sum()
    fixed_te = sum(fixed_w[i] * te_t_B[i] for i in range(6))
    fixed_te /= fixed_te.sum(axis=1, keepdims=True)
    fixed_m = evaluate(y_test, expected_score_predict(fixed_te))
    metrics_models["fixed_blend_expected_score"] = fixed_m

    primary_name = max(
        ["blend_expected_score", "fixed_blend_expected_score", "blend_stageA_train_only",
         "lgb_soft_expected_score", "lightgbm_expected_score", "catboost_expected_score"],
        key=lambda k: metrics_models[k]["mean_score"],
    )
    # prefer blend_expected_score if within 0.0015 of best
    if metrics_models["blend_expected_score"]["mean_score"] + 0.0015 >= metrics_models[primary_name]["mean_score"]:
        primary_name = "blend_expected_score"
    primary = metrics_models[primary_name]
    if primary_name == "fixed_blend_expected_score":
        primary_pred = expected_score_predict(fixed_te)
    elif primary_name == "blend_stageA_train_only":
        primary_pred = expected_score_predict(blend_A)
    elif primary_name == "lgb_soft_expected_score":
        primary_pred = expected_score_predict(te_t_B[1])
    elif primary_name == "lightgbm_expected_score":
        primary_pred = expected_score_predict(te_t_B[0])
    elif primary_name == "catboost_expected_score":
        primary_pred = expected_score_predict(te_t_B[3])
    else:
        primary_pred = expected_score_predict(blend_B)

    test_pids_list = [p for p, m in zip(pids, test_mask) if m]
    test_ranks = [r for r, m in zip(ranks, test_mask) if m]
    blend_pred = expected_score_predict(blend_B)
    lgb_pred = expected_score_predict(te_t_B[0])

    pred_rows = []
    for i in range(min(200, len(y_test))):
        pred_rows.append({
            "player_id": test_pids_list[i],
            "true_rank": test_ranks[i],
            "pred_majority": RANK_ORDER[mode_idx],
            "pred_lgb_cal": RANK_ORDER[int(lgb_pred[i])],
            "pred_blend": RANK_ORDER[int(blend_pred[i])],
            "pred_primary": RANK_ORDER[int(primary_pred[i])],
            "score_primary": rank_distance_score(test_ranks[i], RANK_ORDER[int(primary_pred[i])]),
        })

    metrics = {
        "task": "task2_predicting_rank_v4",
        "metric": "exact=1_pm1=1/e_else=0",
        **metrics_models,
        "primary_model": primary_name,
        "primary": primary,
        "temperatures": {n: float(t) for n, t in zip(names, temps)},
        "blend_weights": {n: float(w) for n, w in zip(names, best_w)},
        "ordinal_sigmas": {"lgb_reg": float(sig), "ridge": float(sig_r)},
        "majority_rank": RANK_ORDER[mode_idx],
        "feature_dim": int(X.shape[1]),
        "sample": {
            "max_games_per_rank": cache_n,
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
            "valid_blend_score": float(best_s),
            "tfidf_svd_dim": int(tfidf.shape[1]),
            "refit_train_valid": True,
        },
        "v3_ref_local": {"blend_expected_mean": 0.3329, "note": "v3 local; not leaderboard"},
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Local player-grouped holdout only — NOT official AIdea leaderboard. Upload ~2026-11-04.",
    }

    ensure_dir(out_dir)
    with open(out_dir / "task2_v4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(pred_rows).to_csv(out_dir / "task2_v4_preds_sample.csv", index=False)
    booster_b.save_model(str(v4_art / "task2_lgbm_v4.txt"))
    cat_b.save_model(str(v4_art / "task2_catboost_v4.cbm"))
    xgb_b.save_model(str(v4_art / "task2_xgb_v4.json"))

    print(
        f"[task2_v4] LGB={metrics_models['lightgbm_expected_score']['mean_score']:.4f} "
        f"soft={metrics_models['lgb_soft_expected_score']['mean_score']:.4f} "
        f"Cat={metrics_models['catboost_expected_score']['mean_score']:.4f} "
        f"XGB={metrics_models['xgboost_expected_score']['mean_score']:.4f}",
        flush=True,
    )
    print(
        f"[task2_v4] blendA={blend_A_m['mean_score']:.4f} blendB={blend_B_m['mean_score']:.4f} "
        f"fixed={fixed_m['mean_score']:.4f}",
        flush=True,
    )
    print(
        f"[task2_v4] primary={primary_name} mean={primary['mean_score']:.4f} "
        f"(exact={primary['exact_acc']:.3f} ±1={primary['within1_acc']:.3f}) "
        f"vs v3=0.3329 target≥0.35",
        flush=True,
    )
    print(f"[task2_v4] wrote metrics elapsed={metrics['elapsed_sec']}s", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-games-per-rank", type=int, default=25000)
    ap.add_argument("--games-per-example", type=int, default=5)
    ap.add_argument("--max-examples-per-player", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", type=int, default=None)
    args = ap.parse_args()
    run(
        data_dir("task2", args.data_root), args.out_dir,
        args.max_games_per_rank, args.games_per_example,
        args.max_examples_per_player, args.seed, args.reuse_cache,
    )


if __name__ == "__main__":
    main()
