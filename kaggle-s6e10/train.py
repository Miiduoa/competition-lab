#!/usr/bin/env python3
"""Playground S6E10 — Predicting Airline Satisfaction (ROC-AUC, probabilities)."""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, Pool

    HAS_CB = True
except Exception:
    HAS_CB = False

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
ORIG_DIR = DATA_DIR / "original"
TARGET = "satisfaction"
ID_COL = "id"
NTHREAD = 4

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
DIGITAL = ["Inflight wifi service", "Ease of Online booking", "Online boarding"]
INFLIGHT = [
    "Food and drink",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Cleanliness",
]
GROUND = [
    "Departure/Arrival time convenient",
    "Gate location",
    "Baggage handling",
    "Checkin service",
]
COMBO_CATS = ["travel_class", "customer_class", "travel_customer", "trio"]
TE_PRIOR = 80.0

PARAMS_LGBM = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "min_gain_to_split": 0.0,
    "max_bin": 511,
    "cat_smooth": 10.0,
    "verbosity": -1,
    "num_threads": NTHREAD,
    "force_col_wise": True,
}
PARAMS_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.02,
    "max_depth": 7,
    "min_child_weight": 16,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "max_bin": 256,
    "tree_method": "hist",
    "nthread": NTHREAD,
    "enable_categorical": True,
}
PARAMS_CB = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 4.0,
    "border_count": 254,
    "random_strength": 0.3,
    "bagging_temperature": 0.2,
    "iterations": 6000,
    "od_type": "Iter",
    "od_wait": 200,
    "thread_count": NTHREAD,
    "verbose": False,
    "allow_writing_files": False,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _trailing_zeros(a: np.ndarray) -> np.ndarray:
    z = np.zeros(len(a), dtype=np.int8)
    for k, m in enumerate([10, 100, 1000, 10000], start=1):
        z[a % m == 0] = k
    return z


def encode_target(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s.astype(np.int8).to_numpy()
    v = s.astype(str).str.strip().str.lower()
    return v.isin(["true", "1", "yes", "satisfied"]).astype(np.int8).to_numpy()


def add_combo_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Customer Type"].astype(str)
    t = d["Type of Travel"].astype(str)
    k = d["Class"].astype(str)
    d["travel_class"] = t + "|" + k
    d["customer_class"] = c + "|" + k
    d["travel_customer"] = t + "|" + c
    d["trio"] = t + "|" + k + "|" + c
    d["age_class"] = d["Age"].astype(str) + "|" + k
    d["dist_class"] = d["Flight Distance"].astype(str) + "|" + k
    d["wifi_class"] = d["Inflight wifi service"].astype(str) + "|" + k
    d["board_class"] = d["Online boarding"].astype(str) + "|" + k
    return d


def generate_synthetic(n_train: int = 4000, n_test: int = 1200, seed: int = 42):
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _make(n: int, start_id: int, with_y: bool) -> pd.DataFrame:
        age = rng.integers(7, 85, n)
        dist = rng.integers(67, 4000, n)
        ratings = {c: rng.integers(0, 6, n) for c in RATINGS}
        ratings["Baggage handling"] = rng.integers(1, 6, n)
        dep = np.clip(rng.integers(0, 40, n) * (rng.random(n) < 0.15), 0, 200)
        arr = np.clip(dep + rng.integers(-5, 12, n), 0, 220).astype(float)
        arr[rng.random(n) < 0.01] = np.nan
        gender = rng.choice(["Male", "Female"], n)
        cust = rng.choice(["Loyal Customer", "disloyal Customer"], n, p=[0.82, 0.18])
        travel = rng.choice(["Business travel", "Personal Travel"], n, p=[0.7, 0.3])
        klass = rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.48, 0.47, 0.05])
        board = ratings["Online boarding"]
        wifi = ratings["Inflight wifi service"]
        score = (
            (travel == "Business travel") * 1.4
            + (klass == "Business") * 1.2
            + (cust == "Loyal Customer") * 0.5
            + board * 0.45
            + wifi * 0.15
            + ratings["Seat comfort"] * 0.12
            + (age > 40) * 0.2
            + (dist > 1500) * 0.25
            + rng.normal(0, 0.9, n)
        )
        y = score > np.median(score)
        df = pd.DataFrame(
            {
                ID_COL: np.arange(start_id, start_id + n),
                "Age": age,
                "Flight Distance": dist,
                **ratings,
                "Departure Delay in Minutes": dep.astype(int),
                "Arrival Delay in Minutes": arr,
                "Gender": gender,
                "Customer Type": cust,
                "Type of Travel": travel,
                "Class": klass,
            }
        )
        if with_y:
            df[TARGET] = y
        return df

    tr = _make(n_train, 0, True)
    te = _make(n_test, n_train, False)
    mean_y = float(tr[TARGET].mean())
    ss = pd.DataFrame({ID_COL: te[ID_COL], TARGET: mean_y})
    tr.to_csv(DATA_DIR / "train.csv", index=False)
    te.to_csv(DATA_DIR / "test.csv", index=False)
    ss.to_csv(DATA_DIR / "sample_submission.csv", index=False)
    log(f"[smoke] wrote synthetic train={tr.shape} test={te.shape}")


def load_original() -> pd.DataFrame | None:
    paths = [ORIG_DIR / "train.csv", ORIG_DIR / "test.csv"]
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        return None
    og = pd.concat(frames, ignore_index=True)
    drop = [c for c in og.columns if c.lower().startswith("unnamed") or c == "id"]
    og = og.drop(columns=drop, errors="ignore")
    if "Check-in service" in og.columns and "Checkin service" not in og.columns:
        og = og.rename(columns={"Check-in service": "Checkin service"})
    if TARGET not in og.columns:
        return None
    og = og.copy()
    og[TARGET] = encode_target(og[TARGET])
    return og


class GlobalStats:
    """Label-free statistics from train+test and (optional) original data."""

    def __init__(self):
        self.vc: dict[str, pd.Series] = {}
        self.route: dict[str, pd.Series] = {}
        self.orig_mean: dict[str, pd.Series] = {}
        self.orig_count: dict[str, pd.Series] = {}
        self.orig_gm: float = 0.5

    def fit(self, tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame | None):
        all_df = add_combo_cols(pd.concat([tr, te], ignore_index=True))
        keys = [
            "Flight Distance",
            "Age",
            "travel_class",
            "customer_class",
            "travel_customer",
            "trio",
            "age_class",
            "dist_class",
        ]
        for k in keys:
            self.vc[k] = all_df[k].value_counts()
        dist_mod = (all_df["Flight Distance"] % 100).astype(np.int16)
        self.vc["dist_mod100"] = dist_mod.value_counts()
        route_src = all_df.copy()
        route_src["Arrival Delay in Minutes"] = route_src["Arrival Delay in Minutes"].fillna(
            route_src["Departure Delay in Minutes"]
        )
        g = route_src.groupby("Flight Distance")
        for c in RATINGS + ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
            self.route[c] = g[c].mean()

        if og is None:
            return
        ogc = add_combo_cols(og)
        self.orig_gm = float(ogc[TARGET].mean())
        og_keys = CATS + RATINGS + [
            "Age",
            "Flight Distance",
            "travel_class",
            "customer_class",
            "travel_customer",
            "trio",
            "age_class",
        ]
        for k in og_keys:
            if k not in ogc.columns:
                continue
            grp = ogc.groupby(k)[TARGET]
            self.orig_mean[k] = grp.mean()
            self.orig_count[k] = grp.size()


def map_num(series, lookup: pd.Series, default: float) -> np.ndarray:
    return pd.Series(series, copy=False).map(lookup).fillna(default).to_numpy(np.float32)


def base_frame(df: pd.DataFrame, stats: GlobalStats) -> pd.DataFrame:
    d = add_combo_cols(df.reset_index(drop=True))
    n = len(d)
    X = pd.DataFrame(index=range(n))
    X["Age"] = d["Age"].astype(np.float32)
    X["Flight Distance"] = d["Flight Distance"].astype(np.float32)
    dep = d["Departure Delay in Minutes"].astype(np.float32).to_numpy()
    arr_raw = d["Arrival Delay in Minutes"].astype(np.float32).to_numpy()
    arr_na = np.isnan(arr_raw).astype(np.int8)
    arr = np.where(np.isnan(arr_raw), dep, arr_raw).astype(np.float32)
    X["Departure Delay in Minutes"] = dep
    X["Arrival Delay in Minutes"] = arr
    X["arr_missing"] = arr_na
    X["delay_diff"] = (arr - dep).astype(np.float32)
    X["delay_sum"] = (arr + dep).astype(np.float32)
    X["log_dep"] = np.log1p(np.clip(dep, 0, None)).astype(np.float32)
    X["log_arr"] = np.log1p(np.clip(arr, 0, None)).astype(np.float32)
    X["log_dist"] = np.log1p(X["Flight Distance"]).astype(np.float32)
    X["delayed"] = ((dep > 0) | (arr > 0)).astype(np.int8)
    X["long_delay"] = ((dep >= 15) | (arr >= 15)).astype(np.int8)

    R = d[RATINGS].astype(np.float32)
    for c in RATINGS:
        X[c] = R[c]
        X[c + "_is0"] = (R[c] == 0).astype(np.int8)
        X[c + "_is5"] = (R[c] == 5).astype(np.int8)
    X["rating_mean"] = R.mean(axis=1).astype(np.float32)
    X["rating_std"] = R.std(axis=1).astype(np.float32)
    X["rating_min"] = R.min(axis=1).astype(np.float32)
    X["rating_max"] = R.max(axis=1).astype(np.float32)
    X["rating_sum"] = R.sum(axis=1).astype(np.float32)
    X["n_zero"] = (R == 0).sum(axis=1).astype(np.int8)
    X["n_one"] = (R == 1).sum(axis=1).astype(np.int8)
    X["n_five"] = (R == 5).sum(axis=1).astype(np.int8)
    X["n_ge4"] = (R >= 4).sum(axis=1).astype(np.int8)
    X["n_le2"] = ((R > 0) & (R <= 2)).sum(axis=1).astype(np.int8)
    Rnz = R.mask(R == 0)
    X["rating_mean_nz"] = Rnz.mean(axis=1).fillna(X["rating_mean"]).astype(np.float32)
    X["digital_mean"] = d[DIGITAL].astype(np.float32).mean(axis=1)
    X["inflight_mean"] = d[INFLIGHT].astype(np.float32).mean(axis=1)
    X["ground_mean"] = d[GROUND].astype(np.float32).mean(axis=1)
    X["wifi_minus_book"] = (
        d["Inflight wifi service"].astype(np.float32) - d["Ease of Online booking"].astype(np.float32)
    )
    X["board_minus_wifi"] = (
        d["Online boarding"].astype(np.float32) - d["Inflight wifi service"].astype(np.float32)
    )
    X["seat_minus_ent"] = (
        d["Seat comfort"].astype(np.float32) - d["Inflight entertainment"].astype(np.float32)
    )

    dist = d["Flight Distance"].to_numpy(np.int64)
    age = d["Age"].to_numpy(np.int64)
    X["dist_mod10"] = (dist % 10).astype(np.int8)
    X["dist_mod100"] = (dist % 100).astype(np.int16)
    X["dist_mod1000"] = (dist % 1000).astype(np.int16)
    X["dist_div10"] = (dist // 10).astype(np.int32)
    X["dist_div100"] = (dist // 100).astype(np.int16)
    X["dist_tz"] = _trailing_zeros(dist)
    X["dist_round10"] = (dist % 10 == 0).astype(np.int8)
    X["dist_round100"] = (dist % 100 == 0).astype(np.int8)
    X["dist_round1000"] = (dist % 1000 == 0).astype(np.int8)
    X["age_mod10"] = (age % 10).astype(np.int8)
    X["age_tz"] = _trailing_zeros(age)
    X["age_child"] = (age < 18).astype(np.int8)
    X["age_senior"] = (age >= 60).astype(np.int8)

    for c in CATS + COMBO_CATS:
        X[c] = d[c].astype("category")

    for k, lookup in stats.vc.items():
        src = (dist % 100).astype(np.int16) if k == "dist_mod100" else d[k]
        X[f"vc_{k}"] = map_num(src, lookup, 1.0)
        X[f"logvc_{k}"] = np.log1p(X[f"vc_{k}"]).astype(np.float32)

    for c, lookup in stats.route.items():
        X[f"rt_{c}"] = map_num(d["Flight Distance"], lookup, float("nan"))
        X[f"rt_{c}"] = X[f"rt_{c}"].fillna(X[f"rt_{c}"].median()).astype(np.float32)
        if c in d.columns:
            rawc = d[c].astype(np.float32).to_numpy()
            if c == "Arrival Delay in Minutes":
                rawc = arr
            X[f"rt_{c}_delta"] = (rawc - X[f"rt_{c}"].to_numpy()).astype(np.float32)

    for k, lookup in stats.orig_mean.items():
        if k not in d.columns:
            continue
        X[f"ogm_{k}"] = map_num(d[k], lookup, stats.orig_gm)
    for k, lookup in stats.orig_count.items():
        if k not in d.columns:
            continue
        X[f"ogc_{k}"] = map_num(d[k], lookup, 0.0)
    return X


def te_apply(keys_tr, y_tr, keys_app, prior=TE_PRIOR) -> np.ndarray:
    gm = float(np.mean(y_tr))
    s = pd.DataFrame({"k": keys_tr, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    sm = (s["sum"] + prior * gm) / (s["count"] + prior)
    return pd.Series(keys_app).map(sm).fillna(gm).to_numpy(np.float32)


def te_keys(df: pd.DataFrame) -> dict[str, np.ndarray]:
    d = add_combo_cols(df.reset_index(drop=True))
    dist = d["Flight Distance"].to_numpy(np.int64)
    return {
        "te_dist": dist,
        "te_dist10": dist // 10,
        "te_dist100": dist // 100,
        "te_dist_mod100": dist % 100,
        "te_dist_mod1000": dist % 1000,
        "te_age": d["Age"].to_numpy(np.int64),
        "te_age_class": d["age_class"].to_numpy(),
        "te_dist_class": d["dist_class"].to_numpy(),
        "te_travel_class": d["travel_class"].to_numpy(),
        "te_customer_class": d["customer_class"].to_numpy(),
        "te_trio": d["trio"].to_numpy(),
        "te_wifi": d["Inflight wifi service"].to_numpy(np.int64),
        "te_board": d["Online boarding"].to_numpy(np.int64),
        "te_wifi_class": d["wifi_class"].to_numpy(),
        "te_board_class": d["board_class"].to_numpy(),
    }


def add_te(X: pd.DataFrame, src_df: pd.DataFrame, tr_df: pd.DataFrame, y_tr: np.ndarray) -> pd.DataFrame:
    out = X.copy()
    src_k = te_keys(src_df)
    tr_k = te_keys(tr_df)
    for name, k_app in src_k.items():
        out[name] = te_apply(tr_k[name], y_tr, k_app)
    return out


def cat_names(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if str(X[c].dtype) == "category"]


def to_cb(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    Z = X.copy()
    cats = []
    for c in Z.columns:
        if str(Z[c].dtype) == "category":
            Z[c] = Z[c].astype(str)
            cats.append(c)
    return Z, cats


def to_xgb(X: pd.DataFrame) -> pd.DataFrame:
    return X.copy()


def align_cats(train_x: pd.DataFrame, *others: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    for o in others:
        b = o.copy()
        for c in train_x.columns:
            if str(train_x[c].dtype) == "category":
                b[c] = pd.Categorical(b[c].astype(str), categories=train_x[c].cat.categories)
        out.append(b[train_x.columns])
    return out


def train_original_teacher(og: pd.DataFrame, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stats = GlobalStats()
    stats.fit(og.drop(columns=[TARGET]), og.drop(columns=[TARGET]).iloc[:0], None)
    Xog = base_frame(og.drop(columns=[TARGET]), stats)
    yog = og[TARGET].to_numpy(np.int8)
    cats = cat_names(Xog)
    dtrain = lgb.Dataset(Xog, yog, categorical_feature=cats, free_raw_data=False)
    params = {
        **PARAMS_LGBM,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "seed": 7,
    }
    m = lgb.train(params, dtrain, num_boost_round=800)
    ptr = m.predict(base_frame(tr, stats))
    pte = m.predict(base_frame(te, stats))
    log(f"[teacher] original n={len(og)} orig_auc_self={roc_auc_score(yog, m.predict(Xog)):.4f}")
    del m, Xog, dtrain
    gc.collect()
    return ptr.astype(np.float32), pte.astype(np.float32)


def fold_matrices(Xtr, Xte, raw_tr, raw_te, y, tr_idx, va_idx):
    A0 = Xtr.iloc[tr_idx]
    B0 = Xtr.iloc[va_idx]
    T0 = Xte
    y_tr = y[tr_idx]
    A = add_te(A0, raw_tr.iloc[tr_idx], raw_tr.iloc[tr_idx], y_tr)
    B = add_te(B0, raw_tr.iloc[va_idx], raw_tr.iloc[tr_idx], y_tr)
    T = add_te(T0, raw_te, raw_tr.iloc[tr_idx], y_tr)
    B, T = align_cats(A, B, T)
    return A, B, T


def train_lgbm(Xtr, Xte, raw_tr, raw_te, y, folds, t0, rounds=8000, es=200, seed=42):
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    rows = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        A, B, T = fold_matrices(Xtr, Xte, raw_tr, raw_te, y, tr_idx, va_idx)
        cats = cat_names(A)
        d1 = lgb.Dataset(A, y[tr_idx], categorical_feature=cats, free_raw_data=False)
        d2 = lgb.Dataset(B, y[va_idx], categorical_feature=cats, reference=d1, free_raw_data=False)
        m = lgb.train(
            {**PARAMS_LGBM, "seed": seed + k},
            d1,
            rounds,
            valid_sets=[d2],
            callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
        )
        it = m.best_iteration or rounds
        pv = m.predict(B, num_iteration=it)
        pt = m.predict(T, num_iteration=it)
        oof[va_idx] = pv
        test_pred += pt / len(folds)
        auc = roc_auc_score(y[va_idx], pv)
        rows.append({"model": "lgbm", "fold": k, "auc": float(auc), "iter": int(it)})
        log(f"  [LGBM] fold {k}: AUC={auc:.5f} it={it}  [{(time.time()-t0)/60:.1f} min]")
        del A, B, T, d1, d2, m
        gc.collect()
    log(f"  [LGBM] OOF={roc_auc_score(y, oof):.5f}")
    return oof, test_pred, rows


def train_xgb(Xtr, Xte, raw_tr, raw_te, y, folds, t0, rounds=8000, es=200, seed=42):
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    rows = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        A, B, T = fold_matrices(Xtr, Xte, raw_tr, raw_te, y, tr_idx, va_idx)
        dtrain = xgb.DMatrix(to_xgb(A), label=y[tr_idx], enable_categorical=True)
        dvalid = xgb.DMatrix(to_xgb(B), label=y[va_idx], enable_categorical=True)
        dtest = xgb.DMatrix(to_xgb(T), enable_categorical=True)
        m = xgb.train(
            {**PARAMS_XGB, "seed": seed + k},
            dtrain,
            num_boost_round=rounds,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=es,
            verbose_eval=False,
        )
        it = int(m.best_iteration) + 1
        pv = m.predict(dvalid, iteration_range=(0, it))
        pt = m.predict(dtest, iteration_range=(0, it))
        oof[va_idx] = pv
        test_pred += pt / len(folds)
        auc = roc_auc_score(y[va_idx], pv)
        rows.append({"model": "xgboost", "fold": k, "auc": float(auc), "iter": int(it)})
        log(f"  [XGB] fold {k}: AUC={auc:.5f} it={it}  [{(time.time()-t0)/60:.1f} min]")
        del A, B, T, dtrain, dvalid, dtest, m
        gc.collect()
    log(f"  [XGB] OOF={roc_auc_score(y, oof):.5f}")
    return oof, test_pred, rows


def train_cb(Xtr, Xte, raw_tr, raw_te, y, folds, t0, rounds=6000, seed=42):
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    rows = []
    params = dict(PARAMS_CB)
    params["iterations"] = rounds
    for k, (tr_idx, va_idx) in enumerate(folds):
        A, B, T = fold_matrices(Xtr, Xte, raw_tr, raw_te, y, tr_idx, va_idx)
        Ac, cats = to_cb(A)
        Bc, _ = to_cb(B)
        Tc, _ = to_cb(T)
        train_pool = Pool(Ac, y[tr_idx], cat_features=cats)
        valid_pool = Pool(Bc, y[va_idx], cat_features=cats)
        test_pool = Pool(Tc, cat_features=cats)
        m = CatBoostClassifier(**params, random_seed=seed + k)
        m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        it = m.get_best_iteration() or m.tree_count_
        pv = m.predict_proba(valid_pool)[:, 1]
        pt = m.predict_proba(test_pool)[:, 1]
        oof[va_idx] = pv
        test_pred += pt / len(folds)
        auc = roc_auc_score(y[va_idx], pv)
        rows.append({"model": "catboost", "fold": k, "auc": float(auc), "iter": int(it)})
        log(f"  [CB] fold {k}: AUC={auc:.5f} it={it}  [{(time.time()-t0)/60:.1f} min]")
        del A, B, T, Ac, Bc, Tc, train_pool, valid_pool, test_pool, m
        gc.collect()
    log(f"  [CB] OOF={roc_auc_score(y, oof):.5f}")
    return oof, test_pred, rows


def rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    from scipy.stats import rankdata

    ranks = [rankdata(p) / len(p) for p in preds]
    return np.mean(ranks, axis=0)


def hill_climb(oofs: dict[str, np.ndarray], y: np.ndarray, step=0.02):
    names = list(oofs.keys())
    mats = np.column_stack([oofs[n] for n in names])
    best_w = np.zeros(len(names))
    singles = [roc_auc_score(y, mats[:, i]) for i in range(len(names))]
    best_w[int(np.argmax(singles))] = 1.0
    best_auc = max(singles)
    improved = True
    while improved:
        improved = False
        for i in range(len(names)):
            for delta in (+step, -step):
                w = best_w.copy()
                w[i] += delta
                if w[i] < -1e-9 or w.sum() <= 0:
                    continue
                w = np.clip(w, 0, None)
                w = w / w.sum()
                auc = roc_auc_score(y, mats @ w)
                if auc > best_auc + 1e-7:
                    best_auc = auc
                    best_w = w
                    improved = True
    return {n: float(w) for n, w in zip(names, best_w)}, float(best_auc)


def write_sub(ids, pred, path: Path, sample: pd.DataFrame):
    pred = np.clip(np.asarray(pred, dtype=np.float64), 1e-7, 1 - 1e-7)
    sub = pd.DataFrame({ID_COL: ids, TARGET: pred})
    if list(sub.columns) != [ID_COL, TARGET]:
        raise ValueError(f"columns {list(sub.columns)}")
    if len(sub) != len(sample):
        raise ValueError(f"row count {len(sub)} != {len(sample)}")
    if not np.array_equal(sub[ID_COL].to_numpy(), sample[ID_COL].to_numpy()):
        raise ValueError("id order mismatch vs sample_submission")
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    log(f"[write] {path} rows={len(sub)} mean={pred.mean():.5f}")
    return sub


def main():
    global DATA_DIR, ORIG_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="lgbm,xgb,cat")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--es", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    sub_root = ROOT
    if args.smoke:
        DATA_DIR = ROOT / "_smoke_data"
        ORIG_DIR = DATA_DIR / "original"
        OUT_DIR = DATA_DIR / "outputs"
        sub_root = DATA_DIR
        generate_synthetic()
        args.rounds = min(args.rounds, 80)
        args.es = min(args.es, 20)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = encode_target(tr[TARGET])
    raw_tr = tr.drop(columns=[TARGET])
    raw_te = te.copy()
    log(f"[data] train={raw_tr.shape} test={raw_te.shape} pos={y.mean():.5f}")

    og = None if args.smoke else load_original()
    log(f"[data] original={'yes '+str(og.shape) if og is not None else 'no'}")

    stats = GlobalStats()
    stats.fit(raw_tr, raw_te, og)
    Xtr = base_frame(raw_tr, stats)
    Xte = base_frame(raw_te, stats)
    for c in cat_names(Xtr):
        levels = sorted(set(Xtr[c].astype(str)) | set(Xte[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=levels)
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=levels)
    if og is not None:
        ptr, pte = train_original_teacher(og, raw_tr, raw_te)
        Xtr["orig_teacher"] = ptr
        Xte["orig_teacher"] = pte
    log(f"[fe] n_features={Xtr.shape[1]} cats={cat_names(Xtr)}")

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y))

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    oofs: dict[str, np.ndarray] = {}
    tests: dict[str, np.ndarray] = {}
    fold_rows = []

    if "lgbm" in wanted:
        log("\n=== LightGBM ===")
        oof, tp, fr = train_lgbm(Xtr, Xte, raw_tr, raw_te, y, folds, t0, args.rounds, args.es, args.seed)
        oofs["lgbm"] = oof
        tests["lgbm"] = tp
        fold_rows.extend(fr)
        np.save(OUT_DIR / "oof_lgbm.npy", oof)

    if "xgb" in wanted:
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        log("\n=== XGBoost ===")
        oof, tp, fr = train_xgb(Xtr, Xte, raw_tr, raw_te, y, folds, t0, args.rounds, args.es, args.seed)
        oofs["xgboost"] = oof
        tests["xgboost"] = tp
        fold_rows.extend(fr)
        np.save(OUT_DIR / "oof_xgboost.npy", oof)

    if "cat" in wanted:
        if not HAS_CB:
            raise RuntimeError("catboost not installed")
        log("\n=== CatBoost ===")
        cb_rounds = min(args.rounds, PARAMS_CB["iterations"])
        if args.smoke:
            cb_rounds = args.rounds
        oof, tp, fr = train_cb(Xtr, Xte, raw_tr, raw_te, y, folds, t0, cb_rounds, args.seed)
        oofs["catboost"] = oof
        tests["catboost"] = tp
        fold_rows.extend(fr)
        np.save(OUT_DIR / "oof_catboost.npy", oof)

    singles = {k: float(roc_auc_score(y, v)) for k, v in oofs.items()}
    log("\n=== Single-model OOF ===")
    for k, v in singles.items():
        log(f"  {k}: {v:.5f}")

    eq = np.mean(list(oofs.values()), axis=0)
    eq_auc = float(roc_auc_score(y, eq))
    ra = rank_avg(list(oofs.values()))
    ra_auc = float(roc_auc_score(y, ra))
    weights, hc_auc = hill_climb(oofs, y, step=0.02)
    hc_oof = sum(weights[k] * oofs[k] for k in weights)
    hc_test = sum(weights[k] * tests[k] for k in weights)
    keep = [k for k, w in weights.items() if w >= 0.05]
    if len(keep) >= 2:
        ra2 = rank_avg([oofs[k] for k in keep])
        ra2_auc = float(roc_auc_score(y, ra2))
        ra2_test = rank_avg([tests[k] for k in keep])
    else:
        ra2, ra2_auc = ra, ra_auc
        ra2_test = rank_avg(list(tests.values()))

    candidates = {
        "equal_prob": (eq_auc, eq, np.mean(list(tests.values()), axis=0)),
        "rank_avg_all": (ra_auc, ra, rank_avg(list(tests.values()))),
        "hillclimb": (hc_auc, hc_oof, hc_test),
        "rank_avg_kept": (ra2_auc, ra2, ra2_test),
    }
    best_single = max(singles, key=singles.get)
    candidates[f"single_{best_single}"] = (
        singles[best_single],
        oofs[best_single],
        tests[best_single],
    )
    best_name = max(candidates, key=lambda n: candidates[n][0])
    best_auc, _, best_test = candidates[best_name]

    log("\n=== Blend OOF ===")
    for n, (a, _, _) in candidates.items():
        mark = " <-- chosen" if n == best_name else ""
        log(f"  {n}: {a:.5f}{mark}")
    log(f"hill-climb weights: {weights}")

    ids = te[ID_COL]
    write_sub(ids, best_test, OUT_DIR / "submission_blend.csv", sample)
    write_sub(ids, best_test, sub_root / "submission_blend.csv", sample)
    write_sub(ids, tests[best_single], OUT_DIR / f"submission_{best_single}.csv", sample)
    write_sub(ids, tests[best_single], sub_root / f"submission_{best_single}.csv", sample)
    rest = sorted(singles, key=singles.get, reverse=True)
    extra = next((m for m in rest if m != best_single), None)
    if extra is not None:
        write_sub(ids, tests[extra], OUT_DIR / f"submission_{extra}.csv", sample)
        write_sub(ids, tests[extra], sub_root / f"submission_{extra}.csv", sample)

    rec = pd.read_csv(OUT_DIR / "submission_blend.csv")
    rec.to_csv(sub_root / "submission.csv", index=False)
    rec.to_csv(OUT_DIR / "submission.csv", index=False)

    summary = {
        "metric": "roc_auc",
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features_base": int(Xtr.shape[1]),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "used_original": og is not None,
        "smoke": bool(args.smoke),
        "single_oof": singles,
        "blend_oof": {n: float(a) for n, (a, _, _) in candidates.items()},
        "chosen_blend": best_name,
        "chosen_oof_auc": float(best_auc),
        "best_single": best_single,
        "best_single_oof": singles[best_single],
        "hillclimb_weights": weights,
        "fold_scores": fold_rows,
        "submission_mean": float(np.clip(best_test, 1e-7, 1 - 1e-7).mean()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "recommended_file": "submission_blend.csv",
    }
    (OUT_DIR / "cv_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("=" * 62)
    log(f"CHOSEN {best_name} OOF={best_auc:.5f}  [{(time.time()-t0)/60:.1f} min]")
    log("=" * 62)


if __name__ == "__main__":
    main()
