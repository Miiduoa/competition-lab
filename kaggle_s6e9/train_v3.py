#!/usr/bin/env python3
"""
S6E9 v3 — same leak-safe digit/artifact FE + nested TE as v2c,
plus CatBoost (+ optional XGBoost) and OOF hill-climb blend with v2c.
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
TARGET = "Will_Buy_EV"
SEED_LIST_LGBM = [11, 202, 3407]  # only used if retrain_lgbm
SEED_LIST_CB = [11, 202]
SEED_LIST_XGB = [11]
NFOLD = 5
RETRAIN_LGBM = False  # reuse outputs/oof_v2c.npy + submission_v2.csv

NOMINAL = ["Gender", "City_Type", "Current_Car_Type"]
RAW_NUM = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
INC_DIVS = [100, 250, 500, 1000, 2000, 5000]
KM_DIVS = [5, 10, 25, 50]
TE_SPECS = [
    ("te_inc_raw", "inc", 1),
    ("te_inc_100", "inc", 100),
    ("te_inc_1000", "inc", 1000),
    ("te_km", "km", 1),
]
TE_PRIOR = 200.0

PARAMS_LGBM = {
    "learning_rate": 0.0210433543287526,
    "num_leaves": 39,
    "max_depth": 8,
    "min_data_in_leaf": 23,
    "feature_fraction": 0.3715391684717929,
    "bagging_fraction": 0.6316507431228533,
    "bagging_freq": 1,
    "lambda_l1": 0.1401470578462424,
    "lambda_l2": 0.9611382109014246,
    "min_gain_to_split": 5.00349824543674e-05,
    "max_bin": 8191,
    "cat_smooth": 4.72792687673066,
}
LGB_FIXED = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_threads": 8,
    "force_col_wise": True,
}

PARAMS_CB = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
    "random_strength": 0.5,
    "bagging_temperature": 0.2,
    "iterations": 8000,
    "od_type": "Iter",
    "od_wait": 200,
    "thread_count": 8,
    "verbose": False,
    "allow_writing_files": False,
}

PARAMS_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.025,
    "max_depth": 8,
    "min_child_weight": 20,
    "subsample": 0.65,
    "colsample_bytree": 0.40,
    "reg_alpha": 0.15,
    "reg_lambda": 1.0,
    "max_bin": 512,
    "tree_method": "hist",
    "nthread": 8,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _is_text(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))


def _yes(s):
    if _is_text(s):
        return (
            s.astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "y", "true", "1"])
            .astype(np.int8)
            .to_numpy()
        )
    return s.astype(np.int8).to_numpy()


def _tz(a):
    z = np.zeros(len(a), dtype=np.int8)
    for k, m in enumerate([10, 100, 1000, 10000], start=1):
        z[a % m == 0] = k
    return z


def _keys(df):
    inc = df.Annual_Income_USD.to_numpy(np.int64)
    km = np.round(df.Daily_Commute_km.to_numpy(float) * 10).astype(np.int64)
    return inc, km


def base_frame(df):
    X = pd.DataFrame(index=range(len(df)))
    d = df.reset_index(drop=True)
    for c in RAW_NUM:
        X[c] = d[c].astype(np.float32)
    for c in ["Home_Charging_Possible", "Subsidy_Available"]:
        X[c] = _yes(d[c])
    ral = d.Range_Anxiety_Level
    if _is_text(ral):
        X["Range_Anxiety_Level"] = (
            ral.map({"Low": 0, "Medium": 1, "High": 2}).astype(np.int8)
        )
    else:
        X["Range_Anxiety_Level"] = ral.astype(np.int8)
    for c in NOMINAL:
        X[c] = d[c].astype("category")
    inc, km = _keys(d)
    X["inc_last_digit"] = (inc % 10).astype(np.int8)
    X["inc_mod_100"] = (inc % 100).astype(np.int16)
    X["inc_mod_1000"] = (inc % 1000).astype(np.int16)
    X["inc_mod_10000"] = (inc % 10000).astype(np.int32)
    X["inc_tz"] = _tz(inc)
    X["inc_at_floor"] = (inc == 30000).astype(np.int8)
    X["inc_round_100"] = (inc % 100 == 0).astype(np.int8)
    X["inc_round_1000"] = (inc % 1000 == 0).astype(np.int8)
    X["km_last_digit"] = (km % 10).astype(np.int8)
    X["km_mod_100"] = (km % 100).astype(np.int8)
    X["km_at_floor"] = (km == 50).astype(np.int8)
    X["km_round_1"] = (km % 10 == 0).astype(np.int8)
    for dv in INC_DIVS:
        X[f"inc_div_{dv}"] = (inc // dv).astype(np.int32)
    for dv in KM_DIVS:
        X[f"km_div_{dv}"] = (km // dv).astype(np.int16)
    return X


class TargetEncoder:
    def __init__(self, prior=TE_PRIOR, n_splits=5, seed=7):
        self.prior, self.n_splits, self.seed = prior, n_splits, seed
        self.maps_, self.gm_ = {}, None

    @staticmethod
    def _key(df, kind, div):
        inc, km = _keys(df.reset_index(drop=True))
        v = inc if kind == "inc" else km
        return v // div if div > 1 else v

    def fit_transform(self, df, y):
        self.gm_ = float(np.mean(y))
        out = {}
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        folds = list(skf.split(np.zeros(len(y)), y))
        for name, kind, div in TE_SPECS:
            k = self._key(df, kind, div)
            enc = np.full(len(y), self.gm_, dtype=np.float64)
            for a, b in folds:
                s = pd.DataFrame({"k": k[a], "y": y[a]}).groupby("k").y.agg(
                    ["sum", "count"]
                )
                sm = (s["sum"] + self.prior * self.gm_) / (s["count"] + self.prior)
                enc[b] = pd.Series(k[b]).map(sm).fillna(self.gm_).to_numpy()
            out[name] = enc.astype(np.float32)
            s = pd.DataFrame({"k": k, "y": y}).groupby("k").y.agg(["sum", "count"])
            self.maps_[name] = (s["sum"] + self.prior * self.gm_) / (
                s["count"] + self.prior
            )
        return out

    def transform(self, df):
        return {
            name: pd.Series(self._key(df, kind, div))
            .map(self.maps_[name])
            .fillna(self.gm_)
            .to_numpy()
            .astype(np.float32)
            for name, kind, div in TE_SPECS
        }


def build(df_tr, y_tr, *others):
    enc = TargetEncoder()
    cols_tr = enc.fit_transform(df_tr, y_tr)
    A = base_frame(df_tr)
    for k, v in cols_tr.items():
        A[k] = v
    outs = []
    for o in others:
        B = base_frame(o)
        for k, v in enc.transform(o).items():
            B[k] = v
        for c in A.columns:
            if str(A[c].dtype) == "category":
                B[c] = pd.Categorical(B[c], categories=A[c].cat.categories)
        outs.append(B[A.columns])
    return (A, *outs)


def to_cb(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CatBoost: convert category cols to string; return cat feature names."""
    X = df.copy()
    cats = []
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(str)
            cats.append(c)
    return X, cats


def to_xgb(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.astype(np.int16)
    return X.astype(np.float32)


def train_lgbm(tr, te, y, t0):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    fold_rows = []
    for seed in SEED_LIST_LGBM:
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(y))
        for k, (a, b) in enumerate(skf.split(tr, y)):
            A, B, T = build(tr.iloc[a], y[a], tr.iloc[b], te)
            d1 = lgb.Dataset(A, y[a])
            m = lgb.train(
                {**PARAMS_LGBM, **LGB_FIXED, "seed": seed},
                d1,
                20000,
                valid_sets=[lgb.Dataset(B, y[b], reference=d1)],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            it = m.best_iteration
            pv = m.predict(B, num_iteration=it)
            pt = m.predict(T, num_iteration=it)
            seed_oof[b] = pv
            test_pred += pt / (NFOLD * len(SEED_LIST_LGBM))
            auc = roc_auc_score(y[b], pv)
            fold_rows.append({"model": "lgbm", "seed": seed, "fold": k, "auc": float(auc), "iter": int(it)})
            log(f"  [LGBM] seed {seed} fold {k}: AUC={auc:.5f} it={it}  [{(time.time()-t0)/60:.1f} min]")
            del A, B, T, d1, m
            gc.collect()
        oof += seed_oof / len(SEED_LIST_LGBM)
        log(f"  [LGBM] seed {seed} OOF={roc_auc_score(y, seed_oof):.5f}\n")
    return oof, test_pred, fold_rows


def train_catboost(tr, te, y, t0):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    fold_rows = []
    for seed in SEED_LIST_CB:
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(y))
        for k, (a, b) in enumerate(skf.split(tr, y)):
            A, B, T = build(tr.iloc[a], y[a], tr.iloc[b], te)
            Ac, cats = to_cb(A)
            Bc, _ = to_cb(B)
            Tc, _ = to_cb(T)
            # align categories present in train
            for c in cats:
                Bc[c] = Bc[c].astype(str)
                Tc[c] = Tc[c].astype(str)
            train_pool = Pool(Ac, y[a], cat_features=cats)
            valid_pool = Pool(Bc, y[b], cat_features=cats)
            test_pool = Pool(Tc, cat_features=cats)
            m = CatBoostClassifier(**PARAMS_CB, random_seed=seed)
            m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            it = m.get_best_iteration() or m.tree_count_
            pv = m.predict_proba(valid_pool)[:, 1]
            pt = m.predict_proba(test_pool)[:, 1]
            seed_oof[b] = pv
            test_pred += pt / (NFOLD * len(SEED_LIST_CB))
            auc = roc_auc_score(y[b], pv)
            fold_rows.append({"model": "catboost", "seed": seed, "fold": k, "auc": float(auc), "iter": int(it)})
            log(f"  [CB] seed {seed} fold {k}: AUC={auc:.5f} it={it}  [{(time.time()-t0)/60:.1f} min]")
            del A, B, T, Ac, Bc, Tc, train_pool, valid_pool, test_pool, m
            gc.collect()
        oof += seed_oof / len(SEED_LIST_CB)
        log(f"  [CB] seed {seed} OOF={roc_auc_score(y, seed_oof):.5f}\n")
    return oof, test_pred, fold_rows


def train_xgboost(tr, te, y, t0):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    fold_rows = []
    for seed in SEED_LIST_XGB:
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(y))
        for k, (a, b) in enumerate(skf.split(tr, y)):
            A, B, T = build(tr.iloc[a], y[a], tr.iloc[b], te)
            Ax, Bx, Tx = to_xgb(A), to_xgb(B), to_xgb(T)
            dtrain = xgb.DMatrix(Ax, label=y[a])
            dvalid = xgb.DMatrix(Bx, label=y[b])
            dtest = xgb.DMatrix(Tx)
            params = {**PARAMS_XGB, "seed": seed}
            m = xgb.train(
                params,
                dtrain,
                num_boost_round=8000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            it = m.best_iteration + 1
            pv = m.predict(dvalid, iteration_range=(0, it))
            pt = m.predict(dtest, iteration_range=(0, it))
            seed_oof[b] = pv
            test_pred += pt / (NFOLD * len(SEED_LIST_XGB))
            auc = roc_auc_score(y[b], pv)
            fold_rows.append({"model": "xgboost", "seed": seed, "fold": k, "auc": float(auc), "iter": int(it)})
            log(f"  [XGB] seed {seed} fold {k}: AUC={auc:.5f} it={it}  [{(time.time()-t0)/60:.1f} min]")
            del A, B, T, Ax, Bx, Tx, dtrain, dvalid, dtest, m
            gc.collect()
        oof += seed_oof / len(SEED_LIST_XGB)
        log(f"  [XGB] seed {seed} OOF={roc_auc_score(y, seed_oof):.5f}\n")
    return oof, test_pred, fold_rows


def rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    from scipy.stats import rankdata
    ranks = [rankdata(p) / len(p) for p in preds]
    return np.mean(ranks, axis=0)


def hill_climb(oofs: dict[str, np.ndarray], y: np.ndarray, step=0.02):
    """Greedy weight search on OOF (probability space). Start from equal/best."""
    names = list(oofs.keys())
    mats = np.column_stack([oofs[n] for n in names])
    # start: best single
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
                pred = mats @ w
                auc = roc_auc_score(y, pred)
                if auc > best_auc + 1e-7:
                    best_auc = auc
                    best_w = w
                    improved = True
    return {n: float(w) for n, w in zip(names, best_w)}, float(best_auc)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    y = _yes(tr[TARGET])
    tr = tr.drop(columns=[TARGET])
    log(f"[data] train={tr.shape} test={te.shape} pos={y.mean():.5f}")

    oofs = {}
    tests = {}
    fold_rows = []

    # --- v2c LGBM ---
    oof_path = OUT_DIR / "oof_v2c.npy"
    sub_v2 = OUT_DIR / "submission_v2.csv"
    if (not RETRAIN_LGBM) and oof_path.exists() and sub_v2.exists():
        oofs["lgbm_v2c"] = np.load(oof_path)
        tests["lgbm_v2c"] = pd.read_csv(sub_v2)[TARGET].to_numpy(np.float64)
        log(f"[reuse] lgbm_v2c OOF={roc_auc_score(y, oofs['lgbm_v2c']):.5f}")
    else:
        oof, tp, fr = train_lgbm(tr, te, y, t0)
        oofs["lgbm_v2c"] = oof
        tests["lgbm_v2c"] = tp
        fold_rows.extend(fr)
        np.save(OUT_DIR / "oof_v2c.npy", oof)

    # --- CatBoost ---
    log("\n=== CatBoost ===")
    oof_cb, tp_cb, fr_cb = train_catboost(tr, te, y, t0)
    oofs["catboost"] = oof_cb
    tests["catboost"] = tp_cb
    fold_rows.extend(fr_cb)
    np.save(OUT_DIR / "oof_v3_catboost.npy", oof_cb)
    log(f"[CB] overall OOF={roc_auc_score(y, oof_cb):.5f}")

    # --- XGBoost ---
    if HAS_XGB:
        log("\n=== XGBoost ===")
        oof_xgb, tp_xgb, fr_xgb = train_xgboost(tr, te, y, t0)
        oofs["xgboost"] = oof_xgb
        tests["xgboost"] = tp_xgb
        fold_rows.extend(fr_xgb)
        np.save(OUT_DIR / "oof_v3_xgboost.npy", oof_xgb)
        log(f"[XGB] overall OOF={roc_auc_score(y, oof_xgb):.5f}")

    # --- Blends ---
    singles = {k: float(roc_auc_score(y, v)) for k, v in oofs.items()}
    log("\n=== Single-model OOF ===")
    for k, v in singles.items():
        log(f"  {k}: {v:.5f}")

    # equal probability average
    eq = np.mean(list(oofs.values()), axis=0)
    eq_auc = float(roc_auc_score(y, eq))
    # rank average
    ra = rank_avg(list(oofs.values()))
    ra_auc = float(roc_auc_score(y, ra))
    # hill-climb
    weights, hc_auc = hill_climb(oofs, y, step=0.02)
    hc_oof = sum(weights[k] * oofs[k] for k in weights)
    hc_test = sum(weights[k] * tests[k] for k in weights)

    # also try rank-avg of models with weight > 0.05 from hill-climb
    keep = [k for k, w in weights.items() if w >= 0.05]
    if len(keep) >= 2:
        ra2 = rank_avg([oofs[k] for k in keep])
        ra2_auc = float(roc_auc_score(y, ra2))
        ra2_test = rank_avg([tests[k] for k in keep])
    else:
        ra2, ra2_auc, ra2_test = ra, ra_auc, rank_avg(list(tests.values()))

    candidates = {
        "equal_prob": (eq_auc, eq, np.mean(list(tests.values()), axis=0)),
        "rank_avg_all": (ra_auc, ra, rank_avg(list(tests.values()))),
        "hillclimb": (hc_auc, hc_oof, hc_test),
        "rank_avg_kept": (ra2_auc, ra2, ra2_test),
    }
    # include best single
    best_single_name = max(singles, key=singles.get)
    candidates[f"single_{best_single_name}"] = (
        singles[best_single_name],
        oofs[best_single_name],
        tests[best_single_name],
    )

    best_name = max(candidates, key=lambda n: candidates[n][0])
    best_auc, best_oof, best_test = candidates[best_name]

    log("\n=== Blend OOF ===")
    for n, (a, _, _) in candidates.items():
        mark = " <-- chosen" if n == best_name else ""
        log(f"  {n}: {a:.5f}{mark}")
    log(f"hill-climb weights: {weights}")

    # GO rule: OOF >= 0.94540, or >= 0.9453 with diversity (blend of >=2 models)
    diverse = best_name.startswith(("equal", "rank", "hill")) and len([w for w in weights.values() if w > 0.05]) >= 2
    if best_name.startswith("single_"):
        diverse = False
    go = bool(best_auc >= 0.94540 or (best_auc >= 0.9453 and diverse))

    best_test = np.clip(best_test, 1e-7, 1 - 1e-7)
    sub = pd.DataFrame({"id": te["id"], TARGET: best_test})
    sub.to_csv(OUT_DIR / "submission_v3.csv", index=False)
    if go:
        sub.to_csv(ROOT / "submission.csv", index=False)
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        log("GO: wrote submission.csv")
    else:
        log("NO-GO: did not overwrite submission.csv")

    np.save(OUT_DIR / "oof_v3.npy", best_oof)

    summary = {
        "version": "v3",
        "method": "v2c LGBM + CatBoost(+XGB) same leak-safe FE/TE; hill-climb/rank blend",
        "single_oof": singles,
        "blend_oof": {n: float(a) for n, (a, _, _) in candidates.items()},
        "chosen_blend": best_name,
        "chosen_oof_auc": float(best_auc),
        "hillclimb_weights": weights,
        "fold_scores": fold_rows,
        "prior_v2c_oof": 0.94540,
        "go_upload": go,
        "diverse": diverse,
        "n_test": int(len(te)),
        "submission_mean": float(best_test.mean()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (OUT_DIR / "cv_results_v3.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("=" * 62)
    log(f"CHOSEN {best_name} OOF={best_auc:.5f}  go={go}  [{(time.time()-t0)/60:.1f} min]")
    log("=" * 62)
    log(json.dumps({k: summary[k] for k in ["chosen_blend", "chosen_oof_auc", "hillclimb_weights", "go_upload", "single_oof"]}, indent=2))


if __name__ == "__main__":
    main()
