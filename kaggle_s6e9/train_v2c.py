#!/usr/bin/env python3
"""
S6E9 v2c — reproduce strong public recipe (digit/artifact FE + nested TE +
high max_bin LGBM, multi-seed). Reference public OOF ~0.9454.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
TARGET = "Will_Buy_EV"
SEED_LIST = [11, 202, 3407]
NFOLD = 5

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
    "num_threads": -1,
    "force_col_wise": True,
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


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    y = _yes(tr[TARGET])
    tr = tr.drop(columns=[TARGET])
    log(f"[data] train={tr.shape} test={te.shape} pos={y.mean():.5f}")

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    fold_rows = []
    iters = []

    for seed in SEED_LIST:
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
            test_pred += pt / (NFOLD * len(SEED_LIST))
            iters.append(it)
            auc = roc_auc_score(y[b], pv)
            fold_rows.append({"seed": seed, "fold": k, "auc": float(auc), "iter": int(it)})
            log(
                f"  seed {seed} fold {k}: AUC={auc:.5f} it={it}  "
                f"[{(time.time() - t0) / 60:.1f} min]"
            )
            del A, B, T, d1, m
            gc.collect()
        oof += seed_oof / len(SEED_LIST)
        log(f"  seed {seed} OOF={roc_auc_score(y, seed_oof):.5f}\n")

    oof_auc = float(roc_auc_score(y, oof))
    log("=" * 62)
    log(f"OOF ROC-AUC {oof_auc:.5f}   (reference ~0.94540)")
    log(f"mean rounds  {int(np.mean(iters))}")
    log(f"total {(time.time() - t0) / 60:.1f} min")
    log("=" * 62)

    test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)
    sub = pd.DataFrame({"id": te["id"], TARGET: test_pred})
    sub.to_csv(OUT_DIR / "submission_v2.csv", index=False)
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)

    prior_cv = 0.9416286177505029
    prior_public = 0.94540
    delta = oof_auc - prior_cv
    go = bool(oof_auc >= prior_cv + 0.001)

    summary = {
        "version": "v2c",
        "method": "digit/artifact FE + nested TE + LGBM max_bin=8191, 5fold x 3seed",
        "params_lgbm": PARAMS_LGBM,
        "seeds": SEED_LIST,
        "fold_scores": fold_rows,
        "chosen_oof_auc": oof_auc,
        "mean_iter": int(np.mean(iters)),
        "prior_baseline_cv_auc": prior_cv,
        "prior_public_score": prior_public,
        "cv_delta_vs_baseline": delta,
        "go_upload": go,
        "reference_public_oof": 0.94540,
        "n_test": int(len(te)),
        "submission_mean": float(test_pred.mean()),
    }
    (OUT_DIR / "cv_results_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(OUT_DIR / "oof_v2c.npy", oof)
    log(json.dumps({k: summary[k] for k in [
        "chosen_oof_auc", "prior_baseline_cv_auc", "cv_delta_vs_baseline", "go_upload"
    ]}, indent=2))


if __name__ == "__main__":
    main()
