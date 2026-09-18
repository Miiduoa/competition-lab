#!/usr/bin/env python3
"""
S6E9 v2b: native categoricals + Optuna-tuned LGBM/XGB + CatBoost (no class balance)
+ OOF blend. Goal: clear CV lift over 0.94163 baseline.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
TARGET = "Will_Buy_EV"
ID_COL = "id"
SEED = 42
N_SPLITS = 5

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
# low-cardinality numerics as cats too
CAT_AS_CAT = CAT_COLS + [
    "Number_of_Cars_Owned",
    "Environmental_Concern_Level",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
]
NUM_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km"]


def log(msg: str) -> None:
    print(msg, flush=True)


def encode_y(s: pd.Series) -> np.ndarray:
    return (s.astype(str).str.strip().str.lower() == "yes").astype(int).values


def make_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_charging"] = (
        out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    )
    out["log_income"] = np.log1p(out["Annual_Income_USD"])
    out["income_per_car"] = out["Annual_Income_USD"] / (out["Number_of_Cars_Owned"] + 1)
    out["commute_per_charge"] = out["Daily_Commute_km"] / (out["total_charging"] + 1)
    out["env_x_income"] = out["Environmental_Concern_Level"] * out["log_income"]
    # ensure cat dtypes as category codes for lgb/xgb
    for c in CAT_AS_CAT:
        out[c] = out[c].astype(str)
    return out


def to_codes(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    tr, te = train.copy(), test.copy()
    for c in cols:
        cats = pd.Categorical(tr[c].astype(str))
        mapping = {v: i for i, v in enumerate(cats.categories)}
        tr[c] = tr[c].astype(str).map(mapping).astype("int32")
        te[c] = te[c].astype(str).map(mapping).fillna(-1).astype("int32")
    return tr, te


def optuna_lgbm(X, y, cat_idx, n_trials=25):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # subsample rows for speed
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(200_000, len(y)), replace=False)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "subsample_freq": 1,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        scores = []
        for tr, va in skf.split(Xs, ys):
            m = lgb.LGBMClassifier(**params)
            m.fit(
                Xs.iloc[tr],
                ys[tr],
                eval_set=[(Xs.iloc[va], ys[va])],
                eval_metric="auc",
                categorical_feature=cat_idx,
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(Xs.iloc[va])[:, 1]
            scores.append(roc_auc_score(ys[va], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"[optuna-lgbm] best={study.best_value:.5f} params={study.best_params}")
    return study.best_params


def optuna_xgb(X, y, n_trials=20):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(200_000, len(y)), replace=False)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": SEED,
            "n_jobs": -1,
            "early_stopping_rounds": 80,
        }
        scores = []
        for tr, va in skf.split(Xs, ys):
            m = xgb.XGBClassifier(**params)
            # XGB needs category dtype for enable_categorical
            Xtr = Xs.iloc[tr].copy()
            Xva = Xs.iloc[va].copy()
            for c in CAT_AS_CAT:
                if c in Xtr.columns:
                    Xtr[c] = Xtr[c].astype("category")
                    Xva[c] = Xva[c].astype("category")
            m.fit(Xtr, ys[tr], eval_set=[(Xva, ys[va])], verbose=False)
            p = m.predict_proba(Xva)[:, 1]
            scores.append(roc_auc_score(ys[va], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"[optuna-xgb] best={study.best_value:.5f} params={study.best_params}")
    return study.best_params


def blend_weights(oof: dict, y):
    names = list(oof)
    mats = np.column_stack([oof[n] for n in names])
    best_w, best_auc = None, -1.0
    grid = np.linspace(0, 1, 21)
    if len(names) == 3:
        for a in grid:
            for b in grid:
                if a + b > 1:
                    continue
                w = np.array([a, b, 1 - a - b])
                auc = roc_auc_score(y, mats @ w)
                if auc > best_auc:
                    best_auc, best_w = auc, w
    else:
        # generic equal
        best_w = np.ones(len(names)) / len(names)
        best_auc = roc_auc_score(y, mats @ best_w)
    return {n: float(w) for n, w in zip(names, best_w)}, float(best_auc)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = encode_y(train[TARGET])

    train_fe = make_frame(train)
    test_fe = make_frame(test)

    feat_num = NUM_COLS + [
        "total_charging",
        "log_income",
        "income_per_car",
        "commute_per_charge",
        "env_x_income",
    ]
    # frames for LGBM (integer codes + categorical_feature)
    tr_c, te_c = to_codes(train_fe, test_fe, CAT_AS_CAT)
    use_cols = feat_num + CAT_AS_CAT
    X = tr_c[use_cols].copy()
    X_test = te_c[use_cols].copy()
    cat_idx = [use_cols.index(c) for c in CAT_AS_CAT]

    # CatBoost native strings
    cb_cols = feat_num + CAT_AS_CAT
    X_cb = train_fe[cb_cols].copy()
    X_test_cb = test_fe[cb_cols].copy()
    for c in CAT_AS_CAT:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    log(f"[data] train={len(train)} feats={len(use_cols)} cats={len(CAT_AS_CAT)}")

    # Optuna
    try:
        import optuna  # noqa
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])

    lgb_best = optuna_lgbm(X, y, cat_idx, n_trials=20)
    xgb_best = optuna_xgb(X, y, n_trials=15)

    # Full CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(y)) for k in ("lgbm", "xgb", "cat")}
    tpred = {k: np.zeros(len(test)) for k in oof}
    fold_scores = {k: [] for k in oof}

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        log(f"\n===== Fold {fold}/{N_SPLITS} =====")

        # LGBM
        lp = dict(
            n_estimators=8000,
            learning_rate=lgb_best["lr"],
            num_leaves=lgb_best["num_leaves"],
            min_child_samples=lgb_best["min_child_samples"],
            subsample=lgb_best["subsample"],
            subsample_freq=1,
            colsample_bytree=lgb_best["colsample_bytree"],
            reg_alpha=lgb_best["reg_alpha"],
            reg_lambda=lgb_best["reg_lambda"],
            min_split_gain=lgb_best.get("min_split_gain", 0.0),
            random_state=SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        m = lgb.LGBMClassifier(**lp)
        m.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            categorical_feature=cat_idx,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof["lgbm"][va] = p
        tpred["lgbm"] += m.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], p)
        fold_scores["lgbm"].append(float(auc))
        log(f"  LGBM AUC={auc:.5f} iter={m.best_iteration_}")

        # XGB with category dtype
        Xtr = X.iloc[tr].copy()
        Xva = X.iloc[va].copy()
        Xt = X_test.copy()
        for c in CAT_AS_CAT:
            Xtr[c] = Xtr[c].astype("category")
            Xva[c] = pd.Categorical(Xva[c], categories=Xtr[c].cat.categories)
            Xt[c] = pd.Categorical(Xt[c], categories=Xtr[c].cat.categories)
        xp = dict(
            n_estimators=8000,
            learning_rate=xgb_best["lr"],
            max_depth=xgb_best["max_depth"],
            min_child_weight=xgb_best["min_child_weight"],
            subsample=xgb_best["subsample"],
            colsample_bytree=xgb_best["colsample_bytree"],
            reg_alpha=xgb_best["reg_alpha"],
            reg_lambda=xgb_best["reg_lambda"],
            gamma=xgb_best.get("gamma", 0.0),
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED + fold,
            n_jobs=-1,
            early_stopping_rounds=150,
        )
        m = xgb.XGBClassifier(**xp)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], verbose=False)
        p = m.predict_proba(Xva)[:, 1]
        oof["xgb"][va] = p
        tpred["xgb"] += m.predict_proba(Xt)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], p)
        fold_scores["xgb"].append(float(auc))
        log(f"  XGB  AUC={auc:.5f} iter={m.best_iteration}")

        # CatBoost — no class balance (better for AUC probs)
        m = CatBoostClassifier(
            iterations=8000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=4.0,
            random_seed=SEED + fold,
            eval_metric="AUC",
            loss_function="Logloss",
            early_stopping_rounds=150,
            verbose=False,
            thread_count=-1,
            bootstrap_type="Bernoulli",
            subsample=0.85,
        )
        cat_features = [cb_cols.index(c) for c in CAT_AS_CAT]
        m.fit(
            X_cb.iloc[tr],
            y[tr],
            eval_set=(X_cb.iloc[va], y[va]),
            cat_features=cat_features,
            use_best_model=True,
        )
        p = m.predict_proba(X_cb.iloc[va])[:, 1]
        oof["cat"][va] = p
        tpred["cat"] += m.predict_proba(X_test_cb)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], p)
        fold_scores["cat"].append(float(auc))
        log(f"  CAT  AUC={auc:.5f} iter={m.get_best_iteration()}")

    model_stats = {}
    for name in oof:
        auc = float(roc_auc_score(y, oof[name]))
        model_stats[name] = {
            "oof_auc": auc,
            "fold_mean": float(np.mean(fold_scores[name])),
            "fold_std": float(np.std(fold_scores[name])),
            "fold_scores": fold_scores[name],
        }
        log(f"[OOF] {name}: {auc:.5f}")

    weights, blend_auc = blend_weights(oof, y)
    log(f"[BLEND] {weights} OOF={blend_auc:.5f}")

    # rank blend candidate
    from scipy.stats import rankdata

    rank_oof = np.mean([rankdata(oof[n]) / len(y) for n in oof], axis=0)
    rank_auc = float(roc_auc_score(y, rank_oof))
    rank_test = np.mean([rankdata(tpred[n]) / len(test) for n in tpred], axis=0)
    log(f"[RANK] OOF={rank_auc:.5f}")

    if blend_auc >= rank_auc:
        chosen, chosen_auc = "opt_blend", blend_auc
        final = sum(weights[n] * tpred[n] for n in weights)
    else:
        chosen, chosen_auc = "rank_blend", rank_auc
        final = rank_test

    final = np.clip(final, 1e-7, 1 - 1e-7)
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final})
    sub.to_csv(OUT_DIR / "submission_v2.csv", index=False)
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)

    prior_cv = 0.9416286177505029
    prior_public = 0.94540
    delta = chosen_auc - prior_cv
    go = bool(chosen_auc >= prior_cv + 0.0005)

    summary = {
        "version": "v2b",
        "lgb_best_params": lgb_best,
        "xgb_best_params": xgb_best,
        "models": model_stats,
        "blend_weights": weights,
        "blend_oof_auc": blend_auc,
        "rank_oof_auc": rank_auc,
        "chosen_strategy": chosen,
        "chosen_oof_auc": chosen_auc,
        "prior_baseline_cv_auc": prior_cv,
        "prior_public_score": prior_public,
        "cv_delta_vs_baseline": delta,
        "go_upload": go,
        "n_features": len(use_cols),
        "categorical_as_cat": CAT_AS_CAT,
    }
    (OUT_DIR / "cv_results_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # keep v2a results
    log(json.dumps({k: summary[k] for k in [
        "chosen_strategy", "chosen_oof_auc", "prior_baseline_cv_auc",
        "cv_delta_vs_baseline", "go_upload", "blend_weights"
    ]}, indent=2))


if __name__ == "__main__":
    main()
