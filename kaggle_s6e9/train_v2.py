#!/usr/bin/env python3
"""
S6E9 v2: feature engineering + LGBM/XGB/CatBoost OOF ensemble.
Metric: ROC-AUC. Writes submission.csv + outputs/submission_v2.csv.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
TARGET = "Will_Buy_EV"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def encode_target(y: pd.Series) -> np.ndarray:
    return (y.astype(str).str.strip().str.lower() == "yes").astype(int).values


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_charging_stations"] = (
        out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    )
    out["log_income"] = np.log1p(out["Annual_Income_USD"])
    out["sqrt_income"] = np.sqrt(out["Annual_Income_USD"].clip(lower=0))
    out["income_per_car"] = out["Annual_Income_USD"] / (out["Number_of_Cars_Owned"] + 1)
    out["income_per_age"] = out["Annual_Income_USD"] / (out["Age"] + 1)
    out["commute_per_age"] = out["Daily_Commute_km"] / (out["Age"] + 1)
    out["commute_per_station"] = out["Daily_Commute_km"] / (
        out["total_charging_stations"] + 1
    )
    out["stations_home_ratio"] = out["Charging_Stations_Near_Home"] / (
        out["total_charging_stations"] + 1
    )
    out["home_yes"] = (out["Home_Charging_Possible"].astype(str) == "Yes").astype(int)
    out["subsidy_yes"] = (out["Subsidy_Available"].astype(str) == "Yes").astype(int)
    anx_map = {"Low": 0, "Medium": 1, "High": 2}
    out["anxiety_ord"] = out["Range_Anxiety_Level"].astype(str).map(anx_map).fillna(1)
    # Domain interactions
    out["env_x_subsidy"] = out["Environmental_Concern_Level"] * out["subsidy_yes"]
    out["env_x_home"] = out["Environmental_Concern_Level"] * out["home_yes"]
    out["env_minus_anxiety"] = out["Environmental_Concern_Level"] - out["anxiety_ord"]
    out["income_x_env"] = out["log_income"] * out["Environmental_Concern_Level"]
    out["home_x_low_anxiety"] = out["home_yes"] * (out["anxiety_ord"] == 0).astype(int)
    out["subsidy_x_low_anxiety"] = out["subsidy_yes"] * (out["anxiety_ord"] == 0).astype(
        int
    )
    out["urban_flag"] = (out["City_Type"].astype(str) == "Urban").astype(int)
    out["rural_flag"] = (out["City_Type"].astype(str) == "Rural").astype(int)
    out["age_bin"] = pd.cut(
        out["Age"], bins=[0, 30, 40, 50, 60, 100], labels=False
    ).astype(float)
    # Frequency of commute / charging mismatch signal
    out["long_commute"] = (out["Daily_Commute_km"] > 40).astype(int)
    out["high_income"] = (out["Annual_Income_USD"] > 90000).astype(int)
    out["high_env"] = (out["Environmental_Concern_Level"] >= 4).astype(int)
    return out


def add_freq_encode(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()
    for c in cols:
        freq = train[c].astype(str).value_counts(normalize=True)
        train[f"{c}_freq"] = train[c].astype(str).map(freq).astype(float)
        test[f"{c}_freq"] = test[c].astype(str).map(freq).fillna(0.0).astype(float)
    return train, test


def prepare_matrices(train: pd.DataFrame, test: pd.DataFrame):
    train_fe = engineer(train)
    test_fe = engineer(test)
    train_fe, test_fe = add_freq_encode(train_fe, test_fe, CAT_COLS)

    # Label-encode cats for LGBM/XGB; keep string cats for CatBoost
    cat_maps = {}
    for c in CAT_COLS:
        cats = pd.Categorical(train_fe[c].astype(str))
        cat_maps[c] = {v: i for i, v in enumerate(cats.categories)}
        train_fe[f"{c}_le"] = train_fe[c].astype(str).map(cat_maps[c]).astype(int)
        test_fe[f"{c}_le"] = (
            test_fe[c].astype(str).map(cat_maps[c]).fillna(-1).astype(int)
        )

    drop = {ID_COL, TARGET}
    # numeric features for tree models (use LE cats, drop raw strings)
    num_cols = [
        c
        for c in train_fe.columns
        if c not in drop
        and c not in CAT_COLS
        and pd.api.types.is_numeric_dtype(train_fe[c])
    ]
    X = train_fe[num_cols].copy()
    X_test = test_fe[num_cols].copy()

    # CatBoost: mix of numeric + native categoricals
    cat_feature_names = CAT_COLS[:]
    cb_cols = [
        c
        for c in train_fe.columns
        if c not in drop and (c in CAT_COLS or pd.api.types.is_numeric_dtype(train_fe[c]))
        and not c.endswith("_le")  # avoid duplicate encoding
    ]
    # keep raw cats as string for CatBoost
    X_cb = train_fe[cb_cols].copy()
    X_test_cb = test_fe[cb_cols].copy()
    for c in CAT_COLS:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    y = encode_target(train[TARGET])
    return X, X_test, X_cb, X_test_cb, y, num_cols, cat_feature_names


def train_lgbm(X_tr, y_tr, X_va, y_va):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def train_xgb(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=100,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def train_cat(X_tr, y_tr, X_va, y_va, cat_features):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        eval_metric="AUC",
        loss_function="Logloss",
        early_stopping_rounds=100,
        verbose=False,
        thread_count=-1,
        auto_class_weights="Balanced",
    )
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_features if c in X_tr.columns]
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_idx,
        use_best_model=True,
    )
    return model


def optimize_blend_weights(oof_dict: dict[str, np.ndarray], y: np.ndarray):
    """Grid-search simplex weights for soft blend."""
    names = list(oof_dict.keys())
    mats = np.column_stack([oof_dict[n] for n in names])
    best_w, best_auc = None, -1.0
    # coarse grid over simplex
    grid = np.linspace(0, 1, 11)
    if len(names) == 1:
        return {names[0]: 1.0}, float(roc_auc_score(y, mats[:, 0]))
    if len(names) == 2:
        for a in grid:
            w = np.array([a, 1 - a])
            auc = roc_auc_score(y, mats @ w)
            if auc > best_auc:
                best_auc, best_w = auc, w
    else:
        for a in grid:
            for b in grid:
                if a + b > 1:
                    continue
                c = 1 - a - b
                w = np.array([a, b, c])
                auc = roc_auc_score(y, mats @ w)
                if auc > best_auc:
                    best_auc, best_w = auc, w
    return {n: float(w) for n, w in zip(names, best_w)}, float(best_auc)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log(f"[data] train={train.shape} test={test.shape}")

    X, X_test, X_cb, X_test_cb, y, num_cols, cat_names = prepare_matrices(train, test)
    log(f"[feat] numeric_tree={len(num_cols)} catboost_cols={X_cb.shape[1]}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {"lgbm": np.zeros(len(y)), "xgb": np.zeros(len(y)), "cat": np.zeros(len(y))}
    test_preds = {"lgbm": np.zeros(len(test)), "xgb": np.zeros(len(test)), "cat": np.zeros(len(test))}
    fold_scores = {k: [] for k in oof}

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        log(f"\n===== Fold {fold}/{N_SPLITS} =====")
        # LGBM
        m = train_lgbm(X.iloc[tr], y[tr], X.iloc[va], y[va])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof["lgbm"][va] = p
        test_preds["lgbm"] += m.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], p)
        fold_scores["lgbm"].append(float(auc))
        log(f"  LGBM AUC={auc:.5f} best_iter={getattr(m, 'best_iteration_', None)}")

        # XGB
        m = train_xgb(X.iloc[tr], y[tr], X.iloc[va], y[va])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof["xgb"][va] = p
        test_preds["xgb"] += m.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], p)
        fold_scores["xgb"].append(float(auc))
        bi = getattr(m, "best_iteration", None)
        log(f"  XGB  AUC={auc:.5f} best_iter={bi}")

        # CatBoost
        m = train_cat(X_cb.iloc[tr], y[tr], X_cb.iloc[va], y[va], cat_names)
        p = m.predict_proba(X_cb.iloc[va])[:, 1]
        oof["cat"][va] = p
        test_preds["cat"] += m.predict_proba(X_test_cb)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], p)
        fold_scores["cat"].append(float(auc))
        log(f"  CAT  AUC={auc:.5f} best_iter={m.get_best_iteration()}")

    # Per-model OOF
    model_oof_auc = {}
    for name in oof:
        auc = float(roc_auc_score(y, oof[name]))
        model_oof_auc[name] = {
            "oof_auc": auc,
            "fold_mean": float(np.mean(fold_scores[name])),
            "fold_std": float(np.std(fold_scores[name])),
            "fold_scores": fold_scores[name],
        }
        log(f"[OOF] {name}: {auc:.5f} (fold mean {model_oof_auc[name]['fold_mean']:.5f})")

    weights, blend_auc = optimize_blend_weights(oof, y)
    log(f"[BLEND] weights={weights} OOF AUC={blend_auc:.5f}")

    # Also try equal weight and best single
    equal = np.mean([oof[n] for n in oof], axis=0)
    equal_auc = float(roc_auc_score(y, equal))
    best_single = max(model_oof_auc, key=lambda k: model_oof_auc[k]["oof_auc"])
    best_single_auc = model_oof_auc[best_single]["oof_auc"]
    log(f"[BLEND] equal OOF={equal_auc:.5f}; best_single={best_single} OOF={best_single_auc:.5f}")

    # Choose final: max of optimized blend / equal / best single
    candidates = {
        "opt_blend": (
            blend_auc,
            sum(weights[n] * test_preds[n] for n in weights),
            sum(weights[n] * oof[n] for n in weights),
        ),
        "equal": (equal_auc, equal if False else np.mean([test_preds[n] for n in oof], axis=0), equal),
        "best_single": (
            best_single_auc,
            test_preds[best_single],
            oof[best_single],
        ),
    }
    # fix equal test preds
    candidates["equal"] = (
        equal_auc,
        np.mean([test_preds[n] for n in oof], axis=0),
        equal,
    )

    chosen = max(candidates, key=lambda k: candidates[k][0])
    chosen_auc, final_test, final_oof = candidates[chosen]
    log(f"[FINAL] strategy={chosen} OOF AUC={chosen_auc:.5f}")

    # Clip probabilities slightly for numerical safety
    final_test = np.clip(final_test, 1e-7, 1 - 1e-7)

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_test})
    sub_path = OUT_DIR / "submission_v2.csv"
    sub.to_csv(sub_path, index=False)
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)
    log(f"[OUT] {sub_path} rows={len(sub)}")

    # prior baseline reference
    prior_cv = 0.9416286177505029
    prior_public = 0.94540
    delta = chosen_auc - prior_cv

    summary = {
        "version": "v2",
        "competition": "playground-series-s6e9",
        "metric": "ROC-AUC",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features_tree": len(num_cols),
        "models": model_oof_auc,
        "blend_weights": weights,
        "blend_oof_auc": blend_auc,
        "equal_oof_auc": equal_auc,
        "best_single": best_single,
        "best_single_oof_auc": best_single_auc,
        "chosen_strategy": chosen,
        "chosen_oof_auc": chosen_auc,
        "prior_baseline_cv_auc": prior_cv,
        "prior_public_score": prior_public,
        "cv_delta_vs_baseline": delta,
        "go_upload": bool(chosen_auc >= prior_cv + 0.0003),
        "feature_list": num_cols,
        "submission_v2": str(sub_path),
    }
    (OUT_DIR / "cv_results_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[OUT] {OUT_DIR / 'cv_results_v2.json'}")
    log(json.dumps({k: summary[k] for k in [
        "chosen_strategy","chosen_oof_auc","prior_baseline_cv_auc",
        "cv_delta_vs_baseline","go_upload","blend_weights"
    ]}, indent=2))


if __name__ == "__main__":
    main()
