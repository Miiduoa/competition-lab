#!/usr/bin/env python3
"""V3 train: striatum features + LGBM/HGB/LR ensemble + prior blend."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))
from features import FEATURE_NAMES, extract_features  # noqa: E402

DATA = ROOT / "data"
NIFTI_DIR = DATA / "niftis"
LABELS = DATA / "train_labels.csv"
ASSETS = ROOT / "submission_src" / "assets"
ART = ROOT / "artifacts"
FEAT_CACHE = ART / "train_features_v3.csv"


def load_or_build_features(force: bool = False) -> pd.DataFrame:
    labels = pd.read_csv(LABELS)
    if FEAT_CACHE.exists() and not force:
        cached = pd.read_csv(FEAT_CACHE)
        if (
            set(cached["uid"]) == set(labels["uid"])
            and all(c in cached.columns for c in FEATURE_NAMES)
            and len(cached) == len(labels)
        ):
            print(f"Using cached features: {FEAT_CACHE}", flush=True)
            return cached.merge(labels, on="uid", how="inner")

    rows = []
    t0 = time.time()
    n = len(labels)
    for i, row in enumerate(labels.itertuples(index=False)):
        uid = row.uid
        path = NIFTI_DIR / f"{uid}.nii.gz"
        if not path.exists():
            feats = np.zeros(len(FEATURE_NAMES))
        else:
            img = nib.load(str(path))
            arr = np.asanyarray(img.dataobj)
            feats = extract_features(arr)
        rows.append({"uid": uid, **dict(zip(FEATURE_NAMES, feats))})
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  features {i+1}/{n}  ({time.time()-t0:.1f}s)", flush=True)
    feat_df = pd.DataFrame(rows)
    ART.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEAT_CACHE, index=False)
    return feat_df.merge(labels, on="uid", how="inner")


def _lgbm():
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=24,
        max_depth=5,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=42,
        verbose=-1,
    )


def _hgb():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=42,
    )


def _lr():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    C=0.3,
                    class_weight=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def blend_prior(p: np.ndarray, prior: float, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * p + alpha * prior


def tune_alpha(y: np.ndarray, p: np.ndarray, prior: float) -> tuple[float, float]:
    best_a, best_ll = 0.0, log_loss(y, np.clip(p, 1e-3, 1 - 1e-3))
    for a in np.linspace(0.0, 0.30, 31):
        pb = np.clip(blend_prior(p, prior, a), 1e-3, 1 - 1e-3)
        ll = log_loss(y, pb)
        if ll < best_ll:
            best_ll, best_a = ll, float(a)
    return best_a, float(best_ll)


def train(df: pd.DataFrame) -> dict:
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["is_pathologic"].values.astype(int)
    prior = float(np.clip(y.mean(), 0.01, 0.99))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y))
    oof_hgb = np.zeros(len(y))
    oof_lr = np.zeros(len(y))
    fold_metrics = []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        # Nested calibration: CalibratedClassifierCV with cv=3 on train fold
        lgb_cal = CalibratedClassifierCV(_lgbm(), method="isotonic", cv=3)
        hgb_cal = CalibratedClassifierCV(_hgb(), method="isotonic", cv=3)
        lr_cal = CalibratedClassifierCV(_lr(), method="sigmoid", cv=3)

        lgb_cal.fit(X[tr], y[tr])
        hgb_cal.fit(X[tr], y[tr])
        lr_cal.fit(X[tr], y[tr])

        p_lgb = lgb_cal.predict_proba(X[va])[:, 1]
        p_hgb = hgb_cal.predict_proba(X[va])[:, 1]
        p_lr = lr_cal.predict_proba(X[va])[:, 1]
        oof_lgb[va] = p_lgb
        oof_hgb[va] = p_hgb
        oof_lr[va] = p_lr

        # Soft ensemble weights tuned later; report simple mean for fold
        p_ens = (p_lgb + p_hgb + p_lr) / 3.0
        ll = log_loss(y[va], np.clip(p_ens, 1e-3, 1 - 1e-3))
        auc = roc_auc_score(y[va], p_ens)
        fold_metrics.append(
            {
                "fold": fold,
                "log_loss": float(ll),
                "auc": float(auc),
                "ll_lgb": float(log_loss(y[va], np.clip(p_lgb, 1e-3, 1 - 1e-3))),
                "ll_hgb": float(log_loss(y[va], np.clip(p_hgb, 1e-3, 1 - 1e-3))),
                "ll_lr": float(log_loss(y[va], np.clip(p_lr, 1e-3, 1 - 1e-3))),
            }
        )
        print(
            f"fold {fold}: ens_ll={ll:.4f} auc={auc:.4f} "
            f"(lgb={fold_metrics[-1]['ll_lgb']:.4f} hgb={fold_metrics[-1]['ll_hgb']:.4f} lr={fold_metrics[-1]['ll_lr']:.4f})",
            flush=True,
        )

    # Optimize ensemble weights on OOF
    best_w = (1 / 3, 1 / 3, 1 / 3)
    best_ll = 1e9
    for w1 in np.linspace(0.2, 0.6, 9):
        for w2 in np.linspace(0.1, 0.5, 9):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05 or w3 > 0.5:
                continue
            p = w1 * oof_lgb + w2 * oof_hgb + w3 * oof_lr
            ll = log_loss(y, np.clip(p, 1e-3, 1 - 1e-3))
            if ll < best_ll:
                best_ll = ll
                best_w = (float(w1), float(w2), float(w3))

    oof_ens = best_w[0] * oof_lgb + best_w[1] * oof_hgb + best_w[2] * oof_lr
    alpha, ll_blend = tune_alpha(y, oof_ens, prior)
    oof_final = np.clip(blend_prior(oof_ens, prior, alpha), 1e-3, 1 - 1e-3)

    cv_ll = float(log_loss(y, oof_final))
    cv_auc = float(roc_auc_score(y, oof_final))
    cv_brier = float(brier_score_loss(y, oof_final))
    prior_ll = float(log_loss(y, np.full_like(oof_final, prior)))
    raw_ll = float(log_loss(y, np.clip(oof_ens, 1e-3, 1 - 1e-3)))

    # Calibration diagnostics: ECE-like
    def ece(y_true, p, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        e = 0.0
        for i in range(n_bins):
            m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
            if m.sum() == 0:
                continue
            e += abs(p[m].mean() - y_true[m].mean()) * (m.sum() / len(p))
        return float(e)

    print(
        f"OOF raw_ens_ll={raw_ll:.4f} blended_ll={cv_ll:.4f} auc={cv_auc:.4f} "
        f"brier={cv_brier:.4f} ece={ece(y, oof_final):.4f} | prior_ll={prior_ll:.4f}",
        flush=True,
    )
    print(f"weights lgb/hgb/lr={best_w} alpha={alpha:.3f}", flush=True)

    # Fit final models on all data
    print("Fitting final models on all data...", flush=True)
    final_lgb = CalibratedClassifierCV(_lgbm(), method="isotonic", cv=5)
    final_hgb = CalibratedClassifierCV(_hgb(), method="isotonic", cv=5)
    final_lr = CalibratedClassifierCV(_lr(), method="sigmoid", cv=5)
    final_lgb.fit(X, y)
    final_hgb.fit(X, y)
    final_lr.fit(X, y)

    ASSETS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_lgb, "feature_names": FEATURE_NAMES}, ASSETS / "lgbm_calibrated.joblib")
    joblib.dump({"model": final_hgb, "feature_names": FEATURE_NAMES}, ASSETS / "hgb_calibrated.joblib")
    joblib.dump({"model": final_lr, "feature_names": FEATURE_NAMES}, ASSETS / "lr_calibrated.joblib")
    (ASSETS / "prior.txt").write_text(f"{prior:.6f}\n")

    meta = {
        "version": "v3",
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "n_train": int(len(y)),
        "prior": prior,
        "ensemble_weights": {"lgbm": best_w[0], "hgb": best_w[1], "lr": best_w[2]},
        "prior_blend_alpha": alpha,
        "clip": [1e-3, 1 - 1e-3],
        "cv_log_loss": cv_ll,
        "cv_log_loss_raw_ens": raw_ll,
        "cv_auc": cv_auc,
        "cv_brier": cv_brier,
        "cv_ece": ece(y, oof_final),
        "prior_log_loss": prior_ll,
        "fold_metrics": fold_metrics,
        "primary_models": [
            "lgbm_calibrated.joblib",
            "hgb_calibrated.joblib",
            "lr_calibrated.joblib",
        ],
        "oof_pred_mean": float(oof_final.mean()),
        "oof_pred_std": float(oof_final.std()),
        "oof_pred_min": float(oof_final.min()),
        "oof_pred_max": float(oof_final.max()),
    }
    (ASSETS / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (ART / "cv_metrics_v3.json").write_text(json.dumps(meta, indent=2))
    # Save OOF for analysis
    oof_df = pd.DataFrame(
        {
            "uid": df["uid"].values,
            "y": y,
            "p_lgb": oof_lgb,
            "p_hgb": oof_hgb,
            "p_lr": oof_lr,
            "p_ens": oof_ens,
            "p_final": oof_final,
        }
    )
    oof_df.to_csv(ART / "oof_v3.csv", index=False)
    print("Saved models to", ASSETS, flush=True)
    return meta


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    print("Building V3 features...", flush=True)
    df = load_or_build_features(force=force)
    print(f"train rows={len(df)} pos_rate={df['is_pathologic'].mean():.4f} n_feat={len(FEATURE_NAMES)}", flush=True)
    # Quick feature sanity
    X = df[FEATURE_NAMES]
    print("feature nan count", int(X.isna().sum().sum()), "const cols", [c for c in FEATURE_NAMES if X[c].std() < 1e-12], flush=True)
    train(df)


if __name__ == "__main__":
    main()
