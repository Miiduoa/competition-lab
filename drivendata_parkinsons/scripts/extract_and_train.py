#!/usr/bin/env python3
"""本地訓練：DaT NIfTI → scale-invariant 特徵 → LightGBM + 機率校準。"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NIFTI_DIR = DATA / "niftis"
LABELS = DATA / "train_labels.csv"
ASSETS = ROOT / "submission_src" / "assets"
ART = ROOT / "artifacts"
FEAT_CACHE = ART / "train_features.csv"

FEATURE_NAMES = [
    "center_global_ratio",
    "center_bg_ratio",
    "peak1_median_ratio",
    "peak5_median_ratio",
    "frac_gt_half_max",
    "frac_gt_p95",
    "lr_asym_center",
    "ap_asym_center",
    "si_asym_center",
    "centroid_offset",
    "cv_nonzero",
    "p90_p50",
    "p99_p50",
    "center_cv",
    "shell_ratio",
    "top_mass_frac",
    "log_nnz_frac",
    "intensity_skew",
]


def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    if b == 0 or not np.isfinite(b):
        return default
    return float(a / b)


def _downsample(x: np.ndarray, max_dim: int = 96) -> np.ndarray:
    """Cheap integer-stride downsample to keep features fast."""
    if max(x.shape) <= max_dim:
        return x
    factors = [max(1, s // max_dim) for s in x.shape]
    return x[:: factors[0], :: factors[1], :: factors[2]]


def extract_features(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 3 or x.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    x = _downsample(x, 96)

    flat = x.ravel()
    nnz_mask = flat > 0
    if int(nnz_mask.sum()) < 10:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    nnz = flat[nnz_mask].astype(np.float64)
    gmean = float(nnz.mean())
    gmed = float(np.median(nnz))
    gmax = float(nnz.max())
    p50, p90, p95, p99 = np.percentile(nnz, [50, 90, 95, 99])

    z, y, xx = x.shape
    cz, cy, cx = z // 2, y // 2, xx // 2
    rz, ry, rx = max(z // 6, 1), max(y // 6, 1), max(xx // 6, 1)
    z0, z1 = max(0, cz - rz), min(z, cz + rz)
    y0, y1 = max(0, cy - ry), min(y, cy + ry)
    x0, x1 = max(0, cx - rx), min(xx, cx + rx)
    center = x[z0:z1, y0:y1, x0:x1]
    c_vals = center[center > 0]
    cmean = float(c_vals.mean()) if c_vals.size else 0.0
    cstd = float(c_vals.std()) if c_vals.size > 1 else 0.0

    mask = np.ones(x.shape, dtype=bool)
    mask[z0:z1, y0:y1, x0:x1] = False
    bg = x[mask & (x > 0)]
    bgmean = float(bg.mean()) if bg.size else gmean

    thr1 = float(np.percentile(nnz, 99))
    thr5 = float(np.percentile(nnz, 95))
    peak1 = float(nnz[nnz >= thr1].mean()) if (nnz >= thr1).any() else gmax
    peak5 = float(nnz[nnz >= thr5].mean()) if (nnz >= thr5).any() else gmax

    frac_half = float((nnz >= 0.5 * gmax).mean())
    frac_p95 = float((nnz >= p95).mean())

    mid_x = max(center.shape[2] // 2, 1)
    left, right = center[:, :, :mid_x], center[:, :, mid_x:]
    lm = float(left[left > 0].mean()) if (left > 0).any() else 0.0
    rm = float(right[right > 0].mean()) if (right > 0).any() else 0.0
    lr_asym = _safe_div(abs(lm - rm), (lm + rm) / 2 + 1e-6, 0.0)

    mid_y = max(center.shape[1] // 2, 1)
    ant, pos = center[:, :mid_y, :], center[:, mid_y:, :]
    am = float(ant[ant > 0].mean()) if (ant > 0).any() else 0.0
    pm = float(pos[pos > 0].mean()) if (pos > 0).any() else 0.0
    ap_asym = _safe_div(abs(am - pm), (am + pm) / 2 + 1e-6, 0.0)

    mid_z = max(center.shape[0] // 2, 1)
    sup, inf = center[:mid_z, :, :], center[mid_z:, :, :]
    sm = float(sup[sup > 0].mean()) if (sup > 0).any() else 0.0
    imean = float(inf[inf > 0].mean()) if (inf > 0).any() else 0.0
    si_asym = _safe_div(abs(sm - imean), (sm + imean) / 2 + 1e-6, 0.0)

    # Centroid via axis sums (O(n) memory-light)
    w = x.astype(np.float64)
    wsum = float(w.sum()) + 1e-6
    zsum = w.sum(axis=(1, 2))
    ysum = w.sum(axis=(0, 2))
    xsum = w.sum(axis=(0, 1))
    zz = np.arange(z, dtype=np.float64)
    yy = np.arange(y, dtype=np.float64)
    xxx = np.arange(xx, dtype=np.float64)
    cz_w = float((zz * zsum).sum() / wsum)
    cy_w = float((yy * ysum).sum() / wsum)
    cx_w = float((xxx * xsum).sum() / wsum)
    offset = float(
        np.sqrt(
            ((cz_w - cz) / max(z, 1)) ** 2
            + ((cy_w - cy) / max(y, 1)) ** 2
            + ((cx_w - cx) / max(xx, 1)) ** 2
        )
    )

    cv = _safe_div(float(nnz.std()), gmean, 0.0)
    center_cv = _safe_div(cstd, cmean + 1e-6, 0.0)
    shell_ratio = _safe_div(cmean, bgmean + 1e-6, 1.0)
    top_mask = x >= thr5
    top_mass_frac = _safe_div(float(x[top_mask].sum()), wsum, 0.0)
    nnz_frac = float(nnz_mask.mean())
    std = float(nnz.std())
    skew = float(((nnz - nnz.mean()) ** 3).mean() / (std**3)) if std > 0 else 0.0

    feats = np.array(
        [
            _safe_div(cmean, gmean + 1e-6, 1.0),
            shell_ratio,
            _safe_div(peak1, gmed + 1e-6, 1.0),
            _safe_div(peak5, gmed + 1e-6, 1.0),
            frac_half,
            frac_p95,
            lr_asym,
            ap_asym,
            si_asym,
            offset,
            cv,
            _safe_div(p90, p50 + 1e-6, 1.0),
            _safe_div(p99, p50 + 1e-6, 1.0),
            center_cv,
            shell_ratio,
            top_mass_frac,
            float(np.log1p(nnz_frac)),
            skew,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def load_or_build_features(force: bool = False) -> pd.DataFrame:
    labels = pd.read_csv(LABELS)
    if FEAT_CACHE.exists() and not force:
        cached = pd.read_csv(FEAT_CACHE)
        if set(cached["uid"]) == set(labels["uid"]) and all(
            c in cached.columns for c in FEATURE_NAMES
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
            # get_fdata can be slow; asanyarray keeps dtype
            arr = np.asanyarray(img.dataobj)
            feats = extract_features(arr)
        rows.append({"uid": uid, **dict(zip(FEATURE_NAMES, feats))})
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  features {i+1}/{n}  ({time.time()-t0:.1f}s)", flush=True)
    feat_df = pd.DataFrame(rows)
    ART.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEAT_CACHE, index=False)
    return feat_df.merge(labels, on="uid", how="inner")


def train(df: pd.DataFrame) -> dict:
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["is_pathologic"].values.astype(int)
    prior = float(np.clip(y.mean(), 0.01, 0.99))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float64)
    lgb_params = dict(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        cal = CalibratedClassifierCV(lgb.LGBMClassifier(**lgb_params), method="sigmoid", cv=3)
        cal.fit(X[tr], y[tr])
        p = np.clip(cal.predict_proba(X[va])[:, 1], 1e-4, 1 - 1e-4)
        oof[va] = p
        ll = log_loss(y[va], p)
        auc = roc_auc_score(y[va], p)
        fold_metrics.append({"fold": fold, "log_loss": float(ll), "auc": float(auc)})
        print(f"fold {fold}: log_loss={ll:.4f} auc={auc:.4f}", flush=True)

    cv_ll = float(log_loss(y, oof))
    cv_auc = float(roc_auc_score(y, oof))
    prior_ll = float(log_loss(y, np.full_like(oof, prior)))
    print(f"OOF log_loss={cv_ll:.4f} auc={cv_auc:.4f} | prior_ll={prior_ll:.4f}", flush=True)

    final = CalibratedClassifierCV(lgb.LGBMClassifier(**lgb_params), method="sigmoid", cv=5)
    final.fit(X, y)

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=2000, C=0.5),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )
    lr.fit(X, y)

    ASSETS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_names": FEATURE_NAMES}, ASSETS / "lgbm_calibrated.joblib")
    joblib.dump({"model": lr, "feature_names": FEATURE_NAMES}, ASSETS / "lr_calibrated.joblib")
    (ASSETS / "prior.txt").write_text(f"{prior:.6f}\n")
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_train": int(len(y)),
        "prior": prior,
        "cv_log_loss": cv_ll,
        "cv_auc": cv_auc,
        "prior_log_loss": prior_ll,
        "fold_metrics": fold_metrics,
        "primary_model": "lgbm_calibrated.joblib",
        "fallback_model": "lr_calibrated.joblib",
    }
    (ASSETS / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (ART / "cv_metrics.json").write_text(json.dumps(meta, indent=2))
    print("Saved models to", ASSETS, flush=True)
    return meta


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    # drop incomplete cache from killed run
    if FEAT_CACHE.exists():
        try:
            c = pd.read_csv(FEAT_CACHE)
            if len(c) < len(pd.read_csv(LABELS)):
                FEAT_CACHE.unlink()
                print("Removed incomplete feature cache", flush=True)
        except Exception:
            FEAT_CACHE.unlink(missing_ok=True)
    print("Building features...", flush=True)
    df = load_or_build_features(force=False)
    print(f"train rows={len(df)} pos_rate={df['is_pathologic'].mean():.4f}", flush=True)
    train(df)


if __name__ == "__main__":
    main()
