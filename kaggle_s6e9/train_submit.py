#!/usr/bin/env python3
"""
Kaggle Playground S6E9 — Predicting Electric Vehicle Purchases
一鍵：下載(若有金鑰) → 訓練 baseline → 交叉驗證 → 產出 submission.csv

評分指標：ROC-AUC（預測 Will_Buy_EV 機率）
用法：
  python train_submit.py              # 自動找 data/，無資料則合成煙測
  python train_submit.py --smoke      # 強制合成煙測
  python train_submit.py --model hgb  # HistGradientBoosting（預設）
  python train_submit.py --model lgbm # LightGBM
  python train_submit.py --model both # 兩者都跑，取 CV 較優者交檔
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
TARGET = "Will_Buy_EV"
ID_COL = "id"

# 靈感來源：EV Adoption Behavior and Range Anxiety（競賽合成資料欄位可能略有差異）
# 真實競賽資料載入後會自動偵測欄位，不依賴此清單硬編碼。
SYNTH_CAT = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Available",
    "Subsidy_Available",
    "Environmental_Concern_Level",
    "Range_Anxiety_Level",
]
SYNTH_NUM = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def try_kaggle_download() -> bool:
    """若有 Kaggle 憑證則下載競賽資料到 data/。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".kaggle" / "access_token",
        ROOT / "kaggle.json",
    ]
    has_env = bool(
        __import__("os").environ.get("KAGGLE_USERNAME")
        and __import__("os").environ.get("KAGGLE_KEY")
    ) or bool(__import__("os").environ.get("KAGGLE_API_TOKEN"))
    has_file = any(p.exists() for p in token_paths)
    if not (has_env or has_file):
        log("[資料] 未偵測到 Kaggle API 金鑰（~/.kaggle/kaggle.json 或環境變數）。")
        return False
    log("[資料] 偵測到 Kaggle 憑證，嘗試下載 playground-series-s6e9 …")
    try:
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            "playground-series-s6e9",
            "-p",
            str(DATA_DIR),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            log(f"[資料] kaggle download 失敗：{r.stderr.strip() or r.stdout.strip()}")
            return False
        # unzip
        for z in DATA_DIR.glob("*.zip"):
            subprocess.run(["unzip", "-o", str(z), "-d", str(DATA_DIR)], check=False)
        return (DATA_DIR / "train.csv").exists()
    except Exception as e:
        log(f"[資料] 下載例外：{e}")
        return False


def generate_synthetic(n_train: int = 4000, n_test: int = 1000, seed: int = 42) -> None:
    """依原資料集欄位風格產生合成煙測資料（非真實競賽分數）。"""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _make(n: int, start_id: int) -> pd.DataFrame:
        age = rng.integers(22, 70, n)
        gender = rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
        income = rng.normal(85000, 35000, n).clip(25000, 200000)
        city = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.45, 0.35, 0.20])
        commute = rng.uniform(5, 80, n)
        cars = rng.integers(0, 4, n)
        car_type = rng.choice(["Sedan", "SUV", "Hatchback", "Truck"], n)
        ch_home = rng.integers(0, 15, n)
        ch_work = rng.integers(0, 20, n)
        home_chg = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
        subsidy = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
        env = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.45, 0.30])
        anxiety = rng.choice(["Low", "Medium", "High"], n, p=[0.30, 0.40, 0.30])

        # 簡易生成規則：環保高、焦慮低、充電多 → 較可能買 EV
        score = (
            (env == "High").astype(float) * 1.2
            + (env == "Medium").astype(float) * 0.4
            + (anxiety == "Low").astype(float) * 1.0
            + (anxiety == "Medium").astype(float) * 0.3
            + (home_chg == "Yes").astype(float) * 0.8
            + (subsidy == "Yes").astype(float) * 0.7
            + (ch_home + ch_work) / 20.0
            + (city == "Urban").astype(float) * 0.3
            - commute / 100.0
            + (income - 85000) / 100000.0
            + rng.normal(0, 0.8, n)
        )
        # 約 2% 缺值（仿原資料）
        income = income.astype(float)
        commute = commute.astype(float)
        miss_i = rng.random(n) < 0.02
        miss_c = rng.random(n) < 0.02
        income[miss_i] = np.nan
        commute[miss_c] = np.nan

        df = pd.DataFrame(
            {
                ID_COL: np.arange(start_id, start_id + n),
                "Age": age,
                "Gender": gender,
                "Annual_Income_USD": income,
                "City_Type": city,
                "Daily_Commute_km": commute,
                "Number_of_Cars_Owned": cars,
                "Current_Car_Type": car_type,
                "Charging_Stations_Near_Home": ch_home,
                "Charging_Stations_Near_Work": ch_work,
                "Home_Charging_Available": home_chg,
                "Subsidy_Available": subsidy,
                "Environmental_Concern_Level": env,
                "Range_Anxiety_Level": anxiety,
            }
        )
        # 額外噪音欄（逼近競賽「約 31 欄」規模，僅煙測用）
        for i in range(1, 16):
            df[f"noise_num_{i}"] = rng.normal(0, 1, n)
        for i in range(1, 4):
            df[f"noise_cat_{i}"] = rng.choice(["A", "B", "C"], n)
        prob = 1 / (1 + np.exp(-score))
        df[TARGET] = (rng.random(n) < prob).astype(int)
        return df

    train = _make(n_train, 0)
    test = _make(n_test, n_train)
    test_ids = test[ID_COL].copy()
    y_test_hold = test[TARGET].copy()  # 僅煙測可算假 AUC；不寫入 test.csv
    test = test.drop(columns=[TARGET])

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET: 0.5}).to_csv(
        DATA_DIR / "sample_submission.csv", index=False
    )
    # 內部煙測標籤（不上傳）
    pd.DataFrame({ID_COL: test_ids, TARGET: y_test_hold}).to_csv(
        DATA_DIR / "_synthetic_test_labels.csv", index=False
    )
    meta = {
        "synthetic": True,
        "source": "synthetic_smoke",
        "note": "合成煙測資料，非 Kaggle 真實分數。請放置官方 train.csv/test.csv。",
        "n_train": n_train,
        "n_test": n_test,
        "columns_train": list(train.columns),
    }
    (DATA_DIR / "_data_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[資料] 已寫入合成煙測 train={n_train}, test={n_test} → {DATA_DIR}")


def ensure_data(force_smoke: bool) -> dict:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    meta = {"synthetic": False, "source": "official"}

    if force_smoke:
        generate_synthetic()
        meta = json.loads((DATA_DIR / "_data_meta.json").read_text(encoding="utf-8"))
        return meta

    if train_path.exists() and test_path.exists():
        # 判斷是否為先前合成
        if (DATA_DIR / "_data_meta.json").exists():
            meta = json.loads((DATA_DIR / "_data_meta.json").read_text(encoding="utf-8"))
        log(f"[資料] 使用既有檔案：{train_path}, {test_path}")
        return meta

    if try_kaggle_download() and train_path.exists():
        meta = {"synthetic": False, "source": "kaggle_api"}
        (DATA_DIR / "_data_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    log(
        "[資料] 無官方資料且無 API 金鑰 → 改跑合成煙測。"
        " 請自行下載後放到 data/train.csv 與 data/test.csv，"
        " 或提供 kaggle.json 後重跑。"
    )
    generate_synthetic()
    meta = json.loads((DATA_DIR / "_data_meta.json").read_text(encoding="utf-8"))
    return meta


def encode_target(y: pd.Series) -> np.ndarray:
    # 官方資料可能是 Yes/No 字串（含 pandas StringDtype / Arrow str）
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    lowered = y.astype(str).str.strip().str.lower()
    uniq = set(lowered.dropna().unique())
    if uniq and uniq <= set(mapping):
        return lowered.map(mapping).astype(int).values
    if y.dtype == object or str(y.dtype).startswith("string") or str(y.dtype) == "str":
        le = LabelEncoder()
        return le.fit_transform(y.astype(str))
    return y.astype(int).values


def split_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    cat, num = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            # 整數且基數小 → 可能是類別，但仍可當數值給樹模型
            num.append(c)
        else:
            cat.append(c)
    return cat, num


def make_hgb(cat: list[str], num: list[str]) -> Pipeline:
    # OrdinalEncoder + HGB；類別缺值用 "missing"
    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat,
            ),
            (
                "num",
                SimpleImputer(strategy="median"),
                num,
            ),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_rf(cat: list[str], num: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat,
            ),
            ("num", SimpleImputer(strategy="median"), num),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_lgbm(cat: list[str], num: list[str]):
    import lightgbm as lgb

    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat,
            ),
            ("num", SimpleImputer(strategy="median"), num),
        ],
        remainder="drop",
    )
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def cross_val_auc(
    model_factory, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5
) -> tuple[float, float, list[float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = model_factory()
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y[va], proba)
        scores.append(float(auc))
        log(f"  fold {fold}: AUC={auc:.5f}")
    return float(np.mean(scores)), float(np.std(scores)), scores


def run(model_name: str, force_smoke: bool, n_splits: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = ensure_data(force_smoke)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log(f"[資料] train={train.shape}, test={test.shape}, columns={list(train.columns)}")

    if TARGET not in train.columns:
        raise SystemExit(f"train.csv 缺少目標欄 {TARGET}")
    if ID_COL not in train.columns:
        # 相容 Buyer_ID
        if "Buyer_ID" in train.columns:
            train = train.rename(columns={"Buyer_ID": ID_COL})
        else:
            train[ID_COL] = np.arange(len(train))
    if ID_COL not in test.columns:
        if "Buyer_ID" in test.columns:
            test = test.rename(columns={"Buyer_ID": ID_COL})
        else:
            test[ID_COL] = np.arange(len(train), len(train) + len(test))

    y = encode_target(train[TARGET])
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    # 對齊 test 欄位
    for c in feature_cols:
        if c not in test.columns:
            log(f"[警告] test 缺少欄位 {c}，填 NA")
            test[c] = np.nan
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    cat, num = split_features(pd.concat([X, X_test], axis=0))
    log(f"[特徵] 類別={len(cat)}, 數值={len(num)}")

    factories = {}
    if model_name in ("hgb", "both"):
        factories["hgb"] = lambda: make_hgb(cat, num)
    if model_name in ("lgbm", "both"):
        try:
            import lightgbm  # noqa: F401

            factories["lgbm"] = lambda: make_lgbm(cat, num)
        except ImportError:
            log("[模型] LightGBM 未安裝，略過；改用 HGB。")
            factories["hgb"] = lambda: make_hgb(cat, num)
    if model_name == "rf":
        factories["rf"] = lambda: make_rf(cat, num)
    if not factories:
        factories["hgb"] = lambda: make_hgb(cat, num)

    results = {}
    best_name, best_mean = None, -1.0
    for name, factory in factories.items():
        log(f"\n[CV] 模型={name}, folds={n_splits}")
        mean_auc, std_auc, fold_scores = cross_val_auc(factory, X, y, n_splits=n_splits)
        log(f"[CV] {name} mean AUC={mean_auc:.5f} ± {std_auc:.5f}")
        results[name] = {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "fold_scores": fold_scores,
        }
        if mean_auc > best_mean:
            best_mean, best_name = mean_auc, name

    log(f"\n[訓練] 以全量資料重訓最佳模型：{best_name}")
    final = factories[best_name]()
    final.fit(X, y)
    proba = final.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: proba})
    sub_path = OUT_DIR / "submission.csv"
    # 同時放一份在根目錄方便上傳
    sub.to_csv(sub_path, index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)
    log(f"[產出] {sub_path}  rows={len(sub)}")

    smoke_auc = None
    labels_path = DATA_DIR / "_synthetic_test_labels.csv"
    if meta.get("synthetic") and labels_path.exists():
        lab = pd.read_csv(labels_path)
        merged = sub.merge(lab, on=ID_COL, suffixes=("_pred", "_true"))
        smoke_auc = float(
            roc_auc_score(merged[f"{TARGET}_true"], merged[f"{TARGET}_pred"])
        )
        log(f"[煙測] 合成 test AUC（僅供驗證 pipeline）= {smoke_auc:.5f}")

    summary = {
        "competition": "playground-series-s6e9",
        "metric": "ROC-AUC",
        "synthetic_data": bool(meta.get("synthetic")),
        "data_source": meta.get("source", "unknown"),
        "best_model": best_name,
        "cv": results,
        "cv_best_mean_auc": best_mean,
        "smoke_test_auc": smoke_auc,
        "submission": str(sub_path),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(feature_cols),
        "note_zh": (
            "目前為合成煙測，CV/煙測 AUC 不可代表 Kaggle 排行榜。"
            if meta.get("synthetic")
            else "已使用本地官方資料完成 baseline。"
        ),
    }
    (OUT_DIR / "cv_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[產出] {OUT_DIR / 'cv_results.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="S6E9 EV Purchase baseline")
    p.add_argument(
        "--model",
        choices=["hgb", "lgbm", "rf", "both"],
        default="both",
        help="baseline 模型（預設 both：HGB + LightGBM 取優）",
    )
    p.add_argument("--smoke", action="store_true", help="強制合成煙測")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    summary = run(args.model, args.smoke, args.folds)
    log("\n===== 摘要 =====")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["synthetic_data"]:
        log(
            "\n【下一步】請將 Kaggle 官方 train.csv / test.csv 放入 data/，"
            "或設定 kaggle.json 後重跑：python train_submit.py"
        )
        log(
            "取得金鑰：https://www.kaggle.com/settings → API → Create New Token"
            " → 存成 ~/.kaggle/kaggle.json"
        )
    else:
        log(
            "\n【上傳】至 https://www.kaggle.com/competitions/playground-series-s6e9/submit"
            f" 上傳 {ROOT / 'submission.csv'}"
        )


if __name__ == "__main__":
    main()
