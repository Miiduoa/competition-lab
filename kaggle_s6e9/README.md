# Kaggle Playground S6E9 — Predicting Electric Vehicle Purchases

| 項目 | 內容 |
|------|------|
| 競賽 | [Playground Series - Season 6 Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9) |
| 題目 | 預測潛在買家是否會購買電動車（`Will_Buy_EV` 機率） |
| 指標 | **ROC-AUC** |
| 截止 | **2026-09-30**（全程線上） |
| 參賽者 | 顧晉瑋（本目錄為可重現本地作品） |

## 題目摘要

資料靈感來自公開資料集 *EV Adoption Behavior and Range Anxiety*（特徵分布相近但不完全相同）。  
訓練集含目標欄 `Will_Buy_EV`；測試集需對每個 `id` 輸出購買機率。

提交格式：

```text
id,Will_Buy_EV
668665,0.2
668666,0.3
668667,0.2
```

官方檔案：`train.csv`、`test.csv`、`sample_submission.csv`（約 26 MB）。

## 本作品方法（Baseline）

1. **前處理**：自動分辨數值／類別欄；中位數／眾數補缺；類別 `OrdinalEncoder`。
2. **模型**（一鍵腳本預設兩者比較，取 CV 較優）：
   - `sklearn.ensemble.HistGradientBoostingClassifier`
   - `lightgbm.LGBMClassifier`
   - （可選）`RandomForestClassifier`：`--model rf`
3. **驗證**：分層 K-Fold（預設 5-fold）ROC-AUC。
4. **產出**：`submission.csv`、`outputs/cv_results.json`。

> 此為可重現 baseline；官方資料 CV≈0.9416（LGBM）。**不會**自動登入繳交 Leaderboard。

## 目錄結構

```text
kaggle_s6e9/
├── README.md                 # 本說明（繁中）
├── requirements.txt
├── train_submit.py           # 一鍵訓練＋產出 submission
├── submission.csv            # 執行後產生（可上傳）
├── data/
│   ├── README.md             # 如何取得資料
│   ├── train.csv             # 【需自行放入或 API 下載】
│   └── test.csv
├── outputs/
│   ├── submission.csv
│   └── cv_results.json
└── docs/
    └── 備審寫法.md
```

## 目前狀態（2026-09-14）

- 帳號 **kuchinwei** 已 Join；**官方資料已放入 `data/`**。
- Baseline 已跑完：最佳 **LightGBM**，5-fold CV AUC **≈ 0.94163**。
- 可上傳：`submission.csv`（尚未自動繳交 Leaderboard）。
- 詳見 `STATUS.md`。

## 如何取得資料

### 方式 A：Kaggle API（建議，需 token）

1. 登入 [Kaggle Settings → API](https://www.kaggle.com/settings) → **Create New Token**。
2. 將 `kaggle.json` 放到 `~/.kaggle/kaggle.json` 或本目錄 `./kaggle.json`。
3. 執行 `python train_submit.py`（會嘗試 `kaggle competitions download -c playground-series-s6e9`）。

### 方式 B：網頁手動下載

1. 開啟 [Data 頁](https://www.kaggle.com/competitions/playground-series-s6e9/data)。
2. 按 **Download All**，解壓 `train.csv`、`test.csv`、`sample_submission.csv` 到 `data/`。
3. 執行 `python train_submit.py`。

### 方式 C：僅驗證 pipeline（合成煙測）

```bash
python train_submit.py --smoke
```

> 無官方資料且無 API 時，腳本會自動改跑合成煙測，並標示分數不可上榜。

## 如何訓練

```bash
cd /workspace/im-admissions-116/prep/competitions/kaggle_s6e9
pip install -r requirements.txt

# 預設：HGB + LightGBM，5-fold CV，產出 submission.csv
python train_submit.py

# 只用 HistGradientBoosting
python train_submit.py --model hgb

# 只用 LightGBM
python train_submit.py --model lgbm
```

成功後會看到：

- `submission.csv`（根目錄與 `outputs/`）
- `outputs/cv_results.json`（含各 fold AUC）

## 如何上傳 Kaggle（手動）

**本流程不會用你的帳號自動登入或繳交。**

1. 確認 `submission.csv` 欄位為 `id,Will_Buy_EV`，值為機率。
2. 開啟：https://www.kaggle.com/competitions/playground-series-s6e9/submit
3. 上傳 `submission.csv`，填寫簡短說明（例如「HistGB / LightGBM baseline, 5-fold CV」）。
4. 截止前可多次提交；以 Public LB 觀察，最終以 Private 為準。

## 誠實聲明

- 目前 CV ≈ **0.9416** 為官方資料本地 5-fold 結果，**不是** Public LB 分數（尚未上傳）。
- 備審／履歷可寫「已建立可重現 LightGBM／HGB baseline 並產出 submission」；**勿虛構排行榜名次**。
- 詳見 `docs/備審寫法.md` 與 `STATUS.md`。

## 參考

- 競賽 Overview：https://www.kaggle.com/competitions/playground-series-s6e9/overview
- 靈感資料集：https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety
