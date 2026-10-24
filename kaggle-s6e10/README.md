# Kaggle Playground S6E10 — Predicting Airline Satisfaction

| 項目 | 內容 |
|------|------|
| 競賽 | [Playground Series - Season 6 Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10) |
| 題目 | 預測旅客是否滿意（`satisfaction` 機率） |
| 指標 | **ROC-AUC**（提交機率，非 0/1） |
| 截止 | **2026-10-31 23:59 UTC** |
| 提交格式 | `id,satisfaction`（與 `sample_submission.csv` 同 id 順序、同列數） |

## 方法

沿用本 repo `kaggle_s6e9` 的分層 K-fold + GBDT 融合流程：

1. **前處理**：`Arrival Delay` 缺值補 `Departure Delay` 並加缺值旗標；四個類別欄走原生 category。
2. **特徵**：評分彙總與 0/5 次數、數位／機上／地面分組平均、航程與年齡尾數／取整、train+test 頻次、依 Flight Distance 的無標籤路線輪廓、fold 內 target encoding。另用原始 [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) 做 lookup 與 original-only teacher。
3. **模型**：LightGBM、XGBoost、CatBoost；`StratifiedKFold(5, shuffle=True, random_state=42)`，驗證折 early stopping。
4. **融合**：等權機率、rank average、OOF hill-climb；依 OOF AUC 選最佳。

## 本地 5-fold OOF（非 Public LB）

| 模型 | OOF AUC |
|------|---------|
| LightGBM | 0.96089 |
| XGBoost | 0.96093 |
| CatBoost | 0.96083 |
| equal-prob blend | 0.96105 |
| hill-climb blend | 0.96105 |
| **rank-average blend** | **0.96106** |

Hill-climb 權重約 LGBM 0.31 / XGB 0.35 / CatBoost 0.35。訓練約 37 分鐘（4 CPU）。

## 建議上傳順序

1. **`submission_blend.csv`**（或同內容的 `submission.csv`）— rank-average，OOF 最高，請先交這份
2. `submission_xgboost.csv` — 最優單模
3. `submission_lgbm.csv` — 次優單模

三份皆 299844 列，欄位 `id,satisfaction`，id 與官方 sample 一致。blend 為 rank-space 分數（AUC 只看排序）。

本流程不會自動登入或繳交 Kaggle。

## 訓練

```bash
cd kaggle-s6e10
pip install -r requirements.txt
# 將 train.csv / test.csv / sample_submission.csv 放入 data/
python train.py
python train.py --smoke --models lgbm --folds 2   # 合成煙測，不覆蓋正式提交檔
```
