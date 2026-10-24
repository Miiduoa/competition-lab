# Kaggle Playground S6E10 — Predicting Airline Satisfaction

| 項目 | 內容 |
|------|------|
| 競賽 | [Playground Series - Season 6 Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10) |
| 題目 | 預測旅客是否滿意（`satisfaction` 機率） |
| 指標 | **ROC-AUC**（提交機率，非 0/1） |
| 截止 | **2026-10-31 23:59 UTC** |
| 提交格式 | `id,satisfaction`（與 `sample_submission.csv` 同 id 順序） |

## 方法

沿用本 repo `kaggle_s6e9` 的分層 K-fold + GBDT 融合流程，針對航空滿意度表格改特徵：

1. **前處理**：`Arrival Delay` 缺值補 `Departure Delay` 並加缺值旗標；四個類別欄（Gender / Customer Type / Type of Travel / Class）走原生 category。
2. **特徵**：評分彙總（mean/std、0/5 次數）、數位服務／機上／地面分組平均、航程與年齡的尾數／取整（合成資料常見人工痕跡）、train+test 頻次、依 Flight Distance 的路線輪廓（無標籤）、fold 內 target encoding。若有原始 [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) 則加 lookup 與 original-only teacher。
3. **模型**：LightGBM、XGBoost、CatBoost；預設 5-fold `StratifiedKFold(shuffle=True, random_state=42)`，驗證折 early stopping。
4. **融合**：等權機率平均、rank average、OOF hill-climb；依 OOF AUC 選最佳。

## 訓練

```bash
cd kaggle-s6e10
pip install -r requirements.txt
python train.py
python train.py --smoke --models lgbm --folds 2   # 合成煙測
```

## CV（本地 OOF，非 Public LB）

訓練完成後寫入 `outputs/cv_results.json`。目前分數見該檔與下方提交檔說明。

## 建議上傳順序

1. `submission_blend.csv`（OOF 最佳融合）
2. 次佳單模（見 `outputs/cv_results.json` 的 `best_single`）

本流程**不會**自動登入或繳交 Kaggle。
