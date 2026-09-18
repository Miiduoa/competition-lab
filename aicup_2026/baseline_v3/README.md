# AI CUP 2026 圍棋 — Baseline v3（本地強化版）

本地 holdout 驗證用，**不是** AIdea 排行榜分數。測試上傳預計 **2026-11-04 11:00（台灣時間）** 開放。

## 相對 v2 的改動

| 面向 | v2 | v3 |
| --- | --- | --- |
| 每盤特徵 | 74 維 + 開局 tokens | **~192 維**：joseki hash、中盤區域 n-gram、執色條件區域、接觸／提子 proxy |
| 抽樣量 | 7k／等級 | 預設 **15k／等級**（可快取 parquet） |
| Task1 | TF-IDF + dense 融合 | 雙 TF-IDF（開局＋定石／圖案）、執色條件 dense、**ANN shortlist + rerank**、gallery cleaning、權重隨機搜尋 |
| Task2 | LightGBM + HistGB | **更大 LightGBM + CatBoost + HistGB + ordinal Ridge**；溫度校準；驗證集調 blend |

## 環境

```bash
pip install pandas numpy scikit-learn lightgbm catboost pyarrow
```

## 執行

```bash
cd /workspace/im-admissions-116/prep/competitions/aicup_2026/baseline_v3
bash run_all.sh
```

快速煙測：

```bash
python3 task1_v3.py --max-games-per-rank 2000 --max-players 600 --no-tune
python3 task2_v3.py --max-games-per-rank 2500 --max-examples-per-player 2
```

## 輸出

| 檔案 | 說明 |
| --- | --- |
| `../outputs/task1_v3_metrics.json` | 本地 mean Top-5 exp-decay |
| `../outputs/task1_v3_preds_sample.csv` | 樣本 Top-5 |
| `../outputs/task2_v3_metrics.json` | majority / LGB / Cat / blend |
| `../outputs/task2_v3_preds_sample.csv` | 樣本等級預測 |
| `artifacts/features_v3_*.parquet` | 特徵快取 |
| `artifacts/task2_*_v3.*` | 模型檔 |

## 注意

- 分數皆為本地 holdout，**勿**當成官方成績。
- 正式 submission schema 待 **2026-11-04** 測試集開放後對齊。
