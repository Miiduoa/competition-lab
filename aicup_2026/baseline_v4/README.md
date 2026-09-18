# AI CUP 2026 Go — Baseline v4

本地 holdout 驗證用（**非** AIdea 排行榜）。測試上傳預計 **2026-11-04 11:00（台灣時間）**。

## 內容

| 檔案 | 說明 |
| --- | --- |
| `features_v4.py` | 重用 v3 稠密特徵抽取 |
| `task2_v4.py` | Task2：TF-IDF+SVD + soft/ordinal + CatBoost/XGB blend + refit |
| `run_task2.sh` | 一鍵跑 Task2 |

## Task2 結果摘要（本地，非排行榜）

- 資料：25k／等級；玩家 train／valid／test
- **Primary（CatBoost expected-score）≈ 0.3730**（exact≈0.256，±1≈0.574）
- Blend refit ≈ 0.3700；對照 v3 blend ≈ **0.3329**；目標 ≥0.35 **已達**

```bash
cd baseline_v4
PYTHONUNBUFFERED=1 python3 task2_v4.py --reuse-cache 25000
```

輸出：`../outputs/task2_v4_metrics.json`、`task2_v4_preds_sample.csv`
