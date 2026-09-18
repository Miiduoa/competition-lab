# AI CUP 2026 圍棋 — Baseline v2（本地強化版）

本地 holdout 驗證用，**不是** AIdea 排行榜分數。測試上傳預計 **2026-11-04 11:00（台灣時間）** 開放。

## 相對 v1（`../baseline/`）的改動

| 面向 | v1 | v2 |
| --- | --- | --- |
| 每盤特徵 | 棋長 bin、開局 8 手字串、首著 | 開局 12 手 + token／bigram、區域分布、走子距離 hist、早期 3×3 熱度、pass rate 等 |
| Task1 檢索 | Jaccard + 長度 cosine | **TF-IDF（開局 tokens）+ 稠密向量 cosine + Jaccard** 加權融合 |
| Task2 模型 | LogisticRegression | **LightGBM multiclass** + HistGB；以 **期望分數**（exact / ±1）選類別 |
| 抽樣量 | ~2.5k–3.5k／等級 | 預設 **7k／等級**（可調） |
| Holdout | 玩家級切分 | 同；Task1 `min_games=6`，Task2 玩家 group split |

## 環境

```bash
pip install pandas numpy scikit-learn lightgbm
```

## 執行

```bash
cd /workspace/im-admissions-116/prep/competitions/aicup_2026/baseline_v2
bash run_all.sh
# 或
python3 task1_v2.py
python3 task2_v2.py
```

快速煙測：

```bash
python3 task1_v2.py --max-games-per-rank 2000 --max-players 600
python3 task2_v2.py --max-games-per-rank 2500 --max-examples-per-player 2
```

## 輸出

| 檔案 | 說明 |
| --- | --- |
| `../outputs/task1_v2_metrics.json` | 本地 mean Top-5 exp-decay |
| `../outputs/task1_v2_preds_sample.csv` | 樣本 Top-5 |
| `../outputs/task2_v2_metrics.json` | majority / LGB / HistGB |
| `../outputs/task2_v2_preds_sample.csv` | 樣本等級預測 |
| `artifacts/task2_lgbm.txt` | LightGBM 模型檔 |

## 提交格式備註

`docs/AIcupTutorial-main.zip` 為**往年** Dan／Kyu／PlayStyle 教學（下一手預測 CSV），**不是** 2026 Task1／Task2 官方答案檔格式。2026 submission schema 待 **2026-11-04** 測試集開放後再對齊。

## 注意

- 分數皆為本地 holdout，**勿**當成官方成績。
- 特徵仍為 SGF 統計 proxy，未含完整目數／死活引擎。
