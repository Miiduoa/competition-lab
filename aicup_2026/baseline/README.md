# AI CUP 2026 圍棋競賽 — 輕量 Baseline

本地驗證用 baseline（**非**官方排行榜分數）。測試上傳開放日：**2026-11-04 11:00（台灣時間）**。

## 資料配置

```
aicup_2026/
├── data/
│   ├── aicup2026_go_training.tar.xz   # 官方壓縮包（~215MB）
│   ├── task1/train_{D,C,B,A,1D…6D}.csv
│   └── task2/…                       # 與 task1 相同內容
├── baseline/                         # 本目錄
└── outputs/                          # 指標與樣本預測輸出
```

官方資料集：https://github.com/AILAB-NDHU/aicup2026-go-dataset  
SHA256：`959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b`

欄位：`player_id, game_id, rank, color, sgf_content`（每等級 100,000 盤）。

## 環境

```bash
pip install pandas numpy scikit-learn
```

建議在 `baseline/` 目錄執行（`features.py` 以相對 import）。

## 一鍵執行

```bash
cd /workspace/im-admissions-116/prep/competitions/aicup_2026/baseline
bash run_all.sh
```

或分開跑：

```bash
python3 task1_baseline.py
python3 task2_baseline.py
```

常用參數（可調小以求更快）：

```bash
python3 task1_baseline.py --max-games-per-rank 2000 --max-players 800 --query-k 3
python3 task2_baseline.py --max-games-per-rank 3000 --games-per-example 5
```

## Baseline 做什麼

### Task1 — 辨識棋士（Top-5 exponential decay）

1. 每個等級串流讀 CSV，抽樣約 ≤3000 盤（預設）。
2. 篩選有足夠對局的玩家（預設 ≥5 盤），最多 1500 人。
3. 每人 hold-out `K=3` 盤當 query set，其餘當 gallery。
4. 玩家 profile：棋長 histogram、開局 fingerprint 頻率、執黑比例、首著座標。
5. Query 與候選做加權相似度，排出 Top-5；分數為 \(e^{-(k-1)}\)（命中第 k 名）。

### Task2 — 預測棋力（exact=1，±1=1/e）

1. 同樣抽樣後，把同一玩家的若干盤聚合成一個「query set」特徵。
2. **多數決 baseline**：訓練集最常見等級。
3. **LogisticRegression**：平均/標準差棋長、length-bin hist、opening hash buckets、執黑比例等。
4. 以玩家為單位切 train/test，降低洩漏。

## 輸出

| 檔案 | 說明 |
| --- | --- |
| `../outputs/task1_baseline_metrics.json` | 本地 mean Top-5 分數等 |
| `../outputs/task1_baseline_preds_sample.csv` | 樣本 Top-5 預測 |
| `../outputs/task2_baseline_metrics.json` | majority / logistic 分數 |
| `../outputs/task2_baseline_preds_sample.csv` | 樣本等級預測 |

## 限制

- **積極抽樣**：未用滿 100 萬盤；數字只反映本地 holdout。
- 特徵極簡（棋長 + 開局字串），沒有盤面／目數／吃子等。
- Task1 候選池 = 抽樣後的 gallery 玩家，與正式測試「開放候選」設定可能不同。
- 正式答案檔格式在測試開放前可能未公開；目前輸出為本地驗證格式。
- 請勿把本地分數當成 AIdea 排行榜成績回報。

## 下一步

1. 等 **2026-11-04** 測試資料與上傳格式。
2. 強化特徵（SGF 座標統計、常見定石、執色條件開局）。
3. Task1 可改用 embedding / ANN；Task2 可改用有序迴歸（ordinal）。
