# AI CUP 2026 Autumn (AIdea) — 進度狀態

更新：2026-09-18 12:30（台灣時間，UTC+8）

隊伍：Task1 `TEAM_10925`｜Task2 `TEAM_10926`  
報名 Email：demohan513@gmail.com

---

## 資料下載 — 完成

來源：[AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset) Releases

| 檔案 | 大小 | SHA256 |
| --- | --- | --- |
| `data/aicup2026_go_training.tar.xz` | 215 MB（224,515,476 bytes） | `959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b` |

解壓後（`data/task1/`，`task2/` 為相同內容／硬連結）：

| 檔案 | 約略大小 |
| --- | --- |
| `train_D.csv` … `train_A.csv` | 各 ~133–135 MB |
| `train_1D.csv` … `train_6D.csv` | 各 ~127–133 MB |
| 合計 | 約 1.3 GB（10 × 100,000 列） |

欄位：`player_id, game_id, rank, color, sgf_content`  
等級序（弱→強）：`D C B A 1D 2D 3D 4D 5D 6D`

---

## Baseline — 已完成（本地驗證）

路徑：`baseline/`（見 `baseline/README.md`）

| 腳本 | 說明 |
| --- | --- |
| `features.py` | SGF 棋長／開局 fingerprint／串流讀 CSV |
| `task1_baseline.py` | 玩家 profile + Top-5 檢索 |
| `task2_baseline.py` | 多數決 + LogisticRegression |
| `run_all.sh` | 一鍵跑兩邊 |

### Task1 本地 holdout（非排行榜）

- 抽樣：每等級 2500 盤 → 25,000 盤／5,303 玩家；holdout 1,200 玩家、K=3
- **mean Top-5 exp-decay ≈ 0.0220**
- hit@1 ≈ 0.015｜hit@5 ≈ 0.051
- 輸出：`outputs/task1_baseline_metrics.json`、`outputs/task1_baseline_preds_sample.csv`
- 耗時 ≈ 20s

### Task2 本地 holdout（非排行榜）

- 抽樣：每等級 3500 盤 → 35,000 盤；3,417 個 query-set 例（每例 5 盤）
- **majority mean ≈ 0.1919**（exact≈0.126；mode=`B`）
- **logistic mean ≈ 0.2213**（exact≈0.139，±1≈0.364）
- 輸出：`outputs/task2_baseline_metrics.json`、`outputs/task2_baseline_preds_sample.csv`
- 耗時 ≈ 8s

重跑：

```bash
cd baseline && bash run_all.sh
# 或
python3 task1_baseline.py && python3 task2_baseline.py
```

---



## Baseline v2 — 已完成（本地驗證，非排行榜）

路徑：`baseline_v2/`（見 `baseline_v2/README.md`、`IMPROVE_v2.md`）

| 腳本 | 說明 |
| --- | --- |
| `features_v2.py` | 較豐富 SGF 特徵（區域／delta／開局 tokens） |
| `task1_v2.py` | TF-IDF + dense cosine + Jaccard 融合 |
| `task2_v2.py` | LightGBM／HistGB + 期望分數校準 |
| `run_all.sh` | 一鍵跑兩邊 |

### Task1 v2 本地 holdout（**非排行榜**）

- 主設定：每等級 7000 盤 → 70,000 盤；holdout 2,000 玩家、K=3
- **mean Top-5 exp-decay ≈ 0.0868**（hit@1≈0.070｜hit@5≈0.161）
- 對齊 baseline 協定（2500／級、1200 人）：**≈ 0.1108**（對照 baseline ≈0.0220）
- 輸出：`outputs/task1_v2_metrics.json`、`task1_v2_preds_sample.csv`、`task1_v2_matched_protocol_metrics.json`
- 耗時 ≈ 2 分鐘（主設定）

### Task2 v2 本地 holdout（**非排行榜**）

- 每等級 7000 盤；7,254 個 query-set 例；玩家 group split
- **LightGBM 期望分數 mean ≈ 0.2972**（exact≈0.202；±1≈0.461）
- HistGB 期望分數 ≈ 0.285；majority ≈ 0.202
- 對照 baseline logistic ≈ **0.2213** → 明顯提升
- 輸出：`outputs/task2_v2_metrics.json`、`task2_v2_preds_sample.csv`；模型 `baseline_v2/artifacts/task2_lgbm.txt`
- 耗時 ≈ 1.5 分鐘

重跑：

```bash
cd baseline_v2 && bash run_all.sh
```

## AIdea Data 分頁（登入後，2026-09-18）

登入帳號 `demohan513` 後，Task1／Task2 的 **Data** 分頁已開通：

- **Stage 1**：訓練集 + 範例程式可下載（頁面另導向 GitHub release `aicup2026_go_training.tar.xz`；範例程式 zip：`docs/AIcupTutorial-main.zip`／`AIcupTutorial-main_task1_task2_20260918.zip`）
- **Stage 2（測試集）**：頁面原文提示約 *File will be ready to download from 2026-11-03 19:00:00 till 2026-11-24 07:59:59.*（對應測試窗口；正式上傳以簡章 11/04 11:00 為準）
- 截圖：`outputs/task1_data_loggedin_20260918.png`、`outputs/task2_data_loggedin_20260918.png`

## 賽程重點

| 項目 | 時間（台灣） |
| --- | --- |
| 訓練資料／模型窗口 | 2026-09-18 11:00 – 2026-11-24 23:59:59 |
| **測試資料／上傳開放** | **2026-11-04 11:00 – 2026-11-24 23:59:59** |
| 報名截止 | 2026-11-03 23:59:59 |
| 成績公布 | 2026-11-25 16:00 |
| 每日上傳上限 | 5 次（以最後一次計分） |

AIdea：
- Task1：https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688
- Task2：https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403

---

## 下一步

1. **等 2026-11-04** 測試集與官方答案檔格式；對齊 submission schema 後再上傳。
2. v2 已落地（TF-IDF／稠密特徵／LightGBM）；見 `IMPROVE_v2.md`。
3. Task1：FAISS／對比學習、執色條件開局；Task2：ordinal／更多盤。
4. 對齊 2026-11-04 官方 submission schema（教學 zip 為往年格式）。
5. 保留每日 ≤5 次上傳額度；本地 CV 穩定後再送測。

**注意：** 上列分數皆為本地 holdout，**不是** AIdea leaderboard。
