# AI CUP 2026 v4 改進說明（本地驗證）

更新：2026-09-18（台灣時間 UTC+8）  
**以下分數皆為本地 holdout，非 AIdea 排行榜。**  
測試上傳預計 **2026-11-04 11:00（台灣時間）** 開放。

## 做了什麼

路徑：`baseline_v4/`

### Task2（棋力預測）— 重點

1. **更多資料**：每等級 **25,000** 盤（v3 為 20,000）→ 250,000 盤；約 22,845 個 5 盤 query-set。
2. **Set 聚合微增**：組內 pair-cosine、執色／長度等額外標量（dense 聚合約 1554 維）。
3. **開局／pattern TF-IDF → TruncatedSVD（48 維）**：僅在 **train 玩家** 上 fit，避免洩漏。
4. **±1 感知**：LightGBM soft-neighbor 標籤擴增（鄰級權重 0.22）+ 序數回歸（LGB reg → Gaussian soft probs）。
5. **多模型**：LGB / LGB-soft / LGB-reg / **CatBoost** / XGBoost / Ridge；valid 調溫度與 blend。
6. **Stage B refit**：超參／溫度／權重在 train／valid 上決定後，於 train+valid（留一小 ES）重訓再測 test（仍無 test 洩漏）。
7. **棄用**：train-gallery prototype／kNN 特徵（ablation 顯示會讓 LGBM `best_iter≈9`、分數崩到 ~0.25）。HistGB 在高維上過慢已跳過。

Task1：**未改動**（維持 v3 本地 ≈0.2112）。

## 本地指標對照（非排行榜）

| 設定 | mean score | exact | ±1 | 備註 |
| --- | --- | --- | --- | --- |
| v3 blend（20k） | **0.3329** | 0.219 | 0.529 | 先前 primary |
| v4 CatBoost expected（25k+TF-IDF+refit） | **0.3730** | 0.256 | 0.574 | **本版 primary** |
| v4 blend refit | **0.3700** | 0.249 | 0.578 | 更穩健的備選 |
| v4 LGB / soft / XGB | 0.355 / 0.358 / 0.360 | — | — | 皆已過 0.35 |
| 目標 | ≥0.35 | — | — | **已達成（本地）** |

輸出：

- `outputs/task2_v4_metrics.json`、`task2_v4_preds_sample.csv`
- `baseline_v4/artifacts/`（25k 特徵快取與模型）

## 誠實說明

1. 分數為 **玩家分組 holdout**，**不是** AIdea leaderboard。
2. Valid 上 blend 曾見 ≈0.366；test primary CatBoost ≈0.373 — 與 v3「valid 虚高」不同，此處 test 不差於 valid。
3. 特徵仍是 SGF 統計 + token TF-IDF，**無**完整形勢／死活／目數；再往上可能需要棋盤 CNN／對比學習／更強序數損失。
4. 下一步天花板：輕量棋盤 CNN（CPU 可跑的薄網路）、真正 soft ordinal CE、OOF stacking、或等 11/04 測試協定對齊後做協定匹配。

## 重跑

```bash
cd baseline_v4
PYTHONUNBUFFERED=1 python3 task2_v4.py --reuse-cache 25000 --max-games-per-rank 25000
```

**再次強調：上表非官方成績。**
