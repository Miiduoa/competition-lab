# AI CUP 2026 v2 改進說明（本地驗證）

更新：2026-09-18（台灣時間 UTC+8）  
**以下分數皆為本地 holdout，非 AIdea 排行榜。**

## 做了什麼

路徑：`baseline_v2/`（延伸 `baseline/features.py` 模式，未整包重寫）。

### 共同特徵（`features_v2.py`）

- 開局 **12 手** fingerprint + **8 手**；move／bigram **tokens**
- 盤面 **區域分布**（四角／四邊／中央）、早期 20 手 **3×3 熱度**
- 連續落子 **曼哈頓距離 histogram**、pass rate、執色
- 開局區域 n-gram **hash bag**（固定稠密向量 74 維）
- 仍串流讀 CSV，預設每等級抽樣 7,000 盤

### Task1（玩家辨識）

- Gallery profile：稠密向量均值 + 開局 Counter + TF-IDF 文件
- 相似度融合：dense cosine (0.28) + **TF-IDF cosine (0.32)** + opening Jaccard 等
- 玩家級 holdout（query K=3），無洩漏

### Task2（棋力預測）

- Set 聚合：dense 的 mean／std／q25／q75 + 集合大小等（約 301 維）
- **LightGBM** multiclass + HistGB 對照
- 以機率計算 **期望分數**（exact=1、±1=1/e）再選類別
- 玩家 group split

## 本地指標對照

| 任務 | Baseline（本地） | v2（本地） | 備註 |
| --- | --- | --- | --- |
| Task1 mean Top-5 | **≈0.0220**（1200 候選） | **≈0.0868**（2000 候選，7k／級） | 明顯提升；候選變多更難 |
| Task1 對齊協定 | 同上 | **≈0.1108**（2500／級、1200 人） | 與 baseline 抽樣對齊，約 **5×** |
| Task2 logistic | **≈0.2213** | — | — |
| Task2 v2 primary | — | **≈0.2972**（LGB 期望分數） | exact≈0.202，±1≈0.461 |

輸出：

- `outputs/task1_v2_metrics.json`、`task1_v2_preds_sample.csv`
- `outputs/task1_v2_matched_protocol_metrics.json`（對齊協定）
- `outputs/task2_v2_metrics.json`、`task2_v2_preds_sample.csv`
- `baseline_v2/artifacts/task2_lgbm.txt`

## 提交格式備註

`docs/AIcupTutorial-main.zip` 為**往年**下一手預測教學（Dan／Kyu／PlayStyle），模板 CSV 非 2026 Task1／Task2 官方格式。正式 submission schema 待 **2026-11-04** 測試集開放後對齊。

## 下一步想法

1. Task1：更大 gallery、FAISS／ANN、執色條件開局、中盤定石 n-gram、對每位玩家多 query 平均
2. Task1：學習 metric（對比學習／Siamese）取代手調權重
3. Task2：ordinal／CORAL、更多盤／例、棋長與開局交互特徵
4. 兩者：輕量 CNN／圖特徵於前 N 手（CPU 可承受範圍）
5. 等測試集後對齊上傳 CSV；每日 ≤5 次額度，本地 CV 穩定再送

**再次強調：上表非官方成績。**
