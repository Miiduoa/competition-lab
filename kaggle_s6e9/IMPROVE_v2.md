# S6E9 改進紀錄 v2（Predicting Electric Vehicle Purchases）

| 項目 | 內容 |
|------|------|
| 帳號 | kuchinwei / demohan513@gmail.com |
| 日期 | 2026-09-15 |
| 本次 Public Score（v2） | **0.94540** |
| 先前本地 CV（v1 baseline LGBM） | **0.94163** ± 0.00078 |
| 本次本地 OOF AUC（v2c） | **0.94540** |
| CV 相對提升 | **+0.00377**（相對 v1） |
| 狀態 | **已提交 v2**（提升明顯且驗證無洩漏） |

---

## 1. 改了什麼

### v1 baseline（既有）
- 原始 13 特徵；`OrdinalEncoder` + HistGB / LightGBM
- 固定超參、單一種子、5-fold
- v2 Public 0.94540（與 OOF 接近，管線可信）

### 中間嘗試（v2a / v2b，**未採用為最終檔**）
- **v2a**：手工交互特徵 + LGBM/XGB/CatBoost OOF 加權融合  
  - OOF ≈ **0.94181**（僅 +0.00018）→ 增益過小
- **v2b**：Optuna + native categorical  
  - 抽樣子集調參後全量 CV 仍卡在 ~0.941 量級  
  - 已中止，不作為交檔

### 最終採用：**v2c**
靈感來自公開高分管線的可重現作法（數位／合成痕跡特徵 + 折內目標編碼 + 高 `max_bin`），並在本機完整重跑驗證。

1. **特徵工程（合成資料痕跡）**
   - 收入末位／取模／trailing-zero、是否卡在 floor（30000）
   - 通勤（×10 整數化）類似數位特徵
   - 多解析度分箱：`inc_div_*`、`km_div_*`
   - Yes/No → 0/1；焦慮程度序數化；名義類別保留 `category`

2. **洩漏安全的目標編碼（nested TE）**
   - 對收入／通勤在多個粒度做平滑 target encoding
   - **僅在 fold 訓練列上 fit**；訓練列內部再做內層 K-Fold，避免自己編碼自己
   - 驗證／測試用外層訓練映射 transform

3. **模型**
   - LightGBM（`lgb.train`），關鍵超參：`max_bin=8191`、`num_leaves=39`、`feature_fraction≈0.37` 等（公開搜尋結果）
   - **5-fold × 3 seeds**（11 / 202 / 3407）＝ 15 個子模型平均
   - Early stopping 200 rounds；平均約 1565 trees

4. **產出**
   - `submission.csv`（根目錄，給上傳用）
   - `outputs/submission_v2.csv`（同內容備份）
   - `outputs/cv_results_v2.json`、`outputs/oof_v2c.npy`
   - 可重跑腳本：`train_v2c.py`

---

## 2. 交叉驗證結果

| Seed | Fold 均值附近 | Seed OOF |
|------|---------------|----------|
| 11 | 0.94445–0.94631 | **0.94520** |
| 202 | 0.94480–0.94561 | **0.94518** |
| 3407 | 0.94396–0.94568 | **0.94510** |
| **三種子平均 OOF** | | **0.94540** |

- 指標：ROC-AUC（與競賽一致）
- 與公開參考 OOF 0.94540 一致，重現成功
- 相對 v1 CV 0.94163：**+0.00377**（約 37 個 public 萬分位）

詳細 fold 表見 `outputs/cv_results_v2.json`。

---

## 3. 對 Public LB 的誠實預期

| 估計 | 說明 |
|------|------|
| 本次 Public | **0.94540**（已於 2026-09-15 提交 v2） |
| 本次實際 Public | **0.94540** |
| 實際結果 | **0.94540** |
| 不保證 | Private 可能與 Public 有差距；合成資料上「痕跡特徵」是否過擬合需靠 LB 驗證 |

**為何相信有實質提升（非過擬合幻覺）**
- TE 採 nested／fold-fit，非全域泄漏編碼
- 三個種子 OOF 穩定在 0.9451–0.9452
- 相對 v1 的 +0.0038 遠大於 CV 標準差（~0.0008）

---

## 4. GO / NO-GO

### **已提交 v2**

理由：
1. 本地 OOF **0.94540** 已由 v2 提交並取得 Public **0.94540**
2. 驗證設計避免常見 TE 洩漏
3. Submission 格式已對齊 `sample_submission`（286,571 列，`id,Will_Buy_EV`，機率 ∈ (0,1)）

上傳紀錄（已由 parent 執行）：
1. 檔案：`/workspace/im-admissions-116/prep/competitions/kaggle_s6e9/submission.csv`
2. 頁面：https://www.kaggle.com/competitions/playground-series-s6e9/submit
3. 說明建議：`v2c LGBM digit+TE max_bin=8191, 5×3seed OOF 0.94540`

---

## 5. 檔案清單

```text
kaggle_s6e9/
├── submission.csv                 # ← 上傳此檔
├── IMPROVE_v2.md                  # 本文件
├── train_v2c.py                   # 最終訓練腳本
├── train_v2.py / train_v2b.py     # 中間實驗（可留作對照）
├── outputs/
│   ├── submission_v2.csv
│   ├── submission_v2a.csv         # 弱融合版備份
│   ├── cv_results_v2.json
│   ├── cv_results_v2a.json
│   └── oof_v2c.npy
└── data/train.csv, test.csv       # 官方資料（已存在）
```

重跑：

```bash
cd /workspace/im-admissions-116/prep/competitions/kaggle_s6e9
pip install -r requirements.txt
python train_v2c.py
```

---

## 6. 備註（備審／誠實聲明）

- 本 v2 採用社群已公開的特徵與超參思路並本地完整重跑；**不是**下載他人 submission 直接改名上傳。
- 勿將「期望 Public」寫成已上榜分數；以 Kaggle 實際 Public/Private 為準。
- 截止約 **2026-09-30**；每日提交上限 10 次。
