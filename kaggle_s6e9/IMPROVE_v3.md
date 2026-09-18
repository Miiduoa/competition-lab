# S6E9 改進紀錄 v3（Predicting Electric Vehicle Purchases）

| 項目 | 內容 |
|------|------|
| 帳號 | kuchinwei |
| 日期 | 2026-09-17 |
| 方法 | 沿用 v2c 洩漏安全特徵 + CatBoost／XGBoost + OOF hill-climb 融合 |
| 本地 OOF AUC（chosen） | **0.94543**（hill-climb） |
| 相對 v2c OOF 0.94540 | **+0.00003** |
| Public Score | **尚未取得**（CLI 未登入；本 executor 無 browserUse／computerUse） |
| 狀態 | **檔案已就緒、GO 上傳；待 Kaggle 登入後提交** |

---

## 1. 改了什麼

### 沿用（與 v2c 相同、洩漏安全）
- digit／artifact FE（收入／通勤末位、取模、trailing-zero、分箱）
- nested target encoding（僅 fold 訓練列 fit；內層 K-Fold）
- 重用已存 `outputs/oof_v2c.npy` 與 `outputs/submission_v2.csv`（不重跑 15 個 LGBM）

### 新增模型（相同 FE／TE 管線）
1. **CatBoost**：5-fold × 2 seeds（11, 202），`border_count=254`，early stopping 200  
   - 單模 OOF **0.94525**
2. **XGBoost**：5-fold × 1 seed（11），`tree_method=hist`，`max_bin=512`  
   - 單模 OOF **0.94497**（融合權重被壓到 0）

### 融合
- 候選：等權機率平均、rank-average、hill-climb、最佳單模
- **採用 hill-climb（機率空間）**  
  - 權重：`lgbm_v2c ≈ 0.705`，`catboost ≈ 0.295`，`xgboost = 0`
  - OOF **0.94543**
- rank-avg（保留權重≥0.05 的模型）OOF 0.94542（次佳）

---

## 2. 交叉驗證結果

| 模型／融合 | OOF AUC |
|------------|---------|
| LGBM v2c（重用） | 0.94540 |
| CatBoost | 0.94525 |
| XGBoost | 0.94497 |
| equal_prob | 0.94536 |
| rank_avg_all | 0.94536 |
| **hillclimb（採用）** | **0.94543** |
| rank_avg_kept | 0.94542 |

- 指標：ROC-AUC
- 訓練總時長約 **33 分**（CPU，8 threads，無 GPU）
- TE／特徵與 v2c 一致；未改用全域 TE

詳細 fold 見 `outputs/cv_results_v3.json`。

---

## 3. GO / NO-GO

### **GO 上傳**（本地條件已滿足）

理由：
1. Chosen OOF **0.94543 ≥ 0.94540**
2. 且為多样性融合（LGBM + CatBoost），非單純重交 v2
3. Submission 格式對齊 sample（286,571 列，`id,Will_Buy_EV`）

### 上傳阻礙（誠實）
- `kaggle` CLI：**未登入**（無 `~/.kaggle/kaggle.json`／`access_token`／`KAGGLE_API_TOKEN`）
- 本 executor 工具集**沒有** `browserUse`／`computerUse`／`Task`，無法依 box-desktop skill 自行開瀏覽器上傳
- **尚未**在 Kaggle 看到 Complete／Public Score；**不得虛構分數**

建議 parent／使用者擇一：
1. 登入後執行：  
   `kaggle competitions submit playground-series-s6e9 -f submission.csv -m "v3 hillclimb LGBM0.70+CB0.30 OOF 0.94543"`
2. 或用 browser／computerUse 上傳  
   https://www.kaggle.com/competitions/playground-series-s6e9/submit  
   檔案：`/workspace/im-admissions-116/prep/competitions/kaggle_s6e9/submission.csv`  
   說明：`v3 hillclimb LGBM0.70+CB0.30 OOF 0.94543`

---

## 4. 檔案清單

```text
kaggle_s6e9/
├── train_v3.py
├── submission.csv                 # = v3（已覆寫，待上傳）
├── IMPROVE_v3.md                  # 本文件
├── STATUS.md
└── outputs/
    ├── submission_v3.csv
    ├── oof_v3.npy
    ├── oof_v3_catboost.npy
    ├── oof_v3_xgboost.npy
    ├── cv_results_v3.json
    └── train_v3.log
```

重跑：

```bash
cd /workspace/im-admissions-116/prep/competitions/kaggle_s6e9
python3 train_v3.py
```

---

## 5. 備註

- 增益相對 v2c 很小（+0.00003）；Public 可能持平或微幅上下，屬預期。
- XGBoost 在此特徵／超參下弱於 LGBM／CB，hill-climb 自動歸零合理。
- 截止約 **2026-09-30**；每日提交上限 10 次。
