# DrivenData DaT Parkinson's Challenge（SFMN）

緊急備戰專案｜帳號 Email：`demohan513@gmail.com`  
競賽頁：https://www.drivendata.org/competitions/311/sfmn-parkinsons-challenge/

## 截止確認（2026-09-14 查證）

| 項目 | 內容 |
|------|------|
| **截止** | **2026-09-16 23:59 UTC**（約剩 2 天） |
| 主辦 | SFMN + Health Data Hub + GaelO / DrivenData |
| 獎金 | €25,000（1st €12,500 / 2nd €7,500 / 3rd €5,000） |
| 任務 | 將 DaT scan（`.nii.gz` 3D）分為 **normal / abnormal**，輸出異常機率 |
| 評分 | **Log loss**（愈低愈好）；AUROC 僅參考 |
| 提交型態 | **Code execution**：上傳 `submission.zip`（根目錄必含 `main.py`），容器內推論 |

## 資料與規則（必守）

- 訓練：`data/niftis/*.nii.gz` + `train_labels.csv`（`uid`, `is_pathologic` ∈ {0.0, 1.0}）
- 測試集**不可下載**，僅在平台容器掛載於 `/code_execution/data/`
- **禁止**將競賽資料上傳／貼到會**留存**的雲端 LLM（ChatGPT、Gemini、Codex 等）
- 僅限競賽期間使用；結束後刪除本地資料（除非另有授權）
- 外部資料／預訓練權重允許，但須有權使用；若要爭獎，外部資料須公開且授權允許商業釋出（不可 NC）
- 推論環境：**無網路**、Python 3.12、A100 80GB、≤3 小時（smoke ≤6 分鐘）

## 專案結構

```
drivendata_parkinsons/
├── README.md                 # 本檔
├── 報名與提交步驟.md         # 註冊／下載／提交 checklist
├── submission_src/           # 打包進 ZIP 的推論碼
│   ├── main.py
│   └── assets/prior.txt
├── scripts/
│   ├── pack_submission.sh
│   ├── compute_prior.py
│   └── smoke_local_format.py
├── data/                     # 【需登入下載】放訓練／smoke 資料（已 .gitignore）
└── submission/               # pack 產出 submission.zip
```

## 如何註冊（需 browser）

> **需 browser 註冊**：DrivenData 帳號註冊、Compete 報名、資料下載皆需登入，本機無法匿名下載（data 頁回 403）。

1. 開啟競賽頁 → 註冊／登入（建議用 `demohan513@gmail.com`）
2. 側欄按 **Compete!** 報名並同意規則
3. 至 [Data](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/data/) 下載：
   - 訓練資料（含 `niftis/`、`train_labels.csv`）
   - `smoke_test_data.tar.gz`（本地／smoke 測試用）
4. 解壓到本專案 `data/`（勿 git commit、勿上傳雲端）

詳見 `報名與提交步驟.md`。

## Baseline 策略

### 立即可交（本 repo 已備）

1. **常數先驗**：`train_labels` 陽性比例 → `assets/prior.txt`（log loss 安全底線）
2. **無標籤時**：中心體積相對強度啟發式（僅驗證格式；分數通常很差）
3. 打包：`bash scripts/pack_submission.sh` → `submission/submission.zip`
4. 平台先交 **Smoke test**，通過再交正式 submission

### 有資料後建議（48h 衝刺）

1. 重採樣到固定 spacing／尺寸（例如 2mm isotropic、128³）
2. 簡單 3D CNN 或 2.5D（多軸切片 + 2D backbone）二分類
3. 分層 K-fold，輸出校準機率（Platt / temperature）— log loss 關鍵
4. 可考慮公開外部 DaT／預訓練（須符合授權與回報義務）
5. 用官方 runtime：https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime  
   `just pack-submission` → `just test-submission`（需 Docker + smoke 資料）

### 提交輸出格式

```csv
uid,is_pathologic
xaji0y6d,0.04
pbhsahxt,0.12
```

`is_pathologic` 為 **[0,1] 機率**，勿交硬 0/1。

## 本地進度狀態

| 項目 | 狀態 |
|------|------|
| 截止／規則／格式查證 | ✅ 完成 |
| 專案目錄＋README＋步驟文件 | ✅ 完成 |
| Baseline `main.py`＋打包腳本 | ✅ 完成 |
| 格式 smoke（無真實 NIfTI） | 見下方指令 |
| DrivenData 註冊／Compete | ⏳ **需 computerUse／browser** |
| 下載訓練／smoke 資料 | ⏳ **需登入** |
| 真實模型訓練 | ⏳ 等資料 |
| 平台 smoke／正式提交 | ⏳ 需登入上傳 |

## 快速指令

```bash
# 格式驗證（無需競賽資料）
python3 scripts/smoke_local_format.py

# 有 train_labels 後更新先驗
python3 scripts/compute_prior.py

# 打包 ZIP
bash scripts/pack_submission.sh
```

## 重要連結

- Overview：https://www.drivendata.org/competitions/311/sfmn-parkinsons-challenge/
- Problem：https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/990/
- Code submission format：https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/
- Runtime repo：https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime
- Submissions：https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/submissions/
