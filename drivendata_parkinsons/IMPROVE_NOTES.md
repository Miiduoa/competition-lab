# 改進說明（IMPROVE_NOTES）

日期：2026-09-15  
帳號：demohan513  
舊提交：ID `322418`（Pending；本機無 browser，**無法在此查詢最新狀態**）

## 做了什麼

### 資料
- 確認 `data/`：`niftis.zip`（1362 張）、`train_labels.csv`（1362 列，陽性率 ≈ **0.548**）、`smoke_test_data.tar.gz`
- 已解壓 `niftis/` 與 smoke；**未**將醫學影像／標籤外傳至雲端 LLM

### 模型（中等 baseline，非完整 3D CNN）
本機無 GPU、記憶體約 16GB，完整 3D CNN 訓練成本過高，改採：

1. **Scale-/size-invariant 影像特徵**（`features.py`）  
   體積先下採樣至約 96³，再計算：
   - 中心 ROI／全域／背景攝取比（紋狀體相對攝取）
   - 高峰強度比、高分位 voxel 佔比
   - 左／右、前／後、上／下不對稱
   - 強度加權質心偏移、CV、偏度等  
   → 適應多中心不同矩陣尺寸與強度尺度

2. **LightGBM + sigmoid 機率校準**（`CalibratedClassifierCV`）  
   - 5-fold OOF：**log loss ≈ 0.496**、**AUC ≈ 0.857**  
   - 常數先驗 log loss ≈ **0.688**（明顯改善）  
   - 權重：`assets/lgbm_calibrated.joblib`（備援 `lr_calibrated.joblib`）  
   - `assets/prior.txt` 更新為真實陽性率 `0.548458`

3. **推論** `main.py`  
   讀 `/code_execution/data/`，抽同樣特徵 → 校準機率 → 寫 `submission.csv`（夾到 `[1e-4, 1-1e-4]`）

### 本地驗證
- Smoke（20 例，含標籤）：log loss ≈ 0.33、AUC 1.0（樣本小且可能與 train 重疊，僅作格式／sanity）
- ZIP 根目錄含 `main.py`；依賴皆在官方 runtime（nibabel / lightgbm / sklearn / joblib）

## 限制（請知悉）
- **非**深度 3D／2.5D CNN；未做空間正規化至標準模板（runtime 有 ANTs／MONAI，後續可加強）
- 特徵為粗略解剖假設（中心＝紋狀體），跨中心定位誤差會影響分數
- OOF 0.50 為誠實估計；全資料重訓後 in-sample 分數會過度樂觀，勿當 leaderboard 預期
- 未在官方 Docker runtime 跑 `just test-submission`（本環境無該映像）

## 產出路徑
- **新 submission**：`/workspace/im-admissions-116/prep/competitions/drivendata_parkinsons/submission/submission.zip`（約 786 KB）
- 訓練腳本：`scripts/extract_and_train.py`
- 指標：`artifacts/cv_metrics.json`、`submission_src/assets/model_meta.json`

## 建議上傳與否
- **建議上傳**：相對舊常數 prior baseline，預期 log loss 明顯下降；格式符合 code execution
- 建議流程：先 **Smoke test** → 通過再正式提交
- 上傳由 **parent computerUse** 執行；本子代理未自行開 browser
