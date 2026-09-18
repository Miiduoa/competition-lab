# FINAL V3 NOTES — DrivenData DaT Parkinson’s Challenge

日期：2026-09-15  
帳號：demohan513  
對照舊最佳公開分：**0.6478**（submission 322436）  
本機 OOF（校準後）：**log loss ≈ 0.4598** / **AUC ≈ 0.878**  
Smoke 本地（20 例）：**log loss ≈ 0.275** / AUC 1.0（樣本小且可能與 train 重疊，僅格式／sanity）

## 建議：**GO**（建議上傳）

理由：
1. OOF **0.460 ≤ 0.48**，明顯優於 v2 的 ~0.496。
2. 針對「幾何中心 ≠ 紋狀體」做了強度定位；訓練集抽樣約 **86%** 體積亮區質心相對幾何中心有明顯偏移，這很可能是 0.50→0.65 公開崩塌的主因之一。
3. 以 isotonic 校準 + **prior blend α=0.10** + clip `[0.01, 0.99]` 壓低極端機率，降低 log loss 在分布偏移下的懲罰（v2 公開失敗模式：過度自信）。
4. ZIP 格式合格、smoke 4.6s / 20 例，遠低於 6 分鐘限制。

預期公開區間（誠實估計）：**約 0.52–0.62**（若紋狀體定位與校準對上測試分布）；若仍有中心偏移／多中心強度差異，可能落在 **0.60–0.68**。要穩贏 0.6478 並非保證，但相對 v2 有明確機制修復。

## 相對 v2 改了什麼

| 項目 | v2 | v3 |
|------|----|----|
| ROI | 固定幾何中心 | **強度 top≈97.5% 亮區質心** 定位紋狀體 |
| 特徵 | 18 維粗特徵 | **38 維**：SBR、L/R binding 不對稱、putamen/caudate-like、熱區形狀、多尺度 shell |
| 模型 | 單一 Calibrated LGBM | **LGBM + HistGB + LR** 加權集成（權重 ≈ 0.60 / 0.35 / 0.05） |
| 校準 | sigmoid CV | Isotonic（樹模型）+ sigmoid（LR）+ **prior blend α=0.10** |
| Clip | `[1e-4, 1-1e-4]` | **`[0.01, 0.99]`**（避免極端） |

## 本地 CV 指標

- 原始集成 OOF log loss ≈ **0.4476**（α=0）
- 提交用（α=0.10 + soft clip）OOF log loss ≈ **0.4598**，AUC ≈ **0.878**，Brier ≈ 0.147
- 各 fold 集成 log loss（α=0 均值前）：約 0.43–0.48
- 先驗常數 log loss ≈ 0.688

詳見 `artifacts/cv_metrics_v3.json`、`artifacts/oof_v3.csv`。

## 為何預期能打贏 0.6478

1. **定位修復**：v2 假設紋狀體在體積中心；資料顯示多數掃描並非如此。v3 用亮區質心對準攝取熱點，SBR／左右不對稱才有臨床意義。
2. **PD 相關特徵**：左右 binding 不對稱、put/cau 比、SBR 是典型 DaT 訊號，比純全域統計更可遷移。
3. **校準對 log loss**：公開崩塌常來自過度自信錯誤；α=0.10 與較緊 clip 犧牲少許 OOF（0.448→0.460），換公開偏移下的穩定性。

## 風險

- Smoke AUC=1.0 不可外推；正式公開仍可能因多中心協定／重建差異掉分。
- 紋狀體定位仍是啟發式（非模板配準）；極端裁切／極低對比掃描可能失敗並退回 prior。
- Isotonic 在小 fold 上可能輕微過擬合校準曲線；已用 prior blend 緩衝。
- 未跑官方 Docker `just test-submission`（本環境無該映像）。
- **最後 1/3 配額**：若 smoke 平台分異常高於本地，先停、勿急交正式。

## 產物路徑

- `submission/submission.zip`（根目錄含 `main.py`）
- `submission_src/{main.py,features.py,assets/*}`
- `scripts/train_v3.py`
- `artifacts/cv_metrics_v3.json`
- `FINAL_V3_NOTES.md`（本檔）

## 上傳流程（給 parent）

1. 先交 **Smoke test**（預期快速通過）。
2. Smoke 分數合理後再交正式 submission。
3. 本子代理**未**開 browser、**未**上傳醫學資料至外部。
