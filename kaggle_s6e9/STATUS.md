# 狀態快照（2026-09-17）

| 項目 | 狀態 |
|------|------|
| Kaggle 帳號 | **kuchinwei**（已 Join `playground-series-s6e9`） |
| 官方資料 | **已在** `data/`（train 668,665 / test 286,571） |
| 版本 | **v3**（見 `IMPROVE_v3.md`） |
| 模型 | v2c LGBM OOF 重用 + CatBoost（5×2）+ XGB（5×1）；hill-climb 融合 |
| OOF AUC | **0.94543**（v2c 曾為 0.94540） |
| Public Score（v3） | **0.94540**（Complete） |
| 可上傳檔 | `submission.csv` / `outputs/submission_v3.csv` |
| 狀態 | **已上傳；Complete／已評分** |
| Leaderboard | v2 Public **0.94540**（已提交）；v3 Public **0.94540** |

## 路徑

- Submission：`kaggle_s6e9/submission.csv`
- 改進說明：`IMPROVE_v3.md`（v2 見 `IMPROVE_v2.md`）
- CV：`outputs/cv_results_v3.json`

## v3 上傳紀錄

- 狀態：**已上傳並完成評分（Complete）**
- 上傳時間：2026-09-17 23:00:53 CST（Taipei，UTC+8）
- 帳號：**kuchinwei**
- 描述：`v3 hillclimb LGBM0.70+CB0.30 OOF 0.94543`
- Public Score：**0.94540**
- 證據截圖：`submit_v3.png`

## 先前 v2 上傳紀錄（保留）

- 上傳時間：2026-09-15 05:35:48 UTC（約 2026-09-15 13:35 CST）
- 帳號：**kuchinwei**
- 描述：`baseline v2 OOF 0.94540`
- Public Score：**0.94540**
- 證據截圖：`outputs/submit_v2.png`
