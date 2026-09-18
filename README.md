# competition-lab｜線上競賽筆記與可重現程式

顧晉瑋（Providence IM）｜Email: demohan513@gmail.com｜Kaggle: [kuchinwei](https://www.kaggle.com/kuchinwei)

本 repo 彙整**真實參與**的競賽筆記、分數表與精簡程式碼。  
**誠實聲明：尚未得獎；不宣稱冠軍或官方名次。** 本地驗證分數 ≠ 官方排行榜。

## 分數總表（公開／據實）

詳見 [`SCORES.md`](SCORES.md)。

| 競賽 | 帳號 | 狀態 | 公開分數／備註 |
|------|------|------|----------------|
| 臺鐵數據力 2026｜RailFlow-DSS | — | 海選企劃**已送出鎖定** | 企劃向；非得獎 |
| Kaggle Playground S6E9 EV | kuchinwei | Complete | Public **0.94540** |
| Kaggriculture | kuchinwei | Complete | Public **600.0** |
| DrivenData DaT Parkinson | demohan513 | Completed | Public **0.5591**（log loss↓越好） |
| AI CUP 2026 Autumn Task1／2 | TEAM_10925／10926 | **已報名／建模中** | 本地 holdout 約 Task1≈0.09、Task2≈0.30（**非官方榜**） |

## 目錄

| 路徑 | 內容 |
|------|------|
| [`tra_data2026/`](tra_data2026/) | RailFlow-DSS 企劃 PDF／MD（無個資附件） |
| [`kaggle_s6e9/`](kaggle_s6e9/) | EV 購買預測訓練腳本＋筆記（無 train/test 大檔） |
| [`kaggriculture/`](kaggriculture/) | Agent `submission/*.py`＋筆記 |
| [`drivendata_parkinsons/`](drivendata_parkinsons/) | 訓練腳本＋策略筆記（無醫學影像） |
| [`aicup_2026/`](aicup_2026/) | `baseline`／`baseline_v2`／`baseline_v3` 程式＋STATUS（**無 1.3GB 資料**） |

## 資料下載（重要）

各子資料夾的 `data/README.md` 說明官方下載方式：

- **AI CUP**：https://github.com/AILAB-NDHU/aicup2026-go-dataset Releases（`aicup2026_go_training.tar.xz`）
- **Kaggle S6E9**：`kaggle competitions download -c playground-series-s6e9`
- **DrivenData Parkinson**：依競賽頁規則下載（勿公開上傳影像）

## 方法一覽（極短）

- **S6E9**：表格分類；LGBM／CatBoost／XGB＋hill-climb 融合；時間／折驗證 OOF  
- **Kaggriculture**：環境 agent（種植／澆水／收成／賣出策略迭代）  
- **Parkinson**：醫學影像分類 baseline → 迭代提交（容器化）  
- **AI CUP**：SGF 特徵／檢索／LightGBM 等；見各 baseline README  
- **RailFlow-DSS**：開放資料＋旅運負荷預測決策支援**企劃**（非已上線系統）

## License

筆記與自寫程式碼以學習／備審展示為主；各競賽資料與官方規則仍受原主辦條款拘束。建議視需要為本 repo 加上 MIT（自寫碼部分）。
