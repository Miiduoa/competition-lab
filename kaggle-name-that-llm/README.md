# Name That LLM（Kaggle Code Competition）

| 項目 | 內容 |
|------|------|
| 競賽 | [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) |
| 題目 | 判斷短段落由 18 個前沿 LLM 的哪一個寫出 |
| 指標 | **balanced accuracy**（18 類召回平均） |
| 報名截止 | 2027-01-06 23:59 UTC |
| 最終提交 | 2027-01-13 23:59 UTC |
| 賽制 | Notebook 提交；隱藏測試約 4,000 列；公開榜約 40%、私榜約 60% |

## 目前公開資訊（2026-10-24）

- 公開榜榜首約 **0.700**（Hod Carriers）
- 官方／熱門 starter（TF-IDF + 邏輯迴歸）公開約 **0.322**
- 社群 CPU 字元+詞 TF-IDF 公開約 **0.353**
- 公開兩階段 LoRA notebook 約 **0.59–0.64**
- 參賽隊約 2,938；Featured 賽銅牌大約是前 10% 量級。以目前分數帶，**銅牌需要明顯高於純 n-gram**，實務上幾乎都要加料資料 + GPU 編碼器。

## 作法（迭代 1，本 notebook）

1. **多視角 TF-IDF**（保留大小寫與標點）：字元 1–5 gram（含單一字元破折號／引號）、詞 1–2 gram、詞形 n-gram。
2. **18 類邏輯迴歸**（`C=10`、`class_weight='balanced'`）。LinearSVC 在同一資料上較差（約 0.329），所以不用。
3. **折內標點合法性**：官方 train 裡 Gemini 3.1／3.6／3.8 沒有 em dash，Claude／GLM 幾乎不用彎引號。某標在該折訓練中某模型從未出現時，測試若出現則軟懲罰該類機率。
4. **可選加料**：掃描 `/kaggle/input` 與 `data/external/` 中帶有 18 個官方標籤的公開段落，權重 0.55。
5. **驗證**：`StratifiedGroupKFold` 依 `prompt_id`。測試 prompt 不在 train 裡；隨機切分會把同題其他模型漏進訓練，CV 會偏低一截。
6. **不做**：依 prompt 叢集做匈牙利指派（社群 CV 好看、公開榜掉分）；本版也不開 GPU。

提交檔必須是 `submission.parquet`，欄位 `id,model`，`model` 只能是 18 個官方名稱。

## 本地 CV

已用公開 Hugging Face 鏡像（2,699 列、150 個 prompt、18 類，與官方 train 對得上）跑 **5-fold `StratifiedGroupKFold(prompt_id)`**：

| 項目 | 值 |
|------|-----|
| 整體 balanced accuracy | **0.340** |
| 五折 | 0.319, 0.396, 0.326, 0.328, 0.333 |
| 較易 | `gemini-3.1-pro-preview` 0.65、`gpt-6-astra`／`claude-opus-5` 0.53 |
| 較難 | `gpt-5.6-terra` 0.20、`claude-fable-5` 0.14、`glm-5.3` 0.19 |

與社群 CPU 基線（分組 CV 0.343、公開榜 0.353）同一帶。同題其他模型的 Gemini Flash／GPT-5.6 兄弟最容易互混。

```bash
python scripts/download_data.py
python scripts/train_cv.py --folds 5
```

分數寫在 `outputs/cv_results.json`。**這是本地分組 CV，不是公開榜。** 無資料時 `--smoke` 只用合成段落檢查管線，分數不可拿去對榜。

## 預期公開榜

| 設定 | 預期公開榜（約） |
|------|------------------|
| 只吃官方 2,699 列（本 notebook 預設） | **約 0.35**（對齊 CV 0.340；社群同結構 CV→LB 約 +0.01） |
| 再掛公開 in-style 加料池 | 0.42–0.47（CV 可能再高 0.03） |
| 迭代 3：Qwen LoRA + 加料 + 融合 | 0.54–0.64（銅牌帶） |

## Kaggle 執行設定

本提交版選：

- **Internet：Off**（計分規定）
- **Accelerator：None / CPU**
- 時限 9 小時；此 CPU 管線在隱藏 4,000 列上應遠低於此
- 可自由掛公開資料與預訓練模型；本 notebook 會掃描 `/kaggle/input` 裡帶有 18 類標籤的 csv／parquet

**不要**為這個版本開 GPU。Kaggle GPU 通常要**手機驗證**；那是迭代 3 才需要。

讀檔路徑同時相容：

- `/kaggle/input/competitions/name-that-llm/test.csv`（Overview 寫法）
- `/kaggle/input/name-that-llm/test.csv`（常見掛載）

## 在 Kaggle 提交（逐步）

1. 用**一個**帳號加入競賽並接受規則（2027-01-06 前）。
2. New Notebook，掛 Name That LLM 競賽資料。
3. 把 `name_that_llm_submit.ipynb` 上傳，或複製儲存格。
4. Settings：**Internet off**，Accelerator **CPU**。
5. Save Version → Save & Run All。互動階段 `test.csv` 只有 1–3 列，很快跑完不代表隱藏集也一樣快，但本 CPU 版沒問題。
6. Viewer → Output → Submit `submission.parquet`。
7. 每天每隊最多 5 次。截止前自選 2 個 Final Submission。

加料（可選）：在 Add Input 掛公開段落資料集，不必改程式。

## 目錄

```text
kaggle-name-that-llm/
├── name_that_llm_submit.ipynb   # 交給 Kaggle 的 notebook
├── README.md
├── ITERATIONS.md
├── requirements.txt
├── ntl/                         # 本地訓練用套件
├── scripts/download_data.py
├── scripts/train_cv.py
├── scripts/predict.py
├── data/README.md               # 資料怎麼取得（csv 不進 git）
└── outputs/cv_results.json
```

## 誠實聲明

- 本地若使用 Hugging Face 鏡像，那是公開可下載的 CC BY-SA 4.0 表，欄位與列數對得上官方 train；**仍須在 Kaggle 同意規則才能正式提交**。
- 未在此環境上傳 Leaderboard，沒有官方名次。
- 未複製公開 LoRA notebook 或他人 checkpoint。
