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

- 公開榜榜首約 **0.703**（Hod Carriers）；前 49 名皆 ≥ **0.636**（Kaggle 只列出前 49）
- 官方 starter 公開約 **0.322**；CPU TF-IDF 約 **0.353**
- 公開兩階段 Qwen LoRA 約 **0.54–0.64**
- 約 **2,928** 隊。Featured、1,000+ 隊的銅牌是 **前 10% ≈ 第 293 名**。第 293 名的分數未公布；合理估計約 **0.48–0.55**。要進銅牌帶必須 GPU LoRA + 公開加料，不能只交 CPU 0.34。

正式提交請用 **`name_that_llm_gpu_submit.ipynb`**（Qwen LoRA）。下面先記 CPU 基線，方便對照。

## 作法（迭代 1，CPU 備援）

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
| 迭代 3：Qwen LoRA + 加料 + TF-IDF 融合（**請交這個**） | 估計公開榜 **0.50–0.58**（對齊社群 8B LoRA+池 0.538，融合略高） |

本環境無 GPU，LoRA **沒有**本地 CV。估計見 `outputs/lora_estimate.json`。社群同結構：只吃 train 的 8B LoRA 分組 CV 0.443／公開 0.466；加公開池 CV 0.576／公開 0.538（CV 約高估 0.03–0.04）。

## Kaggle 執行設定（銅牌／GPU 版）

交 **`name_that_llm_gpu_submit.ipynb`**：

- **Internet：Off**
- **Accelerator：GPU T4 x2**（要手機驗證）
- 時限 9 小時；4-bit LoRA 一輪加料 + 兩輪官方 train + 4,000 列推論，預估數小時
- 權重不要現下：把公開 base 當 Kaggle Model 掛上

### 必須掛上的 Input

| 類型 | 名稱 | 路徑／用法 |
|------|------|------------|
| 競賽資料 | `name-that-llm` | `train.csv` / `test.csv` |
| 預訓練模型（Apache 2.0） | Kaggle Model **`qwen-lm/qwen-3`** → Transformers → **8b-base** / 1 | 通常 `/kaggle/input/qwen-3/transformers/8b-base/1` |
| 加料（至少一個） | 見下表 | 程式掃描 `/kaggle/input` 的 csv／parquet，**忽略**他人 LoRA adapter |

建議加料（皆公開、規則允許的外部資料）：

- `nishantdahal/ntl-two-stage-lora`：只用其中的 **`pool_v1.parquet`**（約 3.5 萬列公開段落）。同資料集裡的 `lora8b_v1/`、`lora14b_v1/` **不會**被讀取。
- `prvsiyan/prvsiyans-extended-name-that-llm-dataset`（18 類齊）
- `seyitkaangunes/ntl-extra-api-passages`
- `willrice/name-that-llm-openrouter-harvest`

Gemini 列若帶 em dash 會被丟掉（官方 train 裡 3.1／3.6／3.8 幾乎為 0）。

CPU 備援 notebook：`name_that_llm_submit.ipynb`（Internet off、無 GPU），本地 CV 0.340，不夠銅牌。

## 在 Kaggle 提交（GPU 版逐步）

1. 帳號完成**手機驗證**（否則沒有 GPU）。
2. 加入競賽並接受規則（2027-01-06 前）。
3. New Notebook，掛上表三類 Input。
4. 上傳 `name_that_llm_gpu_submit.ipynb`。
5. Settings：**Internet off**，Accelerator **GPU T4 x2**。
6. Save Version → Save & Run All。互動階段 test 只有 1–3 列，計分時會重跑約 4,000 列。
7. Output 交 `submission.parquet`。每天 5 次；截止前自選 2 個 Final。

## 目錄

```text
kaggle-name-that-llm/
├── name_that_llm_gpu_submit.ipynb   # 交給 Kaggle 的 GPU 版（LoRA）
├── name_that_llm_submit.ipynb       # CPU 備援
├── README.md
├── ITERATIONS.md
├── ntl/
├── scripts/download_data.py
├── scripts/train_cv.py
├── data/README.md
└── outputs/cv_results.json
└── outputs/lora_estimate.json
```

## 誠實聲明

- 本地若使用 Hugging Face 鏡像，那是公開可下載的 CC BY-SA 4.0 表，欄位與列數對得上官方 train；**仍須在 Kaggle 同意規則才能正式提交**。
- 未在此環境上傳 Leaderboard，沒有官方名次。
- 未複製公開 LoRA notebook 或他人 checkpoint。
