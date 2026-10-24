# 迭代計畫

公開榜（2026-10-24 快照，約 2,938 隊）：榜首 **0.700**；公開 notebook 常見銅牌附近約 **0.62–0.64**。官方 starter（字詞 TF-IDF + 邏輯迴歸）約 **0.322**。純 CPU n-gram 公開約 **0.35–0.42**；目前能穩定過 0.53 的路線幾乎都是「公開加料池 + LoRA 分類器」。

## 已實作：迭代 1（本目錄預設提交）

字元+詞+詞形 TF-IDF + 平衡邏輯迴歸 + 折內標點合法性軟懲罰 + 可選加料池。本地官方 train 鏡像 5-fold 分組 CV **0.340**。

- 驗證：`StratifiedGroupKFold` 依 `prompt_id`（隨機 K-fold 會嚴重低估）
- 執行環境：Kaggle **CPU、Internet off**
- 預期公開榜約 **0.35**（社群同結構約 CV 0.343 → LB 0.353）
- 加掛公開 in-style 池後，社群同結構約 **0.42–0.47**；分組 CV 會再高估約 0.03–0.04

這是現階段在無 GPU、不複製他人 LoRA 權重的前提下，最能完整重現的一版。

## 迭代 2：加料池權重與過濾（建議下一步，仍 CPU）

1. 在 Kaggle 掛上公開加料資料（例如 prvsiyan 18 模型擴充集、OpenRouter harvest）。
2. 只保留單段落、約 40–250 詞、無 markdown。
3. Gemini 列若含 em dash，丟掉或降權（官方 train 裡 3.1／3.6／3.8 幾乎為 0）。
4. 加料樣本權重 0.4–0.85，只在官方 train 上做分組 CV。

程式已會掃描 `/kaggle/input` 與 `data/external/`。這步不改架構，只改資料。

## 迭代 3：Qwen 編碼器 LoRA（衝銅牌／銀牌）

社群公開兩階段 LoRA（Qwen3-8B／14B，先在公開池再在官方 train 微調）公開榜約 **0.54–0.64**，與 n-gram 融合後更高。做法：

1. 手機驗證後開 **GPU T4 x2**，Internet **off**
2. 把 Qwen3-8B/14B base 與自訓 LoRA 當 Input 掛上（或另開 training notebook 產出）
3. 推論 fp16（T4 無 bf16），`max_len` 384，按長度排序
4. 與本 CPU 模型做機率融合；可加 Sinkhorn／IPF 平衡

本環境無 GPU，未在此訓練 LoRA，也不附他人 checkpoint。要上銅牌請在 Kaggle GPU 跑迭代 3，不要只交迭代 1。
