# 迭代計畫

公開榜（2026-10-24）：榜首 **0.703**；前 49 名 ≥ **0.636**。約 2,928 隊。Featured 銅牌 = 前 10% ≈ **第 293 名**（該名分數未公布，估計約 0.48–0.55）。

## 迭代 1（已實作，CPU 備援）

字元+詞+詞形 TF-IDF + 平衡邏輯迴歸 + 標點軟懲罰。本地分組 CV **0.340**，預期公開榜約 0.35。不夠銅牌。

Notebook：`name_that_llm_submit.ipynb`（CPU、Internet off）。

## 迭代 2（已併進 GPU 版）

公開 in-style 加料：掃描 `/kaggle/input` 的 csv／parquet，過濾 markdown、30–300 詞、Gemini em dash 等訓練裡不該出現的標點。權重與 LoRA 兩階段訓練一起用。

## 迭代 3（已實作，請交這個）

Qwen3-8B-Base **4-bit LoRA**（r=16/alpha=32）分類頭：

1. 加料 1 epoch（lr 2e-4），再官方 train 2 epoch（lr 8e-5）
2. 與 TF-IDF 機率 0.78 / 0.22 融合，再標點懲罰與 IPF 平衡
3. 推論 fp16、max_len 384、按長度排序

- Notebook：`name_that_llm_gpu_submit.ipynb`
- 環境：GPU **T4 x2**、Internet **off**
- Base：Kaggle Model `qwen-lm/qwen-3` Transformers **8b-base**（Apache 2.0），不另訓、不斷網下載
- 不加他人 LoRA checkpoint；adapter 在提交 notebook 裡現場訓練
- 估計分組 CV ~**0.55**、公開榜 ~**0.50–0.58**（本機無 GPU，未實跑 LoRA CV）

社群對照：8B LoRA 只吃 train → CV 0.443 / LB 0.466；加公開池 → CV 0.576 / LB 0.538。
