# 資料位置

本目錄的 `train.csv`／`test.csv` **不會**進 git。請用下面任一方式取得。

## 公開鏡像（免 Kaggle 登入）

Hugging Face 資料集 [loeclos/name-that-llm](https://huggingface.co/datasets/loeclos/name-that-llm) 提供與競賽相同欄位的 `train.csv`（2,699 列、150 個 `prompt_id`、18 個模型）與 3 列範例 `test.csv`。授權與競賽資料相同為 CC BY-SA 4.0。

```bash
python scripts/download_data.py
```

## 官方 Kaggle（需先同意規則）

1. 開啟 https://www.kaggle.com/competitions/name-that-llm/data
2. Join Competition 並接受規則
3. 下載 `train.csv`、`test.csv`、`sample_submission.parquet` 放到本目錄

或：

```bash
kaggle competitions download -c name-that-llm -p data/
```

## 額外公開段落（可選）

把社群產生、且標籤正好是 18 個競賽模型名稱的 csv／parquet 放到 `data/external/`。Kaggle 提交時，把那些資料集掛到 notebook 的 Input，程式會自動掃描 `/kaggle/input`。

不要把競賽 `train.csv` 再上傳到公開處；本 repo 也不提交該檔。
