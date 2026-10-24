# 資料（勿提交 CSV）

競賽頁：https://www.kaggle.com/competitions/playground-series-s6e10/data

## 競賽檔

放到本目錄：

```text
train.csv
test.csv
sample_submission.csv
```

```bash
kaggle competitions download -c playground-series-s6e10 -p .
unzip playground-series-s6e10.zip
```

或見 `../scripts/download_data.sh`。

## 原始資料（選用）

[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
（teejmahal20；約 104k + 26k 列）。訓練腳本若在 `original/` 找到檔案，會用來做 lookup 與 teacher 特徵。

```text
original/train.csv
original/test.csv
```
