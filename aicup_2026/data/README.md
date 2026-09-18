# AI CUP 2026 訓練資料（勿提交至本 repo）

訓練集約 **1.3 GB**，請自行從官方 GitHub Release 下載，勿將解壓後的 CSV 推上 GitHub。

## 下載

來源：[AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset) Releases

建議檔案：`aicup2026_go_training.tar.xz`（約 215 MB compressed）

```bash
# 於 competition-lab/aicup_2026/ 下執行
mkdir -p data
cd data
# 以瀏覽器或 gh release download 取得 aicup2026_go_training.tar.xz
tar -xJf aicup2026_go_training.tar.xz
# 解壓後應有 task1/、task2/（或同等結構）之 train_*.csv
```

參考 SHA256（來源 STATUS.md，2026-09-18）：

`959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b`

## 路徑約定

各 baseline 腳本預期資料位於相對路徑 `../data/`（詳見各版 README）。
