# Baseline 與進階策略備註

## 最小可交
- 常數先驗 = mean(train is_pathologic) → log loss 下限參考
- 啟發式中心強度：無訓練時僅驗證管線

## 建議 48h 模型
1. nibabel 讀取 → resample 到固定 spacing（如 2mm）→ crop/pad 128³
2. 輕量 3D CNN 或三平面 2.5D EfficientNet/ResNet
3. BCE / focal；驗證集上 temperature scaling 優化 log loss
4. TTA：輕微翻轉／小位移平均機率
5. 權重打進 submission.zip 的 `assets/`；`main.py` 載入後批次推論

## 合規
- 外部資料回報主辦；爭獎需公開且可商業授權
- 資料不進留存型雲端 LLM
