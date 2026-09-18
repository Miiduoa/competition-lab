#!/usr/bin/env python3
"""從 train_labels.csv 計算陽性比例，寫入 submission_src/assets/prior.txt。"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "train_labels.csv"
OUT = ROOT / "submission_src" / "assets" / "prior.txt"


def main() -> None:
    if not LABELS.exists():
        print(f"找不到 {LABELS}。請先註冊並下載訓練資料到 data/。", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(LABELS)
    prior = float(df["is_pathologic"].mean())
    prior = min(max(prior, 0.01), 0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"{prior:.6f}\n")
    print(f"prior={prior:.6f} -> {OUT}")


if __name__ == "__main__":
    main()
