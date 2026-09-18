#!/usr/bin/env python3
"""不需真實資料：驗證 baseline 能寫出合法 submission.csv 欄位。"""
from pathlib import Path
import tempfile
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data = td / "data"
        niftis = data / "niftis"
        niftis.mkdir(parents=True)
        fmt = pd.DataFrame({"uid": ["aaa11111", "bbb22222"], "is_pathologic": [0.0, 0.0]})
        fmt.to_csv(data / "submission_format.csv", index=False)

        # 暫時改寫路徑：直接呼叫邏輯
        import main as m

        m.DATA_ROOT = data
        m.NIFTI_DIR = niftis
        m.SUBMISSION_FORMAT_PATH = data / "submission_format.csv"
        m.WRITE_SUBMISSION_PATH = td / "submission.csv"
        m.PRIOR_PATH = ROOT / "submission_src" / "assets" / "prior.txt"
        m.main()

        out = pd.read_csv(td / "submission.csv")
        assert list(out.columns) == ["uid", "is_pathologic"]
        assert len(out) == 2
        assert out["is_pathologic"].between(0, 1).all()
        print("OK format smoke:", out.to_dict(orient="records"))


if __name__ == "__main__":
    main()
