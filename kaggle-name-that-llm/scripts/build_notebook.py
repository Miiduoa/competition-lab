#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook from the local package sources."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def strip_relative_imports(src: str) -> str:
    lines = []
    for line in src.splitlines():
        if line.startswith("from .constants import"):
            continue
        if line.startswith("from .features import"):
            continue
        if line.startswith("from .pipeline import"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    constants = (ROOT / "ntl" / "constants.py").read_text()
    features = strip_relative_imports((ROOT / "ntl" / "features.py").read_text())
    pipeline = strip_relative_imports((ROOT / "ntl" / "pipeline.py").read_text())
    driver = r'''
from pathlib import Path
import json

print("Name That LLM — multi-view n-gram + stylometry + family hierarchy")

def _pick(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

train_path = _pick([
    "/kaggle/input/competitions/name-that-llm/train.csv",
    "/kaggle/input/name-that-llm/train.csv",
    "data/train.csv",
    str(Path.cwd() / "data" / "train.csv"),
])
test_path = _pick([
    "/kaggle/input/competitions/name-that-llm/test.csv",
    "/kaggle/input/name-that-llm/test.csv",
    "data/test.csv",
    str(Path.cwd() / "data" / "test.csv"),
])
if train_path is None or test_path is None:
    raise FileNotFoundError("train.csv / test.csv not found. Attach the competition dataset.")

train = normalize_frame(load_table(train_path))
test = normalize_frame(load_table(test_path))
extra_roots = ["/kaggle/input"]
local_extra = Path("data/external")
if local_extra.exists():
    extra_roots.append(local_extra)
extra = load_extra_frames(extra_roots)
if len(extra):
    extra = extra[~extra["text"].isin(set(train["text"]))].reset_index(drop=True)
    extra = extra[~extra["id"].isin(set(test["id"]))].reset_index(drop=True)
print(f"train={len(train)} test={len(test)} extra={len(extra)}")

texts = train["text"].astype(str).tolist()
labels = train["model"].astype(str).tolist()
weights = [1.0] * len(texts)
EXTRA_WEIGHT = 0.55
if len(extra):
    texts.extend(extra["text"].astype(str).tolist())
    labels.extend(extra["model"].astype(str).tolist())
    weights.extend([EXTRA_WEIGHT] * len(extra))

clf = NTLClassifier(extra_weight=EXTRA_WEIGHT, rebalance=True)
clf.fit(texts, labels, sample_weight=np.asarray(weights, dtype=np.float64))
pred = clf.predict(test["text"].astype(str).tolist())
out_path = Path("submission.parquet")
write_submission(test["id"].astype(str), pred, out_path)
check = pd.read_parquet(out_path)
assert list(check.columns) == ["id", "model"]
assert check["id"].nunique() == len(test)
assert set(check["model"]).issubset(set(LABELS))
assert not check.isnull().any().any()
print(check.head())
print("wrote", out_path.resolve(), "rows", len(check))
print(json.dumps({"n_train": len(train), "n_extra": int(len(extra)), "n_pred": len(check)}, indent=2))
'''
    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        new_markdown_cell(
            """# Name That LLM — CPU submission

Identify which of 18 frontier models wrote each passage.

This notebook trains on the competition `train.csv` (and any extra public passages
attached under `/kaggle/input` that use the same 18 labels). It writes
`submission.parquet`.

**Runtime settings**

- Internet: **off**
- Accelerator: **None** (CPU is enough)
- Persistence: files only
- Time budget: well under 9 hours on the hidden ~4000-row rerun

Do not enable GPU for this version. A later LoRA iteration would use GPU T4 x2.
"""
        ),
        new_code_cell(constants),
        new_code_cell(features),
        new_code_cell(pipeline),
        new_code_cell(driver.strip() + "\n"),
    ]
    dest = ROOT / "name_that_llm_submit.ipynb"
    dest.write_text(nbf.writes(nb))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
