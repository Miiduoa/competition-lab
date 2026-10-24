#!/usr/bin/env python3
"""Build the GPU LoRA submission notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def strip_relative_imports(src: str) -> str:
    drop_prefixes = (
        "from .constants import",
        "from .features import",
        "from .pipeline import",
        "from .extra import",
        "from .lora import",
    )
    lines = []
    for line in src.splitlines():
        if line.startswith(drop_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    constants = (ROOT / "ntl" / "constants.py").read_text()
    features = strip_relative_imports((ROOT / "ntl" / "features.py").read_text())
    pipeline = strip_relative_imports((ROOT / "ntl" / "pipeline.py").read_text())
    extra = strip_relative_imports((ROOT / "ntl" / "extra.py").read_text())
    lora = strip_relative_imports((ROOT / "ntl" / "lora.py").read_text())
    driver = r'''
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("Name That LLM — Qwen LoRA + TF-IDF blend")

def _pick(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

train_path = _pick([
    "/kaggle/input/competitions/name-that-llm/train.csv",
    "/kaggle/input/name-that-llm/train.csv",
    "data/train.csv",
])
test_path = _pick([
    "/kaggle/input/competitions/name-that-llm/test.csv",
    "/kaggle/input/name-that-llm/test.csv",
    "data/test.csv",
])
if train_path is None or test_path is None:
    raise FileNotFoundError("Attach the competition dataset (train.csv / test.csv).")

train = normalize_frame(load_table(train_path))
test = normalize_frame(load_table(test_path))
extra = load_and_clean_extra(
    ["/kaggle/input", Path("data/external")],
    train,
    test_ids=test["id"].astype(str),
    max_rows=16_000,
)
print(f"train={len(train)} test={len(test)} extra={len(extra)}")
if extra.empty:
    print("WARNING: no extra public passages attached. Bronze usually needs a public pool.")

texts = train["text"].astype(str).tolist()
labels = train["model"].astype(str).tolist()
weights = [1.0] * len(texts)
if len(extra):
    texts.extend(extra["text"].astype(str).tolist())
    labels.extend(extra["model"].astype(str).tolist())
    weights.extend([0.55] * len(extra))

print("Fitting TF-IDF / logistic regression", flush=True)
tfidf = NTLClassifier(extra_weight=0.55, rebalance=False)
tfidf.fit(texts, labels, sample_weight=np.asarray(weights, dtype=float))
p_tfidf = tfidf.predict_proba(test["text"].astype(str).tolist())
# align to LABELS
col = {str(c): i for i, c in enumerate(tfidf.classes_)}
p_tfidf_aligned = np.zeros((len(test), len(LABELS)))
for j, lab in enumerate(LABELS):
    if lab in col:
        p_tfidf_aligned[:, j] = p_tfidf[:, col[lab]]

print("Loading Qwen base + LoRA", flush=True)
base = find_base_model(["/kaggle/input", "models"])
print("base model:", base, flush=True)
model, tok = two_stage_train(base, train, extra if len(extra) else None)
p_lora = predict_proba(model, tok, test["text"].astype(str).tolist(), batch_size=4)
# map LoRA columns (LABELS order)
assert p_lora.shape[1] == len(LABELS)

probs = blend_and_penalize(
    p_lora,
    p_tfidf_aligned,
    test["text"].astype(str).tolist(),
    train["text"].astype(str).tolist(),
    train["model"].astype(str).tolist(),
    lora_weight=0.78,
    rebalance=True,
)
pred = [LABELS[i] for i in probs.argmax(axis=1)]
out_path = Path("submission.parquet")
write_submission(test["id"].astype(str), pred, out_path)
check = pd.read_parquet(out_path)
assert list(check.columns) == ["id", "model"]
assert check["id"].nunique() == len(test)
assert set(check["model"]).issubset(set(LABELS))
print(check.head())
print("wrote", out_path.resolve(), "rows", len(check), "extra", len(extra))
print(json.dumps({"n_train": len(train), "n_extra": int(len(extra)), "n_pred": len(check)}, indent=2))
'''
    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
    }
    nb.cells = [
        new_markdown_cell(
            """# Name That LLM — Qwen LoRA + TF-IDF (GPU)

Two-stage LoRA on **Qwen3-8B-Base** (4-bit), then blend with the CPU TF-IDF model.

**Runtime settings (required)**

- Internet: **Off**
- Accelerator: **GPU T4 x2**
- Persistence: files only
- Time limit: 9 hours (training + ~4000-row rerun)

**Inputs to attach**

1. Competition data: `name-that-llm`
2. Kaggle Model: `qwen-lm/qwen-3` → Transformers → **8b-base** / 1  
   (Apache 2.0; path usually `/kaggle/input/qwen-3/transformers/8b-base/1`)
3. Extra public passages (at least one; more is better):
   - `nishantdahal/ntl-two-stage-lora` — use **`pool_v1.parquet` only** (CC BY-SA 4.0 public pool). Adapters in that dataset are ignored.
   - `prvsiyan/prvsiyans-extended-name-that-llm-dataset`
   - `seyitkaangunes/ntl-extra-api-passages`
   - `willrice/name-that-llm-openrouter-harvest`

Phone verification is required on Kaggle before GPU sessions.

Do not copy other teams' LoRA checkpoints. This notebook trains its own adapters on the attached public text.
"""
        ),
        new_code_cell(constants),
        new_code_cell(features),
        new_code_cell(pipeline),
        new_code_cell(extra),
        new_code_cell(lora),
        new_code_cell(driver.strip() + "\n"),
    ]
    dest = ROOT / "name_that_llm_gpu_submit.ipynb"
    dest.write_text(nbf.writes(nb))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
