"""Public extra-passage discovery and host-style cleaning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from .constants import LABELS
from .pipeline import load_table, normalize_frame

SKIP_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".npz",
    ".gguf",
    ".msgpack",
}
SKIP_DIR_MARKERS = (
    "lora8b",
    "lora14b",
    "/adapter/",
    "\\adapter\\",
    "tokenizer/",
)
MD_RE = re.compile(r"```|^\s{0,3}#{1,6}\s|^\s*[-*]\s", re.M)
WORD_RE = re.compile(r"\S+")


def _is_weight_or_adapter(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    low = str(path).lower()
    return any(m in low for m in SKIP_DIR_MARKERS)


def discover_extra_files(roots: Sequence[str | Path]) -> list[Path]:
    skip_names = {
        "train.csv",
        "test.csv",
        "sample_submission.parquet",
        "sample_submission.csv",
        "synth_train.csv",
        "synth_test.csv",
    }
    found: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        paths = [root_path] if root_path.is_file() else list(root_path.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            if path.name in skip_names or _is_weight_or_adapter(path):
                continue
            try:
                if path.stat().st_size > 120_000_000:
                    continue
            except OSError:
                continue
            found.append(path)
    return found


def illegal_marks_from_train(train: pd.DataFrame) -> dict[str, set[str]]:
    illegal: dict[str, set[str]] = {"emdash": set(), "curly": set()}
    if train is None or train.empty:
        return illegal
    for model, g in train.groupby("model"):
        if not g["text"].str.contains("—", regex=False).any():
            illegal["emdash"].add(str(model))
        if not g["text"].str.contains(r"[“”‘’]", regex=True).any():
            illegal["curly"].add(str(model))
    return illegal


def clean_extra(df: pd.DataFrame, train: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model"])
    out = df.copy()
    out = out[out["model"].isin(LABELS)]
    words = out["text"].map(lambda t: len(WORD_RE.findall(str(t))))
    out = out[(words >= 30) & (words <= 300)]
    out = out[~out["text"].map(lambda t: bool(MD_RE.search(str(t))))]
    illegal = illegal_marks_from_train(train) if train is not None else {"emdash": set(), "curly": set()}
    if illegal["emdash"]:
        drop = out["model"].isin(illegal["emdash"]) & out["text"].str.contains("—", regex=False)
        out = out[~drop]
    if illegal["curly"]:
        drop = out["model"].isin(illegal["curly"]) & out["text"].str.contains(r"[“”‘’]", regex=True)
        out = out[~drop]
    out = out.drop_duplicates(subset=["text"])
    return out.reset_index(drop=True)


def subsample_extra(df: pd.DataFrame, max_rows: int, seed: int = 0) -> pd.DataFrame:
    if df is None or len(df) <= max_rows:
        return df.reset_index(drop=True) if df is not None else df
    parts = []
    n_lab = df["model"].nunique() or 1
    per = max(max_rows // n_lab, 1)
    rng = seed
    for _, g in df.groupby("model"):
        take = min(len(g), per)
        parts.append(g.sample(n=take, random_state=rng) if take < len(g) else g)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out.reset_index(drop=True)


def load_and_clean_extra(
    roots: Sequence[str | Path],
    train: pd.DataFrame,
    test_ids: Sequence[str] | None = None,
    max_rows: int = 16_000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    train_text = set(train["text"].astype(str)) if train is not None else set()
    banned_ids = set(map(str, test_ids or []))
    for path in discover_extra_files(roots):
        try:
            raw = normalize_frame(load_table(path))
        except Exception:
            continue
        if "model" not in raw.columns:
            continue
        frames.append(raw)
    if not frames:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model"])
    extra = pd.concat(frames, ignore_index=True)
    extra = extra[~extra["text"].isin(train_text)]
    if banned_ids:
        extra = extra[~extra["id"].isin(banned_ids)]
    extra = clean_extra(extra, train)
    extra = subsample_extra(extra, max_rows=max_rows)
    return extra
