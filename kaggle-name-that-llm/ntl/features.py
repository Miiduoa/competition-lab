"""Stylometric dense features and text views for n-gram vectorizers."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from .constants import FUNCTION_WORDS, MARK_PATTERNS, STYLE_MARKERS

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+")
_SENT_RE = re.compile(r"[.!?]+")
_OXFORD_RE = re.compile(r",\s+(?:and|or)\s+", re.I)
_HYPHEN_WORD_RE = re.compile(r"\b\w+-\w+\b")
_LY_RE = re.compile(r"\b\w+ly\b", re.I)
_DIGIT_RE = re.compile(r"\d")
_UPPER_RE = re.compile(r"[A-Z]")
_LETTER_RE = re.compile(r"[A-Za-z]")
_CONTRACTION_RE = re.compile(
    r"(?i)\b\w+(?:n't|['’]t|['’]s|['’]re|['’]ve|['’]ll|['’]d)\b"
)
_START_RES = {
    "start_the": re.compile(r"^The\b"),
    "start_in": re.compile(r"^In\b"),
    "start_imagine": re.compile(r"^Imagine\b"),
    "start_when": re.compile(r"^When\b"),
    "start_this": re.compile(r"^This\b"),
    "start_a": re.compile(r"^A\b"),
    "start_as": re.compile(r"^As\b"),
}
_PRONOUN_RES = {
    "you": re.compile(r"\byou\b", re.I),
    "we": re.compile(r"\bwe\b", re.I),
    "it": re.compile(r"\bit\b", re.I),
    "one": re.compile(r"\bone\b", re.I),
}
_MARKER_RES = tuple((m, re.compile(re.escape(m), re.I)) for m in STYLE_MARKERS)
_MARK_RES = {name: re.compile(pat) for name, pat in MARK_PATTERNS.items()}


def to_word_shape(text: str) -> str:
    parts: list[str] = []
    for tok in re.findall(r"\S+", text):
        chars: list[str] = []
        for ch in tok:
            if ch.isupper():
                chars.append("X")
            elif ch.islower():
                chars.append("x")
            elif ch.isdigit():
                chars.append("d")
            else:
                chars.append(ch)
        shape = "".join(chars)
        shape = re.sub(r"x{4,}", "x+", shape)
        shape = re.sub(r"X{4,}", "X+", shape)
        shape = re.sub(r"d{3,}", "d+", shape)
        parts.append(shape)
    return " ".join(parts)


def to_func_view(text: str) -> str:
    toks = re.findall(r"[A-Za-z]+|[^\w\s]", text)
    out: list[str] = []
    for tok in toks:
        low = tok.lower()
        if low in FUNCTION_WORDS:
            out.append(low)
        elif tok.isalnum():
            out.append("CONTENT")
        else:
            out.append(tok)
    return " ".join(out)


def to_punct_view(text: str) -> str:
    chars: list[str] = []
    prev_space = False
    for ch in text:
        if ch.isalnum():
            continue
        if ch.isspace():
            if not prev_space:
                chars.append(" ")
            prev_space = True
            continue
        chars.append(ch)
        prev_space = False
    return "".join(chars).strip()


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def dense_feature_names() -> list[str]:
    names = [
        "n_chars",
        "n_words",
        "n_sents",
        "avg_word_len",
        "std_word_len",
        "avg_sent_len",
        "ttr",
        "long_word_rate",
        "short_word_rate",
        "comma_rate",
        "semi_rate",
        "colon_rate",
        "emdash_rate",
        "endash_rate",
        "hyphen_rate",
        "excl_rate",
        "qmark_rate",
        "paren_rate",
        "curly_q_rate",
        "straight_q_rate",
        "apos_curly_rate",
        "ellipsis_rate",
        "slash_rate",
        "oxford_rate",
        "hyphen_word_rate",
        "contraction_rate",
        "ly_rate",
        "digit_rate",
        "upper_rate",
        "comma_per_sent",
        "unique_punct",
        "ends_period",
        "has_emdash",
        "has_curly",
        "has_contraction",
    ]
    names.extend(_START_RES.keys())
    names.extend(f"pron_{k}" for k in _PRONOUN_RES)
    names.extend(f"mark_{m.replace(' ', '_')}" for m, _ in _MARKER_RES)
    names.extend(f"has_{name}" for name in MARK_PATTERNS)
    return names


def extract_dense_row(text: str) -> list[float]:
    n_chars = max(len(text), 1)
    words = _WORD_RE.findall(text)
    n_words = len(words)
    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    n_sents = max(len(sents), 1)
    lengths = [len(w) for w in words] or [0]
    avg_w = float(np.mean(lengths))
    std_w = float(np.std(lengths))
    lower_words = [w.lower() for w in words]
    ttr = _safe_div(len(set(lower_words)), n_words)
    long_rate = _safe_div(sum(1 for L in lengths if L >= 10), n_words)
    short_rate = _safe_div(sum(1 for L in lengths if L <= 3), n_words)

    def rate(count: int) -> float:
        return 1000.0 * count / n_chars

    row = [
        float(n_chars),
        float(n_words),
        float(n_sents),
        avg_w,
        std_w,
        _safe_div(n_words, n_sents),
        ttr,
        long_rate,
        short_rate,
        rate(text.count(",")),
        rate(text.count(";")),
        rate(text.count(":")),
        rate(text.count("—")),
        rate(text.count("–")),
        rate(text.count("-")),
        rate(text.count("!")),
        rate(text.count("?")),
        rate(text.count("(") + text.count(")")),
        rate(len(re.findall(r"[“”‘’]", text))),
        rate(text.count('"') + text.count("'")),
        rate(text.count("’")),
        rate(text.count("…") + text.count("...")),
        rate(text.count("/") + text.count("\\")),
        rate(len(_OXFORD_RE.findall(text))),
        _safe_div(len(_HYPHEN_WORD_RE.findall(text)), n_words),
        _safe_div(len(_CONTRACTION_RE.findall(text)), n_words),
        _safe_div(len(_LY_RE.findall(text)), n_words),
        rate(len(_DIGIT_RE.findall(text))),
        _safe_div(len(_UPPER_RE.findall(text)), max(len(_LETTER_RE.findall(text)), 1)),
        _safe_div(text.count(","), n_sents),
        float(len(set(ch for ch in text if not ch.isalnum() and not ch.isspace()))),
        1.0 if text.rstrip().endswith(".") else 0.0,
        1.0 if "—" in text else 0.0,
        1.0 if re.search(r"[“”‘’]", text) else 0.0,
        1.0 if _CONTRACTION_RE.search(text) else 0.0,
    ]
    for rx in _START_RES.values():
        row.append(1.0 if rx.match(text) else 0.0)
    for rx in _PRONOUN_RES.values():
        row.append(_safe_div(len(rx.findall(text)), n_words))
    for _, rx in _MARKER_RES:
        row.append(float(len(rx.findall(text))))
    for rx in _MARK_RES.values():
        row.append(1.0 if rx.search(text) else 0.0)
    return row


def extract_dense(texts: Iterable[str]) -> np.ndarray:
    return np.asarray([extract_dense_row(t) for t in texts], dtype=np.float32)


def mark_hits(text: str) -> dict[str, bool]:
    return {name: bool(rx.search(text)) for name, rx in _MARK_RES.items()}
