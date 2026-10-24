"""Competition labels, families, and closed-class word lists."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
)

FAMILY_OF: dict[str, str] = {}
for _name in LABELS:
    if _name.startswith("gpt-"):
        FAMILY_OF[_name] = "openai"
    elif _name.startswith("gemini-"):
        FAMILY_OF[_name] = "google"
    elif _name.startswith("claude-"):
        FAMILY_OF[_name] = "anthropic"
    elif _name.startswith("qwen"):
        FAMILY_OF[_name] = "qwen"
    elif _name.startswith("glm-"):
        FAMILY_OF[_name] = "zai"
    elif _name.startswith("kimi-"):
        FAMILY_OF[_name] = "moonshot"
    else:
        FAMILY_OF[_name] = "other"

FAMILIES: tuple[str, ...] = (
    "openai",
    "google",
    "anthropic",
    "qwen",
    "zai",
    "moonshot",
)

# Marks that some labs almost never produce in the host training set.
# Used as soft probability penalties, never as hard bans.
MARK_PATTERNS: dict[str, str] = {
    "emdash": r"—",
    "endash": r"–",
    "curly_dq": r"[“”]",
    "curly_sq": r"[‘’]",
    "ellipsis_uni": r"…",
    "excl": r"!",
    "contraction": r"(?i)\b(?:['’]t|['’]s|['’]re|['’]ve|['’]ll|['’]d|n't)\b",
}

FUNCTION_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while although though because as at by for from in into
    of on to with without within through during before after above below between
    among against about across around until unless whether
    this that these those i you he she it we they me him her us them my your his
    its our their mine yours hers ours theirs
    is are was were be been being do does did have has had can could may might
    must shall should will would
    not no nor so than then there here when where what which who whom whose how
    why whom
    also just only even still yet already often never always perhaps maybe
    however therefore thus hence moreover furthermore instead rather quite
    very much many more most less least such own same other another each every
    both few several some any all
    """.split()
)

STYLE_MARKERS: tuple[str, ...] = (
    "however",
    "therefore",
    "thus",
    "hence",
    "moreover",
    "furthermore",
    "notably",
    "indeed",
    "perhaps",
    "simply",
    "essentially",
    "fundamentally",
    "delve",
    "tapestry",
    "underscore",
    "crucial",
    "vital",
    "intricate",
    "landscape",
    "nuanced",
    "robust",
    "leverage",
    "not only",
    "in other words",
    "that is",
    "for instance",
    "such as",
    "in contrast",
    "on the other hand",
    "as a result",
    "in fact",
    "at its core",
    "the key",
)

KAGGLE_TRAIN_CANDIDATES: tuple[str, ...] = (
    "/kaggle/input/competitions/name-that-llm/train.csv",
    "/kaggle/input/name-that-llm/train.csv",
)

KAGGLE_TEST_CANDIDATES: tuple[str, ...] = (
    "/kaggle/input/competitions/name-that-llm/test.csv",
    "/kaggle/input/name-that-llm/test.csv",
)
