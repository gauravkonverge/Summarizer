"""Heuristic detection of garbled/unreadable message content.

Catches corrupted or unreadable text (e.g. "@@@@ #### NULL ERROR",
"vehicle ???? status ????") before it reaches the summarization prompt.
"""

import re

_WHITELISTED_PUNCTUATION = set(".,!?;:'\"-()%&$")
_NOISE_TOKEN_RE = re.compile(r"^[^a-zA-Z0-9]+$")
_REPLACEMENT_CHAR = "�"

# 3+ repeats of any symbol other than "!"/"?" (e.g. "####", "%%%%", "@@@@").
_REPEATED_SYMBOL_RE = re.compile(r"([^\w\s!?])\1{2,}")
# 4+ repeats of "!"/"?" specifically, so ordinary emphasis like "wait!!!" isn't flagged.
_REPEATED_EMPHASIS_RE = re.compile(r"([!?])\1{3,}")

_NOISE_TOKEN_RATIO_THRESHOLD = 0.5
_SYMBOL_RATIO_THRESHOLD = 0.4
_MIN_RESIDUAL_LETTERS = 3


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbol_chars = sum(
        1
        for char in text
        if not char.isalnum() and not char.isspace() and char not in _WHITELISTED_PUNCTUATION
    )
    return symbol_chars / len(text)


def _noise_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    noise_tokens = sum(1 for token in tokens if _NOISE_TOKEN_RE.match(token))
    return noise_tokens / len(tokens)


def _residual_letter_count(tokens: list[str]) -> int:
    return sum(
        char.isalpha()
        for token in tokens
        if not _NOISE_TOKEN_RE.match(token)
        for char in token
    )


def assess_garbled(text: str) -> tuple[bool, str | None]:
    """Return (is_garbled, reason) for corrupted/unreadable message content.

    Order matters: cheaper, higher-signal checks run first.
    """
    if _REPLACEMENT_CHAR in text:
        return True, "contains unicode replacement characters (encoding failure)"

    match = _REPEATED_SYMBOL_RE.search(text) or _REPEATED_EMPHASIS_RE.search(text)
    if match:
        return True, f"repeated symbol run ({match.group(0)!r})"

    tokens = text.split()

    noise_ratio = _noise_token_ratio(tokens)
    if noise_ratio >= _NOISE_TOKEN_RATIO_THRESHOLD:
        return True, f"{noise_ratio:.0%} of tokens are pure symbol noise"

    symbol_ratio = _symbol_ratio(text)
    if symbol_ratio >= _SYMBOL_RATIO_THRESHOLD:
        return True, f"{symbol_ratio:.0%} of characters are non-alphanumeric symbols"

    if _residual_letter_count(tokens) < _MIN_RESIDUAL_LETTERS:
        return True, "fewer than 3 readable letters after removing noise tokens"

    return False, None
