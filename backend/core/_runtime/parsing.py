"""JSON extraction and repair for noisy LLM output.

The chat helper returns raw text. ``parse_json`` is the single point that
turns that text into a Python ``dict``/``list``. It handles the four
failure modes most common with small instruction-tuned models:

1. Prose before or after the JSON block ("Here you go: { ... }").
2. Markdown code fences (``` or ```json) with stray whitespace.
3. Trailing commas before ``}`` / ``]``.
4. Smart quotes (curly U+201C/U+201D/U+2018/U+2019) instead of straight.

Strategy: try the whole stripped string first, then extract the first
balanced ``{...}`` / ``[...]`` (string-aware so braces inside strings do
not corrupt the depth counter), then run a repair pass, then optionally
defer to a caller-supplied repair callback (typically a model call).

On total failure raises ``JsonParseError`` with the raw output and every
attempted candidate, so the notebook can show useful diagnostics without
hiding the underlying text.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional, Union

JsonValue = Union[dict, list]

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*|```", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_SMART_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


class JsonParseError(ValueError):
    """Raised when JSON extraction and every repair pass have failed."""

    def __init__(self, message: str, raw: str, attempts: list[str]):
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_balanced(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` or ``[...]`` substring.

    Tracks string boundaries (including escaped quotes) so braces inside
    JSON string values do not throw the depth counter off.
    """
    start = -1
    open_char = ""
    close_char = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            open_char = ch
            close_char = "}" if ch == "{" else "]"
            break
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair(text: str) -> str:
    text = text.translate(_SMART_QUOTES)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def parse_json(
    raw: str,
    *,
    repair_callback: Optional[Callable[[str], str]] = None,
) -> JsonValue:
    """Extract and parse a JSON object/array from a noisy model response.

    Tries in order:
      1. Strip fences and parse the whole stripped string.
      2. Extract the first balanced ``{...}`` / ``[...]`` and parse it.
      3. Apply the repair pass (smart quotes, trailing commas) and retry.
      4. If a ``repair_callback`` is given, hand it the raw text, parse
         the result. Useful for model-assisted repair: pass a closure
         that re-prompts the model with the bad output and asks for
         valid JSON.

    Raises ``JsonParseError`` only after every attempt has failed.
    """
    attempts: list[str] = []
    stripped = _strip_fences(raw)

    attempts.append(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    extracted = _extract_balanced(stripped)
    if extracted is not None:
        attempts.append(extracted)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    candidate = extracted if extracted is not None else stripped
    repaired = _repair(candidate)
    if repaired != candidate:
        attempts.append(repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    if repair_callback is not None:
        try:
            fixed = repair_callback(raw)
        except Exception as exc:  # noqa: BLE001
            raise JsonParseError(
                f"repair_callback raised: {exc}",
                raw=raw,
                attempts=attempts,
            ) from exc
        attempts.append(fixed)
        try:
            return parse_json(fixed)
        except JsonParseError as exc:
            raise JsonParseError(
                "Could not extract valid JSON from model output (repair_callback also failed)",
                raw=raw,
                attempts=attempts + exc.attempts,
            ) from exc

    raise JsonParseError(
        "Could not extract valid JSON from model output",
        raw=raw,
        attempts=attempts,
    )


def try_parse_json(raw: str) -> tuple[Optional[JsonValue], Optional[JsonParseError]]:
    """Non-raising variant. Returns ``(value, None)`` on success or
    ``(None, error)`` on failure. Convenient for notebook cells that
    want to render a results table even when one parse failed.
    """
    try:
        return parse_json(raw), None
    except JsonParseError as exc:
        return None, exc
