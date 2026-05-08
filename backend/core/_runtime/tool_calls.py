"""Parser for Gemma 4's native tool-call output.

When ``chat()`` is called with ``tools=...``, the chat template
declares the available tools to the model and the model responds with
zero or more tool-call blocks of the form (transcribed from the
template):

    <|tool_call>call:tool_name{key1:value1,key2:value2,...}<tool_call|>

Where:
  - keys are unquoted identifiers
  - string values are wrapped in ``<|"|>...<|"|>`` markers
  - bools are ``true`` / ``false``
  - sequences are ``[item,item,...]``
  - nested objects use the same brace-comma syntax

This module converts that format to JSON, then parses it. Returns one
``{"name": str, "arguments": dict}`` per call in emission order.
"""

from __future__ import annotations

import json
from typing import Any

_TOOL_CALL_PREFIX = "<|tool_call>call:"
_TOOL_CALL_SUFFIX = "<tool_call|>"
_STRING_MARKER = '<|"|>'


class ToolCallParseError(ValueError):
    """A tool-call block was found but its argument body could not be
    converted to a Python dict."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def _balanced_brace_end(text: str, start: int) -> int:
    """Return the index of the ``}`` that closes the ``{`` at ``start``,
    or -1 if no balanced match exists. Tracks JSON-style ``"…"`` strings
    so braces inside string values do not corrupt the depth counter.
    """
    if start >= len(text) or text[start] != "{":
        return -1
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
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _quote_bare_keys(body: str) -> str:
    """Convert ``key:value`` (unquoted key) to ``"key":value``.

    Walks the input character-by-character, tracking JSON string state
    so identifiers inside string values are left alone.
    """
    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(body):
        ch = body[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(body) and (body[j].isalnum() or body[j] == "_"):
                j += 1
            ident = body[i:j]
            k = j
            while k < len(body) and body[k] in " \t":
                k += 1
            if (
                k < len(body)
                and body[k] == ":"
                and ident not in ("true", "false", "null")
            ):
                out.append(f'"{ident}"')
            else:
                out.append(ident)
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _arguments_to_dict(body: str) -> dict:
    """Translate one tool-call argument body into a Python dict."""
    body = body.replace(_STRING_MARKER, '"')
    body = _quote_bare_keys(body)
    try:
        return json.loads("{" + body + "}")
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(
            f"Could not parse tool-call arguments: {exc}",
            raw=body,
        ) from exc


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract every tool-call block from a model response.

    Returns a list of ``{"name": str, "arguments": dict}`` dicts in
    emission order. Returns ``[]`` when the model emitted no tool
    calls. Raises ``ToolCallParseError`` if a block is found but its
    argument body is malformed — that's a real model failure worth
    surfacing rather than swallowing.
    """
    calls: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find(_TOOL_CALL_PREFIX, pos)
        if start < 0:
            break
        name_start = start + len(_TOOL_CALL_PREFIX)
        brace = text.find("{", name_start)
        if brace < 0:
            break
        name = text[name_start:brace].strip()
        end = _balanced_brace_end(text, brace)
        if end < 0:
            break
        suffix_start = end + 1
        if not text.startswith(_TOOL_CALL_SUFFIX, suffix_start):
            pos = end + 1
            continue
        args_body = text[brace + 1 : end]
        arguments = _arguments_to_dict(args_body)
        calls.append({"name": name, "arguments": arguments})
        pos = suffix_start + len(_TOOL_CALL_SUFFIX)
    return calls
