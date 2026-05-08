"""Loader for tool/function-call definitions stored under
``data/schemas/tools/``.

Each tool lives in its own ``<name>.schema.json`` file so it can be
edited, version-controlled, and validated independently. The on-disk
format is a JSON Schema document with three extra top-level keys —
``name``, ``description``, ``parameters`` — that carry the function
definition the model is shown.

``load_tools()`` returns the OpenAI function-tool list, which is the
shape Gemma 4's chat template expects in its ``tools=`` argument:

    [
      {
        "type": "function",
        "function": {"name": ..., "description": ..., "parameters": {...}},
      },
      ...
    ]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOLS_DIR = REPO_ROOT / "data" / "schemas" / "tools"

REQUIRED_TOOL_KEYS = ("name", "description", "parameters")


class ToolSchemaError(ValueError):
    """Raised when a tool schema file is missing or malformed."""


def _read_tool_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolSchemaError(f"{path.name}: invalid JSON ({exc})") from exc
    missing = [k for k in REQUIRED_TOOL_KEYS if k not in data]
    if missing:
        raise ToolSchemaError(
            f"{path.name}: missing required tool keys: {missing}"
        )
    function = {k: data[k] for k in REQUIRED_TOOL_KEYS}
    return {"type": "function", "function": function}


def load_tools(
    names: Optional[Iterable[str]] = None,
    *,
    tools_dir: Optional[Path] = None,
) -> list[dict]:
    """Load tool definitions from disk.

    With ``names=None`` returns every tool found in ``tools_dir`` sorted
    by filename. With an explicit list, returns the tools in that order
    and raises ``ToolSchemaError`` for any name that has no matching
    file.
    """
    directory = tools_dir or DEFAULT_TOOLS_DIR
    if not directory.is_dir():
        raise ToolSchemaError(f"Tools directory not found: {directory}")

    if names is None:
        files = sorted(directory.glob("*.schema.json"))
        return [_read_tool_file(p) for p in files]

    tools: list[dict] = []
    for name in names:
        path = directory / f"{name}.schema.json"
        if not path.is_file():
            raise ToolSchemaError(
                f"Tool schema not found for '{name}': expected {path}"
            )
        tools.append(_read_tool_file(path))
    return tools
