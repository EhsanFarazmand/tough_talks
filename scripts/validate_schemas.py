#!/usr/bin/env python3
"""Validate every ``*.schema.json`` under ``data/schemas/``.

Two layers of checks:

1. **Generic**: every schema file must have ``$schema``, ``title``, ``type``.
2. **Tool-shape**: any schema under ``data/schemas/tools/`` must additionally
   carry top-level ``name``, ``description``, ``parameters`` keys (the
   OpenAI function-tool fields the runtime's ``load_tools`` reads) and
   ``parameters`` must itself be an object with ``properties`` and
   ``required``.

Exit code is non-zero on any failure so CI can fail loud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMAS_DIR = Path(__file__).parent.parent / "data" / "schemas"
TOOLS_DIR = SCHEMAS_DIR / "tools"

GENERIC_REQUIRED = ("$schema", "title", "type")
TOOL_REQUIRED = ("name", "description", "parameters")


def _check_generic(data: dict) -> list[str]:
    return [f"missing top-level field '{k}'" for k in GENERIC_REQUIRED if k not in data]


def _check_tool_shape(data: dict) -> list[str]:
    errors = [f"missing tool field '{k}'" for k in TOOL_REQUIRED if k not in data]
    params = data.get("parameters")
    if isinstance(params, dict):
        if params.get("type") != "object":
            errors.append("parameters.type must be 'object'")
        if "properties" not in params:
            errors.append("parameters.properties is required")
        if "required" not in params:
            errors.append("parameters.required is required (use [] if none)")
    elif "parameters" in data:
        errors.append("parameters must be an object")
    return errors


def _is_tool_schema(path: Path) -> bool:
    try:
        return path.parent == TOOLS_DIR or TOOLS_DIR in path.parents
    except ValueError:
        return False


def validate_schemas() -> int:
    schemas = sorted(SCHEMAS_DIR.rglob("*.schema.json"))
    if not schemas:
        print("No schemas found in data/schemas/")
        return 0

    errors: list[str] = []
    for schema_path in schemas:
        rel = schema_path.relative_to(SCHEMAS_DIR)
        try:
            data = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{rel}: invalid JSON ({exc})")
            print(f"  FAIL  {rel}: invalid JSON ({exc})")
            continue

        problems = _check_generic(data)
        if _is_tool_schema(schema_path):
            problems.extend(_check_tool_shape(data))

        if problems:
            for p in problems:
                errors.append(f"{rel}: {p}")
                print(f"  FAIL  {rel}: {p}")
        else:
            print(f"  OK    {rel}")

    print()
    if errors:
        print(f"{len(errors)} schema problem(s) found across {len(schemas)} file(s).")
        return 1
    print(f"All {len(schemas)} schemas valid.")
    return 0


if __name__ == "__main__":
    print("Validating schemas under", SCHEMAS_DIR)
    sys.exit(validate_schemas())
