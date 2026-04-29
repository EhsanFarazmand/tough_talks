#!/usr/bin/env python3
"""
Validate all JSON schemas in data/schemas/ are well-formed.
Run: python scripts/validate_schemas.py
"""

import json
import sys
from pathlib import Path

SCHEMAS_DIR = Path(__file__).parent.parent / "data" / "schemas"

def validate_schemas():
    schemas = list(SCHEMAS_DIR.glob("*.schema.json"))
    if not schemas:
        print("No schemas found in data/schemas/")
        return

    errors = []
    for schema_path in schemas:
        try:
            data = json.loads(schema_path.read_text())
            # Basic checks
            assert "$schema" in data, "Missing $schema"
            assert "title" in data, "Missing title"
            assert "type" in data, "Missing type"
            print(f"  ✅  {schema_path.name}")
        except (json.JSONDecodeError, AssertionError, Exception) as e:
            errors.append(f"{schema_path.name}: {e}")
            print(f"  ❌  {schema_path.name}: {e}")

    if errors:
        print(f"\n{len(errors)} schema(s) failed validation.")
        sys.exit(1)
    else:
        print(f"\nAll {len(schemas)} schemas valid.")


if __name__ == "__main__":
    print("Validating schemas...")
    validate_schemas()
