"""Loader for prompt templates stored under ``data/prompts/``.

Each prompt is a markdown file with ``$placeholder`` slots
(``string.Template`` syntax). We use ``string.Template`` rather than
``str.format`` because prompts contain JSON examples with literal
``{`` / ``}`` braces that would otherwise need awkward double-escaping.

Usage:

    from backend.core._runtime import load_prompt

    persona = load_prompt("persona")
    system_prompt = persona.render(
        persona_name="David",
        profile="Deflects with budget language ...",
        user_goal="Ask for a 15% raise.",
    )
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data" / "prompts"

# Match $name or ${name}. The negative lookbehind skips the $$ escape
# (string.Template's literal-dollar marker).
_PLACEHOLDER_RE = re.compile(r"(?<!\$)\$(?:\{(\w+)\}|(\w+))")


class PromptError(ValueError):
    """Raised when a prompt is missing on disk or rendered with bad inputs."""


@dataclass(frozen=True)
class Prompt:
    name: str
    template: str
    placeholders: frozenset[str]

    def render(self, **values: object) -> str:
        """Substitute placeholders. Raises ``PromptError`` if required
        placeholders are missing or unexpected ones are supplied."""
        provided = set(values)
        missing = self.placeholders - provided
        extra = provided - self.placeholders
        if missing:
            raise PromptError(
                f"prompt '{self.name}' missing placeholders: {sorted(missing)}"
            )
        if extra:
            raise PromptError(
                f"prompt '{self.name}' got unexpected placeholders: {sorted(extra)}"
            )
        return string.Template(self.template).substitute(**values)


def _extract_placeholders(template: str) -> frozenset[str]:
    return frozenset(
        match for groups in _PLACEHOLDER_RE.findall(template) for match in groups if match
    )


def load_prompt(name: str, *, prompts_dir: Optional[Path] = None) -> Prompt:
    """Load a prompt template from ``data/prompts/<name>.md``."""
    directory = prompts_dir or DEFAULT_PROMPTS_DIR
    path = directory / f"{name}.md"
    if not path.is_file():
        raise PromptError(f"Prompt not found: {path}")
    template = path.read_text(encoding="utf-8")
    return Prompt(
        name=name,
        template=template,
        placeholders=_extract_placeholders(template),
    )
