"""Pydantic request envelopes for the Tough Talks API.

Phase 5 / Step 12. These are deliberately thin — the JSON schemas under
``data/schemas/`` are the source of truth for OUTPUT shapes, and each
runtime ``XxxConfig`` dataclass is the source of truth for INPUT shape.
The Pydantic models here just expose those config fields to FastAPI so
the framework can:

* validate request bodies and surface field-level errors before any
  runtime call;
* render the OpenAPI documentation at ``/docs``.

Nested artifacts that the runtime treats as opaque dicts
(``transcript``, ``person_profile``, ``talk_dna_profile``, ``prior_pulse``,
etc.) are typed as ``dict[str, Any]`` here — the runtime's
``_coerce_*_payload`` and ``format_*_block`` helpers already handle the
shape validation and the system-contract enforcement. Re-typing them in
Pydantic would duplicate the JSON schemas without adding signal and would
break the moment a schema evolves.

The response models are NOT defined — every route returns the runtime
output dict directly (FastAPI serialises it). The dicts already match
``data/schemas/<component>.schema.json``.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

__all__ = [
    "AftermathRequest",
    "DebriefRequest",
    "PersonVaultRequest",
    "PersonaReplyRequest",
    "PersonaRunRequest",
    "PremortemRequest",
    "PulseRequest",
    "TalkDNARequest",
]


# ---------------------------------------------------------------------------
# Text-only routes
# ---------------------------------------------------------------------------


class TalkDNARequest(BaseModel):
    """Input envelope for ``POST /talk-dna/analyze``.

    Mirrors :class:`backend.core._runtime.TalkDNAConfig` plus the
    ``turns`` list (which the runtime takes as a positional arg).
    """

    turns: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Transcript turns. Each entry must at least carry "
            "``speaker`` (`user` / `other`) and ``text``; optional "
            "``emotion`` and timing fields are surfaced when present."
        ),
        min_length=1,
    )
    user_id: str = "local"
    max_new_tokens: int = 768
    prior_profile: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Existing TalkDNA dict for incremental v1 → v2 updates. "
            "Pass it to bump version and accumulate qualitative signal."
        ),
    )
    prompt_name: str = "talk_dna"


class PersonVaultRequest(BaseModel):
    """Input envelope for ``POST /vault/build``."""

    turns: list[dict[str, Any]] = Field(..., min_length=1)
    name: str = "Other"
    relationship_type: Optional[str] = Field(
        default=None,
        description=(
            "Enum value from ``ALLOWED_RELATIONSHIP_TYPES``. The runtime "
            "validates and falls back to ``other`` if unrecognised."
        ),
    )
    person_id: Optional[str] = None
    max_new_tokens: int = 768
    prior_profile: Optional[dict[str, Any]] = None
    prompt_name: str = "person_vault"


class PersonaReplyRequest(BaseModel):
    """Input envelope for ``POST /persona/reply`` — one in-character turn."""

    user_message: str = Field(..., min_length=1)
    persona_profile: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full PersonVault dict (Step 6 output) or its inner ``profile`` "
            "block. The runtime extracts ``name`` / ``relationship_type`` "
            "from the wrapper when present."
        ),
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Prior turns of this practice round. Each entry is "
            "``{speaker: 'user', text}`` or a persona reply dict in the "
            "shape the runtime emits."
        ),
    )
    user_goal: str = "Have a constructive conversation about the open issue."
    max_new_tokens: int = 384
    enable_thinking: bool = False
    prompt_name: str = "persona"


class PersonaRunRequest(BaseModel):
    """Input envelope for ``POST /persona/run`` — full multi-turn round.

    The runtime processes every ``user_message`` in order, feeding back
    the rolling history between turns. Returns the final alternating
    user/persona history.
    """

    user_messages: list[str] = Field(..., min_length=1)
    persona_profile: dict[str, Any] = Field(default_factory=dict)
    user_goal: str = "Have a constructive conversation about the open issue."
    max_new_tokens: int = 384
    enable_thinking: bool = False
    prompt_name: str = "persona"


class PremortemRequest(BaseModel):
    """Input envelope for ``POST /premortem`` — three failure scenarios."""

    conversation_description: str = Field(..., min_length=1)
    user_goal: str = "Have a constructive conversation about the open issue."
    person_profile: dict[str, Any] = Field(default_factory=dict)
    talk_dna_profile: dict[str, Any] = Field(default_factory=dict)
    max_new_tokens: int = 768
    enable_thinking: bool = False
    prompt_name: str = "premortem"


class DebriefRequest(BaseModel):
    """Input envelope for ``POST /debrief`` — post-round coaching."""

    transcript: list[dict[str, Any]] = Field(..., min_length=1)
    user_goal: str = "Have a constructive conversation about the open issue."
    person_profile: dict[str, Any] = Field(default_factory=dict)
    talk_dna_profile: dict[str, Any] = Field(default_factory=dict)
    max_new_tokens: int = 768
    enable_thinking: bool = False
    prompt_name: str = "debrief"


class AftermathRequest(BaseModel):
    """Input envelope for ``POST /aftermath`` — plan vs. reality."""

    premortem: dict[str, Any] = Field(
        ...,
        description="Pre-mortem dict from /premortem (or just the inner failure_scenarios list).",
    )
    transcript: list[dict[str, Any]] = Field(..., min_length=1)
    user_goal: str = "Have a constructive conversation about the open issue."
    person_profile: dict[str, Any] = Field(default_factory=dict)
    debrief: dict[str, Any] = Field(default_factory=dict)
    max_new_tokens: int = 1024
    enable_thinking: bool = False
    prompt_name: str = "aftermath"


class PulseRequest(BaseModel):
    """Input envelope for ``POST /pulse`` — relationship pulse across rounds.

    ``rounds`` must hold at least :data:`ROUND_COUNT_MIN` (= 2) entries;
    the runtime enforces this and rejects shorter inputs with a 422.
    """

    rounds: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of round records. Each record carries ``round_id`` / "
            "``started_at`` (verbatim) and may include ``user_goal``, "
            "``aftermath`` (Step 10), and ``debrief`` (Step 9)."
        ),
    )
    person_profile: dict[str, Any] = Field(
        ...,
        description=(
            "PersonVault dict (Step 6). Pulse is per-relationship and "
            "needs a profile to attach to."
        ),
    )
    prior_pulse: Optional[dict[str, Any]] = Field(
        default=None,
        description="Previous pulse dict for incremental v1 → v2 updates.",
    )
    person_id: Optional[str] = None
    max_new_tokens: int = 4096
    enable_thinking: bool = True
    prompt_name: str = "pulse"
