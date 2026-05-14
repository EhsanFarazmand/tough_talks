"""Adversarial Persona Simulator — Phase 3 / Step 7 of Tough Talks.

Drives a multi-turn practice conversation where Gemma 4 E2B plays the
role of a recurring counterparty (sourced from a PersonVault profile)
and the user rehearses what to say. Unlike Steps 05 / 06 — which run a
single-pass analysis over a whole transcript — this module produces
ONE in-character reply per user turn and maintains a rolling history
so the persona's escalation arc carries forward.

Output per turn matches ``data/schemas/persona_reply.schema.json``:

* ``persona_name`` — forced to the configured persona name in code, not
  trusted to the model (same defence-in-depth pattern as Step 04's
  ``whisper_prompt=None for other`` rule).
* ``reply`` — what the persona actually says.
* ``resistance_type`` — closed enum
  (``deflect | guilt_trip | deny | counter_attack | silent | concede``)
  with a synonym normaliser for model vocabulary drift.
* ``escalation_level`` — clamped to ``[0, 1]``.

The runtime also attaches ``turn_id`` and ``timestamp`` (the schema
permits additional properties).

Two-shot retry: greedy first, then a single light-sampling retry
(``RETRY_SAMPLING``) on JSON parse or validation failure. Same pattern
as :mod:`emotion`, :mod:`talk_dna`, and :mod:`person_vault`.

History format. Each entry in ``history`` is a dict:

* ``{"speaker": "user", "text": "..."}`` — what the user said.
* ``{"speaker": "persona", "persona_name": "...", "reply": "...",
  "resistance_type": "...", "escalation_level": ...}`` — a prior reply
  exactly as the runtime emitted it.

When building the messages list for the chat template, prior persona
turns are serialised back to JSON and passed as ``role="assistant"``
content. Rendering them as JSON (rather than just the ``reply`` text)
keeps the model's continuation distribution aligned with its own
training output — the next assistant turn naturally extends the same
JSON shape — and gives the model visibility into its own escalation
trajectory so resistance compounds realistically across turns.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .audio import _extract_text
from .generation import GenerationConfig, chat
from .parsing import JsonParseError, parse_json
from .prompts import load_prompt

__all__ = [
    "ALLOWED_RESISTANCE_TYPES",
    "DEFAULT_PERSONA_PROMPT_NAME",
    "DEFAULT_USER_GOAL",
    "PersonaReplyError",
    "PersonaSimConfig",
    "RETRY_SAMPLING",
    "build_practice_messages",
    "format_persona_profile_block",
    "generate_persona_reply",
    "run_practice_conversation",
]

LOG = logging.getLogger(__name__)

DEFAULT_PERSONA_PROMPT_NAME = "persona"
DEFAULT_USER_GOAL = (
    "Have a constructive conversation about the open issue."
)
DEFAULT_PERSONA_FALLBACK_NAME = "Other"

# Mirrors the enum at
# data/schemas/persona_reply.schema.json#/properties/resistance_type/enum.
# Unit test pins the runtime constant against the on-disk schema so
# drift fails loudly.
ALLOWED_RESISTANCE_TYPES: tuple[str, ...] = (
    "deflect",
    "guilt_trip",
    "deny",
    "counter_attack",
    "silent",
    "concede",
)

# Near-synonym map for ``resistance_type``. Same defence-in-depth
# pattern as Step 06's ``_COMMUNICATION_STYLE_SYNONYMS``: the model
# tends to reach for words near its training distribution
# (``denial``, ``counterattack``, ``silence``) rather than our exact
# enum codes. Accept the obvious synonyms here and map them to the
# canonical value. Truly unknown labels still return ``None`` so the
# retry path fires.
#
# Keys are stored in two forms (space-separated and snake_case-joined)
# so a lookup hits both ``"guilt trip"`` and ``"guilt_trip"`` after the
# normaliser strips casing / hyphens. Values must be in
# :data:`ALLOWED_RESISTANCE_TYPES`.
_RESISTANCE_TYPE_SYNONYMS: dict[str, str] = {
    # Deflect cluster
    "deflection": "deflect",
    "deflects": "deflect",
    "deflecting": "deflect",
    "avoid": "deflect",
    "avoidance": "deflect",
    "evade": "deflect",
    "evasion": "deflect",
    "redirect": "deflect",
    # Guilt trip cluster
    "guilt": "guilt_trip",
    "guilttrip": "guilt_trip",
    "guilting": "guilt_trip",
    "shame": "guilt_trip",
    "shaming": "guilt_trip",
    "martyr": "guilt_trip",
    # Deny cluster
    "denial": "deny",
    "denies": "deny",
    "denying": "deny",
    "reject": "deny",
    "rejection": "deny",
    "dismiss": "deny",
    "dismissal": "deny",
    # Counter-attack cluster
    "counterattack": "counter_attack",
    "counter": "counter_attack",
    "attack": "counter_attack",
    "retaliation": "counter_attack",
    "retaliate": "counter_attack",
    "blame_shift": "counter_attack",
    "blameshift": "counter_attack",
    "blame shifting": "counter_attack",
    # Silent cluster
    "silence": "silent",
    "silent_treatment": "silent",
    "silent treatment": "silent",
    "withdraw": "silent",
    "withdrawal": "silent",
    "withdrawn": "silent",
    "stonewall": "silent",
    "stonewalling": "silent",
    "shutdown": "silent",
    "shut_down": "silent",
    # Concede cluster
    "concession": "concede",
    "concedes": "concede",
    "conceding": "concede",
    "agreement": "concede",
    "agree": "concede",
    "agrees": "concede",
    "yield": "concede",
    "accept": "concede",
    "acceptance": "concede",
}


def _normalize_resistance_type(value: Any) -> Optional[str]:
    """Coerce a model-emitted ``resistance_type`` to a canonical enum value.

    Handles three sources of drift between model output and the schema
    enum:

    * **Casing** — ``"Deflect"`` / ``"DEFLECT"`` → ``"deflect"``.
    * **Word separator** — ``"counter-attack"`` / ``"counter attack"``
      → ``"counter_attack"``.
    * **Near-synonyms** — ``"denial"`` → ``"deny"``,
      ``"counterattack"`` → ``"counter_attack"``,
      ``"silence"`` → ``"silent"``, etc. via
      :data:`_RESISTANCE_TYPE_SYNONYMS`.

    Returns the canonical enum value, or ``None`` if the input is
    unrecognisable. Callers raise a :class:`PersonaReplyError` on
    ``None`` so the retry path can take over.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace("-", " ")
    if not cleaned:
        return None
    snake = cleaned.replace(" ", "_")
    if snake in ALLOWED_RESISTANCE_TYPES:
        return snake
    # Try both the space-separated and snake-joined forms so callers
    # can register synonyms in whichever shape reads more naturally.
    return _RESISTANCE_TYPE_SYNONYMS.get(cleaned) or _RESISTANCE_TYPE_SYNONYMS.get(
        snake
    )


# Light sampling for the retry path — same rationale as the
# talk_dna / person_vault / emotion retries. Pinned by the unit test.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}

# Cap on the persona's spoken `reply` field. A spoken turn that runs
# more than ~800 chars is almost certainly the model drifting into
# narration rather than dialogue; clamp it rather than truncating mid-
# token at decode time.
_MAX_REPLY_CHARS = 800

_NO_PROFILE_SENTINEL = (
    "(no PersonVault profile provided — improvise based on the persona "
    "name and the user's goal alone)"
)
_RELATIONSHIP_UNSPECIFIED = "unspecified"


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class PersonaReplyError(ValueError):
    """The model emitted a malformed persona-reply payload — bad enum
    value, empty / missing ``reply``, non-dict response, etc."""


@dataclass
class PersonaSimConfig:
    """Run-time inputs for :func:`generate_persona_reply`.

    ``persona_profile`` accepts either:

    * a full PersonVault dict (matching
      ``data/schemas/person_vault.schema.json`` — with a top-level
      ``name`` / ``relationship_type`` and an inner ``profile`` object), or
    * the inner ``profile`` dict alone (legacy / hand-rolled callers).

    The runtime extracts ``name`` and ``relationship_type`` from the
    full dict when present, and renders the inner profile object into
    the prompt either way.

    ``user_goal`` is what the USER is trying to achieve in this
    practice round — fed into the prompt so the persona's resistance
    has something concrete to push back against.

    ``enable_thinking`` exposes Gemma 4's thinking channel. Defaults to
    ``False`` (the rule of thumb from ``knowledge/phases/rules.md``);
    the open hypothesis on Step 7 is that thinking might improve
    persona quality. Notebook A/B compares the two.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from the persona
    name, relationship, profile block, and user goal. Pass an explicit
    ``instruction`` only for ablation experiments.
    """

    persona_profile: dict = field(default_factory=dict)
    user_goal: str = DEFAULT_USER_GOAL
    max_new_tokens: int = 384
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_PERSONA_PROMPT_NAME
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Profile-block rendering
# ---------------------------------------------------------------------------


def _is_inner_profile(d: dict) -> bool:
    """Best-effort check that ``d`` is the inner PersonVault profile
    object (the one nested under the ``profile`` key) rather than the
    full top-level wrapper. Used by :func:`_resolve_inner_profile`."""
    if not isinstance(d, dict):
        return False
    # The inner profile carries fields the outer dict never does;
    # the outer wrapper carries fields the inner profile never does.
    inner_fields = {
        "communication_style",
        "emotional_triggers",
        "de_escalation_keys",
        "common_deflections",
        "responds_best_to",
    }
    outer_fields = {"person_id", "version", "conversation_count", "profile"}
    if any(f in d for f in outer_fields):
        return False
    return any(f in d for f in inner_fields)


def _resolve_inner_profile(persona_profile: Any) -> dict:
    """Accept either a full PersonVault dict or the inner ``profile``
    dict and return the inner profile dict.

    Returns ``{}`` when the input is neither — callers fall through to
    the ``no profile`` sentinel.
    """
    if not isinstance(persona_profile, dict):
        return {}
    inner = persona_profile.get("profile")
    if isinstance(inner, dict):
        return inner
    if _is_inner_profile(persona_profile):
        return persona_profile
    return {}


def _resolve_persona_name(
    persona_profile: Any,
    *,
    fallback: str = DEFAULT_PERSONA_FALLBACK_NAME,
) -> str:
    if isinstance(persona_profile, dict):
        name = persona_profile.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return fallback


def _resolve_relationship_block(persona_profile: Any) -> str:
    if isinstance(persona_profile, dict):
        rt = persona_profile.get("relationship_type")
        if isinstance(rt, str) and rt.strip():
            return rt.strip()
    return _RELATIONSHIP_UNSPECIFIED


def format_persona_profile_block(persona_profile: Any) -> str:
    """Render a PersonVault profile as a text block for the prompt.

    Accepts either the full PersonVault dict (with top-level ``profile``
    key) or the inner profile dict directly. Missing fields are
    rendered as ``(unknown)`` / ``(none observed)`` rather than omitted
    so the model sees a stable structure each turn.

    Returns a sentinel "no profile" string when the input is empty so
    the prompt template is never malformed on cold-start calls.
    """
    inner = _resolve_inner_profile(persona_profile)
    if not inner:
        return _NO_PROFILE_SENTINEL

    style = inner.get("communication_style") or "(unknown)"
    triggers = inner.get("emotional_triggers") or []
    de_esc = inner.get("de_escalation_keys") or []
    deflections = inner.get("common_deflections") or []
    responds_best_to = inner.get("responds_best_to") or "(unknown)"
    cultural = inner.get("cultural_context")

    lines = [
        f"- communication_style: {style}",
        (
            "- emotional_triggers (USER-side cues that escalate / make you "
            f"defensive): {list(triggers) if triggers else '(none observed)'}"
        ),
        (
            "- de_escalation_keys (USER-side moves that calm / open you up): "
            f"{list(de_esc) if de_esc else '(none observed)'}"
        ),
        (
            "- common_deflections (YOUR habitual evasion phrases): "
            f"{list(deflections) if deflections else '(none observed)'}"
        ),
        f"- responds_best_to: {responds_best_to}",
    ]
    if isinstance(cultural, str) and cultural.strip():
        lines.append(f"- cultural_context: {cultural.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat-template messages
# ---------------------------------------------------------------------------


def _serialize_persona_turn(turn: dict) -> str:
    """Render a prior persona turn back into the assistant-side JSON
    string for the chat template.

    We pass the same JSON shape the runtime emits so the model's
    continuation distribution stays consistent — it sees its own
    structured trajectory, including escalation_level and
    resistance_type, and the next turn naturally extends that shape.
    """
    persona_name = turn.get("persona_name") or DEFAULT_PERSONA_FALLBACK_NAME
    reply = turn.get("reply") or ""
    resistance_type = turn.get("resistance_type") or "deflect"
    raw_escalation = turn.get("escalation_level", 0.0)
    try:
        escalation = float(raw_escalation)
    except (TypeError, ValueError):
        escalation = 0.0
    payload = {
        "persona_name": persona_name,
        "reply": reply,
        "resistance_type": resistance_type,
        "escalation_level": escalation,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_practice_messages(
    instruction: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """Build the OpenAI-style messages list for one persona-reply call.

    Structure: ``[system, user_1, assistant_1, ..., user_N, assistant_N,
    user_new]``. The system message is the rendered persona instruction;
    each history persona turn is serialised back to JSON.

    Pure function — kept separate from :func:`generate_persona_reply`
    so the message shape can be unit-tested without loading transformers.
    """
    messages: list[dict] = [{"role": "system", "content": instruction}]
    for entry in history:
        speaker = entry.get("speaker") if isinstance(entry, dict) else None
        if speaker == "user":
            text = entry.get("text") or ""
            messages.append({"role": "user", "content": text})
        elif speaker == "persona":
            messages.append(
                {"role": "assistant", "content": _serialize_persona_turn(entry)}
            )
        # Unknown speakers are skipped silently — the caller controls
        # the history shape and we'd rather drop a malformed entry than
        # corrupt the chat-template alignment with a stray role.
    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Payload coercion
# ---------------------------------------------------------------------------


def _clamp_unit(value: Any, *, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, val))


def _opt_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_persona_reply_payload(
    payload: Any,
    *,
    cfg: PersonaSimConfig,
) -> dict:
    """Turn the model's JSON output into a schema-conforming
    PersonaReply dict.

    System-contract enforcement (in code, not prompt — same rationale
    as Step 04's ``whisper_prompt=None for other`` rule):

    * ``persona_name`` is always forced to the configured persona name.
      The model is told to use it but Step 04's Colab runs showed
      prompt rules aren't reliable.
    * ``resistance_type`` runs through :func:`_normalize_resistance_type`
      so casing / hyphen / near-synonym drift is silently corrected.
    * ``escalation_level`` is clamped to ``[0, 1]``.
    * ``reply`` is required, non-empty, and capped at
      :data:`_MAX_REPLY_CHARS`.

    Raises :class:`PersonaReplyError` for any unrepairable input so the
    retry path can take over.
    """
    if not isinstance(payload, dict):
        raise PersonaReplyError(
            f"persona_reply payload not an object: {type(payload).__name__}"
        )

    reply = _opt_str(payload.get("reply"))
    if not reply:
        raise PersonaReplyError("persona reply is empty or missing")
    if len(reply) > _MAX_REPLY_CHARS:
        reply = reply[: _MAX_REPLY_CHARS - 1].rstrip() + "…"

    raw_rt = payload.get("resistance_type")
    resistance_type = _normalize_resistance_type(raw_rt)
    if resistance_type is None:
        raise PersonaReplyError(
            f"resistance_type not in enum: {raw_rt!r}; "
            f"allowed: {ALLOWED_RESISTANCE_TYPES}"
        )

    escalation = _clamp_unit(payload.get("escalation_level", 0.0))

    persona_name = _resolve_persona_name(cfg.persona_profile)

    return {
        "persona_name": persona_name,
        "reply": reply,
        "resistance_type": resistance_type,
        "escalation_level": escalation,
        "turn_id": f"persona_turn_{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(cfg: PersonaSimConfig) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    persona_name = _resolve_persona_name(cfg.persona_profile)
    relationship_block = _resolve_relationship_block(cfg.persona_profile)
    user_goal = (cfg.user_goal or DEFAULT_USER_GOAL).strip() or DEFAULT_USER_GOAL
    return load_prompt(cfg.prompt_name).render(
        persona_name=persona_name,
        relationship_block=relationship_block,
        profile_block=format_persona_profile_block(cfg.persona_profile),
        user_goal=user_goal,
    )


def _build_generation_config(
    cfg: PersonaSimConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def generate_persona_reply(
    processor: Any,
    model: Any,
    user_message: str,
    *,
    history: Optional[list[dict]] = None,
    cfg: Optional[PersonaSimConfig] = None,
) -> dict:
    """Produce one in-character persona reply to the user's latest message.

    ``history`` is the running list of prior practice-conversation
    turns (each entry is a dict in the shape the runtime emits). Pass
    ``None`` or ``[]`` for the first turn of a new practice round.

    Inference path mirrors the other intelligence components: greedy
    first (deterministic, fastest), then a single retry with light
    sampling (:data:`RETRY_SAMPLING`) on parse / validation failure.
    If both attempts fail, raises :class:`PersonaReplyError` with an
    ``attempts`` attribute carrying each attempt's raw output
    (truncated to 500 chars) for diagnostics.

    Returns a dict matching
    ``data/schemas/persona_reply.schema.json`` plus runtime-attached
    ``turn_id`` and ``timestamp`` fields (the schema permits
    additional properties).
    """
    cfg = cfg or PersonaSimConfig()
    history = list(history or [])

    user_message = (user_message or "").strip()
    if not user_message:
        raise PersonaReplyError("user_message is empty — nothing to reply to")

    instruction = _resolve_instruction(cfg)
    messages = build_practice_messages(instruction, history, user_message)

    attempts: list[dict] = []
    for sampling in (None, RETRY_SAMPLING):
        gen_cfg = _build_generation_config(cfg, sampling)
        raw = chat(
            processor,
            model,
            messages,
            cfg=gen_cfg,
            enable_thinking=cfg.enable_thinking,
        )
        text = _extract_text(processor.parse_response(raw)).strip()
        try:
            payload = parse_json(text)
            return _coerce_persona_reply_payload(payload, cfg=cfg)
        except (JsonParseError, PersonaReplyError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "generate_persona_reply attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = PersonaReplyError(
        f"persona reply generation failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err


def run_practice_conversation(
    processor: Any,
    model: Any,
    user_messages: list[str],
    *,
    cfg: Optional[PersonaSimConfig] = None,
) -> list[dict]:
    """Run a full multi-turn practice conversation non-interactively.

    For each user message in order, generate one persona reply, append
    both the user turn and the persona reply to the running history,
    and continue. Returns the final history as a list of dicts
    alternating ``user`` and ``persona`` entries.

    Convenient for notebook demos and offline replay. The interactive
    Live-Mode path will call :func:`generate_persona_reply` one turn at
    a time instead, threading its own UI between calls.
    """
    cfg = cfg or PersonaSimConfig()
    history: list[dict] = []
    for user_message in user_messages:
        reply = generate_persona_reply(
            processor,
            model,
            user_message,
            history=history,
            cfg=cfg,
        )
        history.append({"speaker": "user", "text": user_message})
        history.append({"speaker": "persona", **reply})
    return history
