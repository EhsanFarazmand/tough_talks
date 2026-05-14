"""Pre-Mortem Generator — Phase 3 / Step 8 of Tough Talks.

One-shot analytical pass that answers "if this conversation went badly,
what are the three specific ways it could fail?". Output is three
parameterised failure scenarios matching
``data/schemas/premortem.schema.json``; each scenario's
``simulation_parameters`` carries the same ``resistance_type`` enum as
:mod:`persona_sim`, so a scenario can be handed to
:func:`generate_persona_reply` to drive a practice round in Step 7.

Architectural shape mirrors Steps 04 / 05 / 06:

* Text-only Gemma 4 (``LoadConfig(multimodal=False)``).
* Single prompt-based JSON call — not multi-turn like persona simulation.
* Two-shot retry: greedy first, then a single light-sampling pass
  (:data:`RETRY_SAMPLING`) on JSON / validation failure.
* System-contract enforcement in code: ``scenario_id`` forced to its
  index+1, ``resistance_type`` synonym-normalised against the
  persona-sim canonical enum, both numerics clamped to ``[0, 1]``,
  required string fields rejected when empty. The model is told what to
  do but not trusted to do it — same defence-in-depth pattern as the
  ``whisper_prompt=None for other`` rule from Step 04 and the
  ``persona_name`` forcing from Step 07.

Inputs accepted (all optional except ``conversation_description``):

* ``conversation_description`` — free-text summary of the upcoming talk.
* ``user_goal`` — what the user wants from the conversation. Falls back
  to :data:`DEFAULT_USER_GOAL`.
* ``person_profile`` — full PersonVault dict from Step 06 (or its inner
  ``profile`` block). Grounds the scenarios in the counterparty's
  observed habits — when present, Jamie's ``common_deflections`` should
  show up verbatim in an ``opening_move``, etc.
* ``talk_dna_profile`` — TalkDNA dict from Step 05. Surfaces the
  user-side weaknesses the opponent could exploit (over-apologising,
  silence under pressure, known escalation triggers).

The runtime renders empty / missing context blocks as sentinels so the
prompt template is never malformed when the user has no prior data.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .audio import _extract_text
from .generation import GenerationConfig, chat
from .parsing import JsonParseError, parse_json
from .persona_sim import (
    ALLOWED_RESISTANCE_TYPES,
    _normalize_resistance_type,
    format_persona_profile_block,
)
from .prompts import load_prompt

__all__ = [
    "DEFAULT_PREMORTEM_PROMPT_NAME",
    "DEFAULT_USER_GOAL",
    "PremortemError",
    "PremortemConfig",
    "RETRY_SAMPLING",
    "REQUIRED_SCENARIO_COUNT",
    "build_premortem_messages",
    "format_talk_dna_block",
    "generate_premortem",
]

LOG = logging.getLogger(__name__)

DEFAULT_PREMORTEM_PROMPT_NAME = "premortem"
DEFAULT_USER_GOAL = (
    "Have a constructive conversation about the open issue."
)

# The schema pins this at exactly three (minItems == maxItems == 3); the
# constant is exported so the unit test can pin against the schema and
# the notebook can render the count without re-reading the schema.
REQUIRED_SCENARIO_COUNT = 3

# Light sampling for the retry path — same temperature / top_p / top_k
# as :data:`backend.core._runtime.persona_sim.RETRY_SAMPLING`. Pinned by
# the unit test so drift is loud.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}

# Caps on free-text fields. The schema has no length constraint; these
# guard against the model running off into multi-paragraph narration
# instead of the requested 2-3 sentence shape.
_MAX_GOAL_CHARS = 400
_MAX_TITLE_CHARS = 120
_MAX_DESCRIPTION_CHARS = 800
_MAX_TRIGGER_CHARS = 400
_MAX_OPENING_MOVE_CHARS = 400

_NO_TALK_DNA_SENTINEL = (
    "(no Talk DNA profile provided — design the scenarios from the "
    "conversation context and person profile alone)"
)


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class PremortemError(ValueError):
    """The model emitted a malformed pre-mortem payload — wrong scenario
    count, missing required field, unknown ``resistance_type``, etc."""


@dataclass
class PremortemConfig:
    """Run-time inputs for :func:`generate_premortem`.

    ``conversation_description`` is the only required field — everything
    else has a sensible default or empty-sentinel rendering. Pass
    ``person_profile`` and ``talk_dna_profile`` when you have them so
    the scenarios can be grounded in observed behaviour rather than
    generic playbook moves.

    ``enable_thinking`` exposes Gemma 4's thinking channel; defaults to
    ``False`` per the rule in ``knowledge/phases/rules.md``. The Phase 4
    open hypothesis is that thinking *might* help on analytical /
    reflection tasks (debrief, premortem, aftermath) where in-character
    voice doesn't matter — the notebook A/B-tests this on a single
    cold-start call.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from the four
    context blocks. Pass an explicit ``instruction`` only for ablation
    experiments.
    """

    conversation_description: str = ""
    user_goal: str = DEFAULT_USER_GOAL
    person_profile: dict = field(default_factory=dict)
    talk_dna_profile: dict = field(default_factory=dict)
    max_new_tokens: int = 768
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_PREMORTEM_PROMPT_NAME
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def _render_str_list(items: Any) -> str:
    if isinstance(items, list) and items:
        cleaned = [str(x).strip() for x in items if str(x).strip()]
        if cleaned:
            return str(cleaned)
    return "(none observed)"


def format_talk_dna_block(talk_dna_profile: Any) -> str:
    """Render the pre-mortem-relevant slice of a TalkDNA profile.

    Only surfaces the fields the opponent could exploit
    (``weaknesses``, ``patterns.escalation_triggers``,
    ``patterns.apology_rate``, ``patterns.silence_under_pressure``,
    ``patterns.sarcasm_frequency``) plus ``strengths`` so the model
    knows what NOT to assume the user fails at.

    Accepts the full TalkDNA dict produced by Step 05; tolerant of
    missing fields (renders ``(none observed)`` / ``(unknown)``
    sentinels). Returns the ``no profile`` sentinel when input is
    empty so the prompt template is never malformed on cold-start.
    """
    if not isinstance(talk_dna_profile, dict) or not talk_dna_profile:
        return _NO_TALK_DNA_SENTINEL

    patterns = talk_dna_profile.get("patterns")
    if not isinstance(patterns, dict):
        patterns = {}

    strengths = talk_dna_profile.get("strengths")
    weaknesses = talk_dna_profile.get("weaknesses")

    apology_rate = patterns.get("apology_rate")
    if isinstance(apology_rate, (int, float)):
        apology_rate_str = f"{float(apology_rate):.2f}"
    else:
        apology_rate_str = "(unknown)"

    silence = patterns.get("silence_under_pressure")
    silence_str = "(unknown)" if silence is None else str(bool(silence)).lower()

    sarcasm = patterns.get("sarcasm_frequency") or "(unknown)"

    lines = [
        f"- weaknesses (USER habits the opponent could exploit): {_render_str_list(weaknesses)}",
        f"- strengths (USER habits already working): {_render_str_list(strengths)}",
        (
            "- escalation_triggers (topics / phrases that tend to "
            f"escalate the USER): {_render_str_list(patterns.get('escalation_triggers'))}"
        ),
        f"- apology_rate (fraction of USER turns containing apologies): {apology_rate_str}",
        f"- silence_under_pressure: {silence_str}",
        f"- sarcasm_frequency: {sarcasm}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat-template messages
# ---------------------------------------------------------------------------


def build_premortem_messages(instruction: str) -> list[dict]:
    """Build the OpenAI-style messages list for one pre-mortem call.

    Single user-side instruction at index 0 — analytical components like
    Steps 04 / 05 / 06 follow the same shape (no system role for
    prompt-based JSON; the instruction goes in the user turn so the
    chat template lays it out as a plain request the model answers
    directly).

    Pure function — kept separate from :func:`generate_premortem` so
    the message shape can be unit-tested without loading transformers.
    """
    return [{"role": "user", "content": instruction}]


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


def _cap(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _coerce_simulation_parameters(
    raw: Any,
    *,
    scenario_id: int,
) -> dict:
    if not isinstance(raw, dict):
        raise PremortemError(
            f"scenario {scenario_id}: simulation_parameters not an object: "
            f"{type(raw).__name__}"
        )

    raw_rt = raw.get("resistance_type")
    resistance_type = _normalize_resistance_type(raw_rt)
    if resistance_type is None:
        raise PremortemError(
            f"scenario {scenario_id}: resistance_type {raw_rt!r} not in "
            f"enum {ALLOWED_RESISTANCE_TYPES}"
        )

    escalation_ceiling = _clamp_unit(raw.get("escalation_ceiling", 0.0))

    opening_move = _opt_str(raw.get("opening_move"))
    if not opening_move:
        raise PremortemError(
            f"scenario {scenario_id}: opening_move is empty or missing"
        )
    opening_move = _cap(opening_move, limit=_MAX_OPENING_MOVE_CHARS)

    return {
        "resistance_type": resistance_type,
        "escalation_ceiling": escalation_ceiling,
        "opening_move": opening_move,
    }


def _coerce_scenario(raw: Any, *, scenario_id: int) -> dict:
    if not isinstance(raw, dict):
        raise PremortemError(
            f"scenario {scenario_id}: payload not an object: {type(raw).__name__}"
        )

    title = _opt_str(raw.get("title"))
    if not title:
        raise PremortemError(f"scenario {scenario_id}: title is empty or missing")
    title = _cap(title, limit=_MAX_TITLE_CHARS)

    description = _opt_str(raw.get("description"))
    if not description:
        raise PremortemError(
            f"scenario {scenario_id}: description is empty or missing"
        )
    description = _cap(description, limit=_MAX_DESCRIPTION_CHARS)

    likely_trigger = _opt_str(raw.get("likely_trigger"))
    if not likely_trigger:
        raise PremortemError(
            f"scenario {scenario_id}: likely_trigger is empty or missing"
        )
    likely_trigger = _cap(likely_trigger, limit=_MAX_TRIGGER_CHARS)

    destabilization_risk = _clamp_unit(raw.get("destabilization_risk", 0.0))

    sim_params = _coerce_simulation_parameters(
        raw.get("simulation_parameters"),
        scenario_id=scenario_id,
    )

    return {
        "scenario_id": scenario_id,
        "title": title,
        "description": description,
        "likely_trigger": likely_trigger,
        "destabilization_risk": destabilization_risk,
        "simulation_parameters": sim_params,
    }


def _coerce_premortem_payload(payload: Any, *, cfg: PremortemConfig) -> dict:
    """Turn the model's JSON output into a schema-conforming PreMortem dict.

    System-contract enforcement (in code, not prompt — same rationale
    as Step 04's ``whisper_prompt=None for other`` rule and Step 07's
    ``persona_name`` forcing):

    * ``goal`` required and non-empty; falls back to
      ``cfg.user_goal`` when the model omits or empties it.
    * ``failure_scenarios`` must be a list of length
      :data:`REQUIRED_SCENARIO_COUNT`. Anything else raises so the
      retry path takes over rather than letting a 2- or 4-scenario
      response leak through.
    * ``scenario_id`` is forced to ``index + 1`` regardless of model
      output — the model frequently mislabels mid-list items and the
      schema requires the canonical 1 / 2 / 3 sequence.
    * ``resistance_type`` runs through
      :func:`_normalize_resistance_type` so casing / hyphen /
      near-synonym drift is silently corrected.
    * ``destabilization_risk`` and ``escalation_ceiling`` clamped to
      ``[0, 1]``.
    * Required string fields rejected when empty.

    Raises :class:`PremortemError` for any unrepairable input.
    """
    if not isinstance(payload, dict):
        raise PremortemError(
            f"premortem payload not an object: {type(payload).__name__}"
        )

    goal = _opt_str(payload.get("goal")) or _opt_str(cfg.user_goal)
    if not goal:
        raise PremortemError(
            "premortem 'goal' is empty and cfg.user_goal is also empty"
        )
    goal = _cap(goal, limit=_MAX_GOAL_CHARS)

    raw_scenarios = payload.get("failure_scenarios")
    if not isinstance(raw_scenarios, list):
        raise PremortemError(
            f"failure_scenarios must be a list, got {type(raw_scenarios).__name__}"
        )
    if len(raw_scenarios) != REQUIRED_SCENARIO_COUNT:
        raise PremortemError(
            f"failure_scenarios must have exactly {REQUIRED_SCENARIO_COUNT} "
            f"items, got {len(raw_scenarios)}"
        )

    scenarios = [
        _coerce_scenario(raw, scenario_id=i + 1)
        for i, raw in enumerate(raw_scenarios)
    ]

    return {
        "goal": goal,
        "failure_scenarios": scenarios,
        "premortem_id": f"premortem_{uuid.uuid4().hex[:12]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(cfg: PremortemConfig) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    description = (cfg.conversation_description or "").strip()
    if not description:
        raise PremortemError(
            "PremortemConfig.conversation_description is empty — pre-mortem "
            "needs a situation to reason about"
        )
    user_goal = (cfg.user_goal or DEFAULT_USER_GOAL).strip() or DEFAULT_USER_GOAL
    return load_prompt(cfg.prompt_name).render(
        conversation_description=description,
        user_goal=user_goal,
        person_profile_block=format_persona_profile_block(cfg.person_profile),
        talk_dna_block=format_talk_dna_block(cfg.talk_dna_profile),
    )


def _build_generation_config(
    cfg: PremortemConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def generate_premortem(
    processor: Any,
    model: Any,
    *,
    cfg: Optional[PremortemConfig] = None,
) -> dict:
    """Produce a 3-scenario pre-mortem for an upcoming difficult conversation.

    Inference path mirrors the other intelligence components: greedy
    first (deterministic, fastest), then a single retry with light
    sampling (:data:`RETRY_SAMPLING`) on parse / validation failure.
    If both attempts fail, raises :class:`PremortemError` with an
    ``attempts`` attribute carrying each attempt's raw output (truncated
    to 500 chars) for diagnostics.

    Returns a dict matching ``data/schemas/premortem.schema.json`` plus
    runtime-attached ``premortem_id`` and ``generated_at`` fields (the
    schema permits additional properties).

    Each scenario's ``simulation_parameters`` is shaped to feed directly
    into a ``PersonaSimConfig``-driven practice round in Step 7 — the
    ``resistance_type`` enum is shared end-to-end across both modules.
    """
    cfg = cfg or PremortemConfig()
    instruction = _resolve_instruction(cfg)
    messages = build_premortem_messages(instruction)

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
            return _coerce_premortem_payload(payload, cfg=cfg)
        except (JsonParseError, PremortemError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "generate_premortem attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = PremortemError(
        f"premortem generation failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err
