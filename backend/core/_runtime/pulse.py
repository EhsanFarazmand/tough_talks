"""Relationship Pulse Tracker — Phase 4 / Step 11 of Tough Talks.

One-shot analytical pass that rolls up multiple practice rounds for a
single (user, counterparty) pair into a single relationship-level pulse
matching ``data/schemas/pulse.schema.json``:

* ``health_score`` — point-in-time read in ``[0, 1]``.
* ``health_trend`` — direction of travel across the rounds (``improving``
  / ``stable`` / ``declining`` / ``volatile`` / ``insufficient_data``).
* ``trajectory_summary`` — short prose on how the relationship has moved.
* ``round_summaries[]`` — one compact entry per round, chronological. The
  ``round_id`` and ``started_at`` are forced from the input record so the
  model never has to re-emit metadata; the model only contributes
  ``goal_status`` (when it disagrees with the per-round aftermath),
  ``prediction_accuracy``, and ``headline``.
* ``recurring_patterns[]`` — shapes observed in TWO OR MORE distinct
  rounds. ``evidence_round_ids`` filtered to known ids and entries with
  ``<2`` valid ids dropped (the schema demands ``minItems: 2`` — a
  pattern is only "recurring" when it shows up in at least two rounds).
  Accumulates across pulse versions like PersonVault's qualitative
  lists.
* ``emerging_concerns[]`` — risks that surfaced in the recent 1-2 rounds.
  Rebuilt fresh each call (concerns are point-in-time, not cumulative).
* ``relationship_wins[]`` — durable user-side wins worth carrying forward.
  Accumulates across pulse versions.
* ``next_step_recommendation`` — single-sentence pointer for the next
  round with this person.

Architectural shape mirrors Steps 04 / 05 / 06 / 08 / 09 / 10:

* Text-only Gemma 4 (``LoadConfig(multimodal=False)``).
* Single prompt-based JSON call — no rolling history. All rounds are
  fed in once via the rendered ``rounds_block``.
* Two-shot retry: greedy first, then a single light-sampling pass
  (:data:`RETRY_SAMPLING`) on JSON / validation failure.
* ``enable_thinking`` defaults to ``True`` (the rule promoted in Step
  10 — ``knowledge/phases/rules.md § Decoding knobs by task type`` —
  applies: pulse is an analytical payload that cross-references
  multiple speaker roles across multiple rounds in a single output
  object). ``max_new_tokens`` defaults to ``4096`` for the same
  reason — pulse's input is even larger than aftermath's (PersonVault
  + N rendered rounds + optional prior pulse), so the thinking
  trace needs room.
* System-contract enforcement in code (defence-in-depth, same shape
  as Step 04's ``whisper_prompt=None for other`` rule, Step 07's
  ``persona_name`` forcing, Step 08's ``scenario_id`` forcing, Step
  09's ``_APOLOGY_CUE_RE``, and Step 10's ``_TRANSCRIPT_META_PREFIX_RE``):
  - ``person_id`` forced from PersonVault input (or ``cfg.person_id``),
    NOT trusted to the model.
  - ``version`` bumped from ``prior_pulse.version + 1`` (or ``1`` on
    cold-start).
  - ``round_count`` derived in code from ``len(round_summaries)``.
  - ``round_summaries`` always rebuilt to exactly one entry per input
    round, in input order — ``round_id`` and ``started_at`` copied
    verbatim from input; the model only contributes ``goal_status``,
    ``prediction_accuracy``, and ``headline``.
  - ``goal_status`` enum-normalised; when the model disagrees with
    the round's own aftermath ``goal_outcome.status``, the
    aftermath wins (it's the authoritative per-round artifact).
  - ``prediction_accuracy`` clamped to ``[0, 1]``; falls back to the
    round's aftermath value when the model omits or garbles it.
  - ``health_trend`` enum-normalised with synonym map. When all
    rounds have ``goal_status == "unknown"`` (no aftermaths), the
    runtime overrides whatever the model emitted to
    ``insufficient_data`` — the model frequently picks ``stable``
    when there's nothing to go on, which is wrong.
  - ``recurring_patterns[].evidence_round_ids`` filtered to known
    ``round_id``s only; entries with ``<2`` valid ids dropped.
  - ``emerging_concerns[].round_id`` validated against known
    ``round_id``s; dropped silently when unknown.
  - ``recurring_patterns`` and ``relationship_wins`` accumulate across
    pulse versions (deduped, capped). Same pattern as PersonVault's
    qualitative lists.
  - Free-text fields capped to guard against multi-paragraph drift.
  - Min ``ROUND_COUNT_MIN == 2`` enforced at instruction-resolution
    time; a single-round pulse is rejected because a trajectory
    needs at least two points.

Inputs accepted:

* ``rounds`` — list of round records. Each record must carry
  ``round_id`` and ``started_at`` (strings, used verbatim) and may
  carry ``user_goal``, ``aftermath`` (from :mod:`aftermath`), and
  ``debrief`` (from :mod:`debrief`). When the aftermath is missing
  the round still counts toward ``round_count``, but its
  ``goal_status`` defaults to ``"unknown"`` and ``prediction_accuracy``
  to ``0.0``.
* ``person_profile`` — full PersonVault dict from Step 06 (or its
  inner ``profile`` block). Required — pulse is per-relationship and
  cannot exist without a person to attach it to.
* ``person_id`` — optional override; defaults to
  ``person_profile["person_id"]``. The runtime raises if neither is
  available.
* ``prior_pulse`` — optional previous pulse dict for incremental
  v1 → v2 updates. ``recurring_patterns`` and ``relationship_wins``
  accumulate from this; ``health_score`` / ``health_trend`` /
  ``trajectory_summary`` / ``emerging_concerns`` /
  ``next_step_recommendation`` are rebuilt fresh.
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
from .person_vault import (
    ALLOWED_RELATIONSHIP_TYPES,
    DEFAULT_PERSON_NAME,
    DEFAULT_RELATIONSHIP_TYPE,
)
from .persona_sim import format_persona_profile_block
from .prompts import load_prompt

__all__ = [
    "ALLOWED_HEALTH_TRENDS",
    "ALLOWED_ROUND_GOAL_STATUSES",
    "DEFAULT_PULSE_PROMPT_NAME",
    "PulseConfig",
    "PulseError",
    "ROUND_COUNT_MIN",
    "RETRY_SAMPLING",
    "build_pulse_messages",
    "format_prior_pulse_block",
    "format_rounds_block",
    "generate_pulse",
]

LOG = logging.getLogger(__name__)

DEFAULT_PULSE_PROMPT_NAME = "pulse"

# Mirrors the enums at data/schemas/pulse.schema.json. Unit tests pin
# these tuples against the on-disk schema so drift fails loudly.
ALLOWED_HEALTH_TRENDS: tuple[str, ...] = (
    "improving",
    "stable",
    "declining",
    "volatile",
    "insufficient_data",
)
# Same set as ALLOWED_GOAL_STATUSES from :mod:`aftermath` plus
# ``"unknown"`` — a round with no aftermath cannot be scored.
ALLOWED_ROUND_GOAL_STATUSES: tuple[str, ...] = (
    "achieved",
    "partial",
    "not_achieved",
    "unknown",
)

# Near-synonym map for ``health_trend``. Same defence-in-depth shape
# as Step 06's ``_COMMUNICATION_STYLE_SYNONYMS``, Step 07's
# ``_RESISTANCE_TYPE_SYNONYMS``, Step 10's
# ``_MATCH_QUALITY_SYNONYMS``: the model reaches for adjacent words
# rather than the exact enum codes. Accept the obvious synonyms here;
# truly unknown labels still return ``None`` so the retry path fires.
_HEALTH_TREND_SYNONYMS: dict[str, str] = {
    # Improving cluster
    "better": "improving",
    "getting better": "improving",
    "improvement": "improving",
    "positive": "improving",
    "trending up": "improving",
    "upward": "improving",
    "progressing": "improving",
    # Stable cluster
    "flat": "stable",
    "steady": "stable",
    "consistent": "stable",
    "unchanged": "stable",
    "plateaued": "stable",
    # Declining cluster
    "worsening": "declining",
    "deteriorating": "declining",
    "regressing": "declining",
    "getting worse": "declining",
    "negative": "declining",
    "trending down": "declining",
    "downward": "declining",
    # Volatile cluster
    "unstable": "volatile",
    "erratic": "volatile",
    "swinging": "volatile",
    "inconsistent": "volatile",
    "mixed": "volatile",
    "fluctuating": "volatile",
    # Insufficient-data cluster
    "unknown": "insufficient_data",
    "not_enough_data": "insufficient_data",
    "not enough data": "insufficient_data",
    "insufficient": "insufficient_data",
    "indeterminate": "insufficient_data",
    "n/a": "insufficient_data",
    "na": "insufficient_data",
}

# Reuses the aftermath goal-status synonyms but adds an explicit
# ``"unknown"`` cluster — aftermath does not allow ``unknown`` (every
# practiced round must have a goal_outcome), but pulse does allow it
# for rounds that lack an aftermath.
_ROUND_GOAL_STATUS_SYNONYMS: dict[str, str] = {
    "yes": "achieved",
    "success": "achieved",
    "successful": "achieved",
    "met": "achieved",
    "complete": "achieved",
    "completed": "achieved",
    "partly": "partial",
    "mixed": "partial",
    "somewhat": "partial",
    "halfway": "partial",
    "half": "partial",
    "no": "not_achieved",
    "failed": "not_achieved",
    "miss": "not_achieved",
    "missed": "not_achieved",
    "unachieved": "not_achieved",
    "did_not_achieve": "not_achieved",
    "did not achieve": "not_achieved",
    "n/a": "unknown",
    "na": "unknown",
    "not_available": "unknown",
    "not available": "unknown",
    "no_aftermath": "unknown",
    "no aftermath": "unknown",
    "tbd": "unknown",
    "pending": "unknown",
}

# Light sampling for the retry path — same temperature / top_p / top_k
# as the other intelligence runtimes. Pinned by the unit test so drift
# is loud.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}

# Minimum number of rounds a pulse can reflect. A single-round
# trajectory does not exist; the schema's ``round_count`` minimum is
# ``2`` and ``round_summaries.minItems`` is also ``2``. Enforced at
# :func:`_resolve_instruction` time so the model never sees an
# under-rounded pulse.
ROUND_COUNT_MIN: int = 2

# Caps on free-text fields and qualitative lists. The schema has no
# length constraints; these guard against the model running into
# multi-paragraph narration or accumulating an unbounded recurring-
# patterns list across many pulse versions.
_MAX_LIST_ITEMS = 8
_MAX_SUMMARY_CHARS = 800
_MAX_HEADLINE_CHARS = 300
_MAX_PATTERN_CHARS = 400
_MAX_CONCERN_CHARS = 400
_MAX_WIN_CHARS = 400
_MAX_RECOMMENDATION_CHARS = 400

_NO_PROFILE_SENTINEL = (
    "(no PersonVault profile provided — pulse is per-relationship and "
    "needs a profile to attach to)"
)
_NO_PRIOR_PULSE_SENTINEL = (
    "(no prior pulse — this is the first time this relationship is "
    "being rolled up across rounds)"
)
_NO_AFTERMATH_SENTINEL = "(no aftermath — round goal_status is unknown)"
_NO_DEBRIEF_SENTINEL = "(no post-round debrief)"


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class PulseError(ValueError):
    """The model emitted a malformed pulse payload — wrong round count,
    missing required field, unrecognisable ``health_trend``, etc. Also
    raised on invalid input (too few rounds, no PersonVault profile,
    no person_id available)."""


@dataclass
class PulseConfig:
    """Run-time inputs for :func:`generate_pulse`.

    ``rounds`` and ``person_profile`` are the only fields without
    automatic empty-sentinel rendering — pulse is per-relationship and
    needs both a person to attach to and at least two practice rounds
    to trace a trajectory. Both are checked at
    :func:`_resolve_instruction` time and raise :class:`PulseError`
    when missing or unusable.

    ``enable_thinking`` defaults to ``True`` per the rule in
    ``knowledge/phases/rules.md § Decoding knobs by task type``: pulse
    is an analytical payload that cross-references multiple speaker
    roles across multiple rounds in one output object, which is
    exactly the case the rule covers. The notebook A/B re-tests the
    rule on a fourth task family.

    ``max_new_tokens`` defaults to ``4096`` for the same reason —
    pulse's input is larger than aftermath's (PersonVault + N
    rendered rounds + optional prior pulse), so the thinking trace
    needs room. The rule of thumb from
    ``rules.md`` (~2x JSON body + ~1024 per extra rendered context
    block) is comfortably satisfied at 4096 for the common 2-3
    round case.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from the four
    context blocks. Pass an explicit ``instruction`` only for
    ablation experiments.
    """

    rounds: list[dict] = field(default_factory=list)
    person_profile: dict = field(default_factory=dict)
    prior_pulse: Optional[dict] = None
    person_id: Optional[str] = None
    max_new_tokens: int = 4096
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_PULSE_PROMPT_NAME
    enable_thinking: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _cap(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _clamp_unit(value: Any, *, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _normalize_enum(
    value: Any,
    *,
    allowed: tuple[str, ...],
    synonyms: dict[str, str],
) -> Optional[str]:
    """Coerce a model-emitted enum value to a canonical entry.

    Handles three sources of drift between model output and the
    schema enum, same shape as :func:`_normalize_match_quality`:

    * **Casing** — ``"Improving"`` → ``"improving"``.
    * **Word separator** — ``"insufficient-data"`` /
      ``"insufficient data"`` → ``"insufficient_data"``.
    * **Near-synonyms** — ``"worsening"`` → ``"declining"``,
      ``"unstable"`` → ``"volatile"``, etc. via the supplied
      synonym map.

    Returns the canonical enum value, or ``None`` if unrecognisable.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace("-", " ")
    if not cleaned:
        return None
    snake = cleaned.replace(" ", "_")
    if snake in allowed:
        return snake
    return synonyms.get(cleaned) or synonyms.get(snake)


def _normalize_health_trend(value: Any) -> Optional[str]:
    return _normalize_enum(
        value,
        allowed=ALLOWED_HEALTH_TRENDS,
        synonyms=_HEALTH_TREND_SYNONYMS,
    )


def _normalize_round_goal_status(value: Any) -> Optional[str]:
    return _normalize_enum(
        value,
        allowed=ALLOWED_ROUND_GOAL_STATUSES,
        synonyms=_ROUND_GOAL_STATUS_SYNONYMS,
    )


# ---------------------------------------------------------------------------
# Per-round artifact extraction
# ---------------------------------------------------------------------------


def _round_aftermath(round_record: dict) -> dict:
    """Return the round's aftermath dict, or ``{}`` when absent."""
    raw = round_record.get("aftermath")
    return raw if isinstance(raw, dict) else {}


def _round_debrief(round_record: dict) -> dict:
    """Return the round's debrief dict, or ``{}`` when absent."""
    raw = round_record.get("debrief")
    return raw if isinstance(raw, dict) else {}


def _round_goal_status_from_aftermath(round_record: dict) -> str:
    """Pull the authoritative goal_status from the round's aftermath.

    Aftermath's ``goal_outcome.status`` is the per-round source of
    truth — Step 10 already enum-normalised it. When the aftermath is
    missing or malformed, returns ``"unknown"`` so the pulse round can
    still be counted but its trajectory contribution is honest about
    the missing signal.
    """
    aftermath = _round_aftermath(round_record)
    goal_outcome = aftermath.get("goal_outcome")
    if isinstance(goal_outcome, dict):
        normalised = _normalize_round_goal_status(goal_outcome.get("status"))
        if normalised is not None and normalised != "unknown":
            return normalised
    return "unknown"


def _round_prediction_accuracy_from_aftermath(round_record: dict) -> float:
    """Pull prediction_accuracy from the round's aftermath, clamped.

    Falls back to ``0.0`` when the round has no aftermath — same
    convention as the schema's ``prediction_accuracy`` floor.
    """
    aftermath = _round_aftermath(round_record)
    if not aftermath:
        return 0.0
    return _clamp_unit(aftermath.get("prediction_accuracy"), default=0.0)


def _round_id(round_record: dict, *, index: int) -> str:
    """Return the round's id, falling back to ``round_{index+1}``."""
    rid = _opt_str(round_record.get("round_id"))
    return rid or f"round_{index + 1}"


def _round_started_at(round_record: dict) -> str:
    """Return the round's started_at timestamp, or empty string."""
    return _opt_str(round_record.get("started_at")) or ""


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def _render_aftermath_block(aftermath: dict) -> list[str]:
    """Render the high-signal aftermath fields for the rounds_block.

    Surfaces ``goal_outcome``, ``prediction_accuracy``,
    ``scenario_outcomes`` (one line per scenario with
    match_quality + a notes excerpt), ``unforeseen_moments`` count
    grouped by kind, and ``next_round_focus``. Skips the full
    ``evidence`` / ``notes`` text to keep the rendered block compact
    — pulse reasons over the SHAPE of multiple rounds, not the prose
    of any single one.
    """
    if not aftermath:
        return ["  aftermath: " + _NO_AFTERMATH_SENTINEL]

    lines: list[str] = []
    goal_outcome = aftermath.get("goal_outcome")
    if isinstance(goal_outcome, dict):
        status = goal_outcome.get("status")
        summary = _opt_str(goal_outcome.get("summary"))
        if summary:
            lines.append(
                f"  aftermath.goal_outcome: {status} — {summary}"
            )
        else:
            lines.append(f"  aftermath.goal_outcome.status: {status}")

    accuracy = aftermath.get("prediction_accuracy")
    if isinstance(accuracy, (int, float)):
        lines.append(f"  aftermath.prediction_accuracy: {float(accuracy):.2f}")

    outcomes = aftermath.get("scenario_outcomes")
    if isinstance(outcomes, list) and outcomes:
        lines.append("  aftermath.scenario_outcomes:")
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            sid = outcome.get("scenario_id")
            resistance = outcome.get("predicted_resistance_type")
            quality = outcome.get("match_quality")
            notes = _opt_str(outcome.get("notes")) or ""
            notes_excerpt = _cap(notes, limit=160) if notes else ""
            base = f"    S{sid} ({resistance}): {quality}"
            if notes_excerpt:
                base += f" — {notes_excerpt}"
            lines.append(base)

    unforeseen = aftermath.get("unforeseen_moments")
    if isinstance(unforeseen, list) and unforeseen:
        risks = sum(
            1 for m in unforeseen
            if isinstance(m, dict) and m.get("kind") == "risk"
        )
        opps = sum(
            1 for m in unforeseen
            if isinstance(m, dict) and m.get("kind") == "opportunity"
        )
        lines.append(
            f"  aftermath.unforeseen_moments: {len(unforeseen)} "
            f"({risks} risk / {opps} opportunity)"
        )

    next_focus = _opt_str(aftermath.get("next_round_focus"))
    if next_focus:
        lines.append(f"  aftermath.next_round_focus: {next_focus}")
    return lines


def _render_debrief_block(debrief: dict) -> list[str]:
    """Render the high-signal debrief fields for the rounds_block.

    Surfaces win/loss/apology/missed-opening counts plus
    ``one_fix_next_time`` — same shape as
    :func:`aftermath.format_debrief_block`. Skips the per-turn entries
    to keep the rendered block compact.
    """
    if not debrief:
        return ["  debrief: " + _NO_DEBRIEF_SENTINEL]

    def _count(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    lines = [
        f"  debrief: wins={_count(debrief.get('wins'))}, "
        f"ground_lost={_count(debrief.get('ground_lost'))}, "
        f"over_apologies={_count(debrief.get('over_apologies'))}, "
        f"missed_openings={_count(debrief.get('missed_openings'))}",
    ]
    one_fix = _opt_str(debrief.get("one_fix_next_time"))
    if one_fix:
        lines.append(f"  debrief.one_fix_next_time: {one_fix}")
    return lines


def format_rounds_block(rounds: list[dict]) -> str:
    """Render the round records as a numbered text block for the prompt.

    Each round is rendered with its ``round_id``, ``started_at``,
    optional ``user_goal``, and a compact dump of the aftermath /
    debrief headlines (no full transcripts — pulse reasons over the
    SHAPE of the round, not the verbatim turns). Returns the empty
    string when no rounds are present so callers can sentinel
    upstream; the runtime rejects empty-rounds inputs at
    :func:`_resolve_instruction` time.
    """
    if not isinstance(rounds, list) or not rounds:
        return ""

    blocks: list[str] = []
    for i, round_record in enumerate(rounds):
        if not isinstance(round_record, dict):
            continue
        rid = _round_id(round_record, index=i)
        started_at = _round_started_at(round_record) or "(no started_at)"
        user_goal = _opt_str(round_record.get("user_goal"))
        header = (
            f"Round {i + 1} (round_id={rid}, started_at={started_at})"
        )
        if user_goal:
            header += f": user_goal={user_goal!r}"
        block_lines = [header]
        block_lines.extend(_render_aftermath_block(_round_aftermath(round_record)))
        block_lines.extend(_render_debrief_block(_round_debrief(round_record)))
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def format_prior_pulse_block(prior: Optional[dict]) -> str:
    """Render the optional prior pulse as a compact text block.

    Surfaces the high-signal fields the model should carry forward —
    prior ``health_score``, ``health_trend``, full
    ``recurring_patterns`` list (with their evidence ids), and the
    full ``relationship_wins`` list. Skips ``trajectory_summary`` /
    ``next_step_recommendation`` / ``emerging_concerns`` because
    those are point-in-time fields the new pulse rebuilds from
    scratch.

    Returns the ``no prior pulse`` sentinel when input is empty so
    the prompt template is never malformed on cold-start.
    """
    if not isinstance(prior, dict) or not prior:
        return _NO_PRIOR_PULSE_SENTINEL

    lines: list[str] = []
    version = prior.get("version")
    round_count = prior.get("round_count")
    lines.append(
        f"- prior_version: {version} (round_count={round_count})"
    )
    health_score = prior.get("health_score")
    if isinstance(health_score, (int, float)):
        lines.append(f"- prior_health_score: {float(health_score):.2f}")
    health_trend = prior.get("health_trend")
    if isinstance(health_trend, str) and health_trend.strip():
        lines.append(f"- prior_health_trend: {health_trend.strip()}")

    patterns = prior.get("recurring_patterns")
    if isinstance(patterns, list) and patterns:
        lines.append("- prior_recurring_patterns (carry forward if still true):")
        for entry in patterns[:_MAX_LIST_ITEMS]:
            if not isinstance(entry, dict):
                continue
            pattern = _opt_str(entry.get("pattern"))
            ids = entry.get("evidence_round_ids")
            if not pattern:
                continue
            ids_str = ", ".join(
                str(i) for i in (ids if isinstance(ids, list) else []) if i
            )
            lines.append(f"    * {pattern} [evidence: {ids_str or '(none)'}]")
    else:
        lines.append("- prior_recurring_patterns: (none)")

    wins = prior.get("relationship_wins")
    if isinstance(wins, list) and wins:
        lines.append("- prior_relationship_wins (carry forward if still true):")
        for entry in wins[:_MAX_LIST_ITEMS]:
            if not isinstance(entry, dict):
                continue
            text = _opt_str(entry.get("win"))
            if not text:
                continue
            lines.append(f"    * {text}")
    else:
        lines.append("- prior_relationship_wins: (none)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat-template messages
# ---------------------------------------------------------------------------


def build_pulse_messages(instruction: str) -> list[dict]:
    """Build the OpenAI-style messages list for one pulse call.

    Single user-side instruction at index 0 — same shape as Steps 04 /
    05 / 06 / 08 / 09 / 10. No system role for prompt-based JSON; the
    instruction goes in the user turn so the chat template lays it
    out as a plain request the model answers directly.

    Pure function — kept separate from :func:`generate_pulse` so the
    message shape can be unit-tested without loading transformers.
    """
    return [{"role": "user", "content": instruction}]


# ---------------------------------------------------------------------------
# Payload coercion
# ---------------------------------------------------------------------------


def _coerce_round_summary(
    raw: Any,
    *,
    round_record: dict,
    index: int,
) -> dict:
    """Coerce one ``round_summaries`` entry against the input round.

    ``round_id`` and ``started_at`` are copied verbatim from the input
    round record — the model is NOT trusted to re-emit them, same
    defence-in-depth pattern as Step 10's ``title`` /
    ``predicted_resistance_type`` forcing.

    ``goal_status`` runs through the round's authoritative aftermath
    first; the model's value is only consulted when the aftermath has
    no signal (``unknown``).

    ``prediction_accuracy`` defaults to the aftermath's value when the
    model omits or garbles it.

    ``headline`` is the one field the model fully owns. Required and
    non-empty; raises :class:`PulseError` when missing.
    """
    raw_dict = raw if isinstance(raw, dict) else {}

    aftermath_status = _round_goal_status_from_aftermath(round_record)
    if aftermath_status != "unknown":
        # Aftermath is authoritative on a round it scored.
        goal_status = aftermath_status
    else:
        # No aftermath signal — fall back to the model's call, then
        # ``"unknown"`` if even that fails to normalise.
        goal_status = (
            _normalize_round_goal_status(raw_dict.get("goal_status"))
            or "unknown"
        )

    aftermath_accuracy = _round_prediction_accuracy_from_aftermath(round_record)
    raw_accuracy = raw_dict.get("prediction_accuracy")
    try:
        # When the model emits a usable number, prefer it (it had to
        # read the aftermath block to write it, so the value is at
        # worst a copy and at best a small correction). On garbage,
        # fall back to the aftermath's own value.
        prediction_accuracy = _clamp_unit(
            raw_accuracy, default=aftermath_accuracy
        )
    except Exception:  # noqa: BLE001 — defensive only
        prediction_accuracy = aftermath_accuracy

    headline = _opt_str(raw_dict.get("headline"))
    if not headline:
        raise PulseError(
            f"round_summaries[{index}]: headline is empty or missing"
        )
    headline = _cap(headline, limit=_MAX_HEADLINE_CHARS)

    return {
        "round_id": _round_id(round_record, index=index),
        "started_at": _round_started_at(round_record),
        "goal_status": goal_status,
        "prediction_accuracy": prediction_accuracy,
        "headline": headline,
    }


def _coerce_recurring_patterns(
    raw: Any,
    *,
    known_round_ids: list[str],
    prior_patterns: Any,
) -> list[dict]:
    """Coerce + accumulate ``recurring_patterns``.

    Filters ``evidence_round_ids`` to known ids only (same defence-in-
    depth shape as Step 10's ``did_not_occur`` forcing empty evidence
    — the schema's ``minItems: 2`` on ``evidence_round_ids`` is the
    very definition of "recurring"; an entry that can't cite two
    distinct rounds isn't recurring and gets dropped silently).

    Prior patterns are merged in at the front (preserving the order
    in which they were observed); new patterns are appended. Same
    accumulation pattern as PersonVault's qualitative lists, capped
    at :data:`_MAX_LIST_ITEMS` total entries.
    """
    known = set(known_round_ids)

    def _normalise(raw_list: Any) -> list[dict]:
        if not isinstance(raw_list, list):
            return []
        out: list[dict] = []
        seen_keys: set[str] = set()
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            pattern = _opt_str(entry.get("pattern"))
            ids = entry.get("evidence_round_ids")
            if not pattern or not isinstance(ids, list):
                continue
            # Filter to known ids; dedup within the entry. Preserve
            # input order so the model's deliberate ordering survives.
            unique_ids: list[str] = []
            seen_ids: set[str] = set()
            for rid in ids:
                if not isinstance(rid, str):
                    continue
                rid_clean = rid.strip()
                if not rid_clean or rid_clean not in known:
                    continue
                if rid_clean in seen_ids:
                    continue
                seen_ids.add(rid_clean)
                unique_ids.append(rid_clean)
            if len(unique_ids) < 2:
                LOG.info(
                    "Dropping recurring_pattern with <2 known evidence "
                    "round_ids: pattern=%r, ids=%r",
                    pattern,
                    ids,
                )
                continue
            key = pattern.strip().lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(
                {
                    "pattern": _cap(pattern, limit=_MAX_PATTERN_CHARS),
                    "evidence_round_ids": unique_ids,
                }
            )
        return out

    new_patterns = _normalise(raw)
    prior_cleaned = _normalise(prior_patterns)

    # Merge: prior first (oldest observations), new last (recent
    # observations). Dedup on pattern text (case-insensitive). Cap
    # tails-first — drop the oldest priors when capped, same shape
    # as :func:`_merge_string_lists`.
    combined: list[dict] = []
    seen: set[str] = set()
    for entry in prior_cleaned + new_patterns:
        key = entry["pattern"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(entry)
    if len(combined) > _MAX_LIST_ITEMS:
        combined = combined[-_MAX_LIST_ITEMS:]
    return combined


def _coerce_emerging_concerns(
    raw: Any,
    *,
    known_round_ids: list[str],
) -> list[dict]:
    """Coerce ``emerging_concerns``.

    Each entry must carry a non-empty ``concern`` string; optional
    ``round_id`` is validated against ``known_round_ids`` and dropped
    silently when unknown (a concern attached to a round that doesn't
    exist in this pulse is unverifiable). Malformed entries dropped
    silently. Rebuilt fresh each call — emerging concerns are
    point-in-time, not cumulative.
    """
    if not isinstance(raw, list):
        return []
    known = set(known_round_ids)
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        concern = _opt_str(entry.get("concern"))
        if not concern:
            continue
        item: dict[str, Any] = {
            "concern": _cap(concern, limit=_MAX_CONCERN_CHARS),
        }
        round_id = _opt_str(entry.get("round_id"))
        if round_id and round_id in known:
            item["round_id"] = round_id
        out.append(item)
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _coerce_relationship_wins(
    raw: Any,
    *,
    known_round_ids: list[str],
    prior_wins: Any,
) -> list[dict]:
    """Coerce + accumulate ``relationship_wins``.

    Optional ``round_id`` validated against ``known_round_ids`` and
    dropped when unknown. Prior wins merged in at the front; new
    wins appended. Dedup on the win text (case-insensitive), cap at
    :data:`_MAX_LIST_ITEMS`. Same accumulation pattern as
    PersonVault's qualitative lists.
    """
    known = set(known_round_ids)

    def _normalise(raw_list: Any) -> list[dict]:
        if not isinstance(raw_list, list):
            return []
        out: list[dict] = []
        seen: set[str] = set()
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            win = _opt_str(entry.get("win"))
            if not win:
                continue
            key = win.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, Any] = {
                "win": _cap(win, limit=_MAX_WIN_CHARS),
            }
            round_id = _opt_str(entry.get("round_id"))
            if round_id and round_id in known:
                item["round_id"] = round_id
            out.append(item)
        return out

    new_wins = _normalise(raw)
    prior_cleaned = _normalise(prior_wins)

    combined: list[dict] = []
    seen: set[str] = set()
    for entry in prior_cleaned + new_wins:
        key = entry["win"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(entry)
    if len(combined) > _MAX_LIST_ITEMS:
        combined = combined[-_MAX_LIST_ITEMS:]
    return combined


def _resolve_person_id(cfg: PulseConfig, prior_pulse: dict) -> str:
    """Resolve the canonical person_id for this pulse.

    Priority order: ``cfg.person_id`` > ``cfg.person_profile["person_id"]``
    > ``prior_pulse["person_id"]``. Raises :class:`PulseError` when
    nothing resolves — pulse is per-relationship and cannot exist
    without an id.
    """
    candidates = [
        cfg.person_id,
        cfg.person_profile.get("person_id")
        if isinstance(cfg.person_profile, dict)
        else None,
        prior_pulse.get("person_id"),
    ]
    for c in candidates:
        cleaned = _opt_str(c)
        if cleaned:
            return cleaned
    raise PulseError(
        "Cannot resolve person_id — pass cfg.person_id or include "
        "person_id in cfg.person_profile / cfg.prior_pulse"
    )


def _resolve_person_name(cfg: PulseConfig, prior_pulse: dict) -> str:
    if isinstance(cfg.person_profile, dict):
        name = _opt_str(cfg.person_profile.get("name"))
        if name:
            return name
    prior_name = _opt_str(prior_pulse.get("person_name"))
    if prior_name:
        return prior_name
    return DEFAULT_PERSON_NAME


def _resolve_relationship_type(cfg: PulseConfig, prior_pulse: dict) -> str:
    """Resolve relationship_type. Validates against the schema enum;
    falls back to ``DEFAULT_RELATIONSHIP_TYPE`` on unknown values."""
    candidates = [
        cfg.person_profile.get("relationship_type")
        if isinstance(cfg.person_profile, dict)
        else None,
        prior_pulse.get("relationship_type"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip().lower() in ALLOWED_RELATIONSHIP_TYPES:
            return c.strip().lower()
    return DEFAULT_RELATIONSHIP_TYPE


def _coerce_pulse_payload(
    payload: Any,
    *,
    cfg: PulseConfig,
    prior_pulse: dict,
) -> dict:
    """Turn the model's JSON output into a schema-conforming Pulse dict.

    System-contract enforcement (in code, not prompt — same rationale
    as Step 04's ``whisper_prompt=None for other`` rule, Step 07's
    ``persona_name`` forcing, Step 08's ``scenario_id`` forcing, Step
    09's ``_APOLOGY_CUE_RE``, and Step 10's
    ``_TRANSCRIPT_META_PREFIX_RE`` / pre-mortem-input forcing):

    * Top-level payload must be a dict — anything else raises so the
      retry path takes over.
    * ``person_id`` forced from input chain (never from model output).
    * ``version`` bumped from ``prior.version + 1`` (or ``1`` cold-start).
    * ``round_count`` derived from ``len(round_summaries)``.
    * ``round_summaries`` always exactly one entry per input round,
      in input order; ``round_id`` and ``started_at`` copied from
      input verbatim.
    * ``health_trend`` enum-normalised. When every round has
      ``goal_status == "unknown"``, the runtime forces
      ``insufficient_data`` regardless of model output.
    * ``health_score`` clamped to ``[0, 1]``.
    * ``recurring_patterns`` filtered + accumulated; ``<2`` evidence
      ids → dropped.
    * ``emerging_concerns`` rebuilt fresh; optional ``round_id``
      validated.
    * ``relationship_wins`` filtered + accumulated; optional
      ``round_id`` validated.
    * ``next_round_focus`` required and non-empty.

    Raises :class:`PulseError` for any unrepairable input.
    """
    if not isinstance(payload, dict):
        raise PulseError(
            f"pulse payload not an object: {type(payload).__name__}"
        )

    rounds = cfg.rounds
    if len(rounds) < ROUND_COUNT_MIN:
        raise PulseError(
            f"pulse needs at least {ROUND_COUNT_MIN} rounds; got {len(rounds)}"
        )

    raw_summaries = payload.get("round_summaries")
    if not isinstance(raw_summaries, list):
        raise PulseError(
            "round_summaries must be a list, got "
            f"{type(raw_summaries).__name__}"
        )

    # Always build exactly one summary per input round, in input order.
    # If the model emitted fewer entries, the missing ones are passed
    # as ``{}`` so the per-entry coercer raises a clean error (which
    # the retry path picks up). Extra entries beyond the round count
    # are ignored — the input is the source of truth for round count.
    round_summaries: list[dict] = []
    for i, round_record in enumerate(rounds):
        raw_entry = raw_summaries[i] if i < len(raw_summaries) else {}
        round_summaries.append(
            _coerce_round_summary(
                raw_entry,
                round_record=round_record,
                index=i,
            )
        )

    known_round_ids = [s["round_id"] for s in round_summaries]

    # Override health_trend to "insufficient_data" when every round is
    # unknown — the model frequently picks "stable" with no signal,
    # which is wrong. Same defence-in-depth shape as Step 09's
    # ``_APOLOGY_CUE_RE``: enforce the system contract in code.
    all_unknown = all(
        s["goal_status"] == "unknown" for s in round_summaries
    )
    if all_unknown:
        health_trend = "insufficient_data"
    else:
        health_trend = _normalize_health_trend(payload.get("health_trend"))
        if health_trend is None:
            raise PulseError(
                f"health_trend {payload.get('health_trend')!r} not in enum "
                f"{ALLOWED_HEALTH_TRENDS}"
            )

    health_score = _clamp_unit(payload.get("health_score"))

    trajectory_summary = _opt_str(payload.get("trajectory_summary"))
    if not trajectory_summary:
        raise PulseError("trajectory_summary is empty or missing")
    trajectory_summary = _cap(trajectory_summary, limit=_MAX_SUMMARY_CHARS)

    recurring_patterns = _coerce_recurring_patterns(
        payload.get("recurring_patterns"),
        known_round_ids=known_round_ids,
        prior_patterns=prior_pulse.get("recurring_patterns"),
    )

    emerging_concerns = _coerce_emerging_concerns(
        payload.get("emerging_concerns"),
        known_round_ids=known_round_ids,
    )

    relationship_wins = _coerce_relationship_wins(
        payload.get("relationship_wins"),
        known_round_ids=known_round_ids,
        prior_wins=prior_pulse.get("relationship_wins"),
    )

    next_step = _opt_str(payload.get("next_step_recommendation"))
    if not next_step:
        raise PulseError("next_step_recommendation is empty or missing")
    next_step = _cap(next_step, limit=_MAX_RECOMMENDATION_CHARS)

    person_id = _resolve_person_id(cfg, prior_pulse)
    person_name = _resolve_person_name(cfg, prior_pulse)
    relationship_type = _resolve_relationship_type(cfg, prior_pulse)
    prior_version = int(prior_pulse.get("version", 0) or 0)

    return {
        "person_id": person_id,
        "person_name": person_name,
        "relationship_type": relationship_type,
        "version": prior_version + 1,
        "round_count": len(round_summaries),
        "health_score": health_score,
        "health_trend": health_trend,
        "trajectory_summary": trajectory_summary,
        "round_summaries": round_summaries,
        "recurring_patterns": recurring_patterns,
        "emerging_concerns": emerging_concerns,
        "relationship_wins": relationship_wins,
        "next_step_recommendation": next_step,
        "pulse_id": f"pulse_{uuid.uuid4().hex[:12]}",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(cfg: PulseConfig, *, prior_pulse: dict) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    if not isinstance(cfg.person_profile, dict) or not cfg.person_profile:
        raise PulseError(
            "PulseConfig.person_profile is required — pulse is per-"
            "relationship and needs a profile to attach to"
        )
    if len(cfg.rounds) < ROUND_COUNT_MIN:
        raise PulseError(
            f"PulseConfig.rounds needs at least {ROUND_COUNT_MIN} entries; "
            f"got {len(cfg.rounds)}"
        )
    rounds_block = format_rounds_block(cfg.rounds)
    if not rounds_block:
        raise PulseError(
            "PulseConfig.rounds renders to an empty block — every "
            "round record must be a dict"
        )
    person_name = _resolve_person_name(cfg, prior_pulse)
    relationship_type = _resolve_relationship_type(cfg, prior_pulse)
    return load_prompt(cfg.prompt_name).render(
        person_name=person_name,
        relationship_type=relationship_type,
        person_profile_block=format_persona_profile_block(cfg.person_profile),
        rounds_block=rounds_block,
        prior_pulse_block=format_prior_pulse_block(cfg.prior_pulse),
    )


def _build_generation_config(
    cfg: PulseConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def generate_pulse(
    processor: Any,
    model: Any,
    *,
    cfg: Optional[PulseConfig] = None,
) -> dict:
    """Produce a relationship pulse from N practice rounds.

    Inference path mirrors the other intelligence components: greedy
    first (deterministic, fastest), then a single retry with light
    sampling (:data:`RETRY_SAMPLING`) on parse / validation failure.
    If both attempts fail, raises :class:`PulseError` with an
    ``attempts`` attribute carrying each attempt's raw output
    (truncated to 500 chars) for diagnostics.

    Returns a dict matching ``data/schemas/pulse.schema.json`` plus
    runtime-attached ``pulse_id`` and ``updated_at`` fields (the
    schema permits additional properties via standard JSON schema
    semantics — no ``additionalProperties: false`` is set).
    """
    cfg = cfg or PulseConfig()
    prior_pulse = cfg.prior_pulse if isinstance(cfg.prior_pulse, dict) else {}

    instruction = _resolve_instruction(cfg, prior_pulse=prior_pulse)
    messages = build_pulse_messages(instruction)

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
            return _coerce_pulse_payload(
                payload,
                cfg=cfg,
                prior_pulse=prior_pulse,
            )
        except (JsonParseError, PulseError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "generate_pulse attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = PulseError(
        f"pulse generation failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err
