"""Aftermath — Phase 4 / Step 10 of Tough Talks.

One-shot analytical pass that compares a pre-mortem (the PLAN, from
Step 08) to the actual practice transcript (the REALITY, from Step 07)
and returns a structured plan-vs-reality comparison matching
``data/schemas/aftermath.schema.json``:

* ``goal_outcome`` — did the user get what they came for? (enum +
  one-or-two-sentence summary)
* ``scenario_outcomes[]`` — one entry per predicted scenario (always
  exactly three, in the order of the pre-mortem input). Each entry
  carries the predicted ``title`` / ``predicted_resistance_type`` from
  the pre-mortem AS-IS (forced in code, not trusted to the model) plus
  the model's ``match_quality`` enum, the derived ``materialized``
  boolean, transcript-grounded ``evidence``, and free-text ``notes``.
* ``unforeseen_moments[]`` — risks and opportunities the pre-mortem did
  NOT predict but actually surfaced in the round.
* ``prediction_accuracy`` — single number in ``[0, 1]`` summarising how
  well the pre-mortem matched reality.
* ``next_round_focus`` — single-sentence pointer for the next round's
  practice target.

Architectural shape mirrors Steps 04 / 05 / 06 / 08 / 09:

* Text-only Gemma 4 (``LoadConfig(multimodal=False)``).
* Single prompt-based JSON call — no rolling history. The whole
  finished transcript, the whole pre-mortem, and the optional debrief
  are fed in once.
* Two-shot retry: greedy first, then a single light-sampling pass
  (:data:`RETRY_SAMPLING`) on JSON / validation failure.
* System-contract enforcement in code (same defence-in-depth pattern
  as Step 04's ``whisper_prompt=None for other`` rule, Step 07's
  ``persona_name`` forcing, Step 08's ``scenario_id`` forcing, and
  Step 09's ``_APOLOGY_CUE_RE``):
  - Exactly three ``scenario_outcomes``, one per pre-mortem scenario.
  - ``scenario_id`` is forced to ``index + 1`` regardless of model output.
  - ``title`` and ``predicted_resistance_type`` are copied verbatim from
    the pre-mortem input — the model is NOT trusted to re-emit them
    correctly, even though the prompt asks it to. The model frequently
    paraphrases titles or drops the snake_case enum form into lowercase
    free text; pinning to the pre-mortem input removes that drift.
  - ``match_quality`` runs through :func:`_normalize_match_quality` so
    casing / hyphen / near-synonym drift is silently corrected.
  - ``materialized`` is derived from ``match_quality`` in code, NOT
    trusted to the model — ``direct_hit`` / ``partial`` → ``True``,
    ``did_not_occur`` → ``False``. Disagreements between the model's
    boolean and its match_quality are resolved by the enum.
  - When ``match_quality`` is ``did_not_occur``, ``evidence`` is forced
    to the empty string regardless of model output — invented evidence
    for a missed prediction is exactly the failure mode the prompt
    forbids.
  - When ``match_quality`` is ``direct_hit`` or ``partial``, missing /
    empty ``evidence`` is treated as a failure mode that the retry
    path should re-roll; if both attempts still leak through, the
    entry is downgraded to ``did_not_occur`` so the output stays valid.
  - ``unforeseen_moments[].kind`` runs through a small enum normaliser;
    optional ``turn`` is clamped to ``[1, user_turn_count]`` when
    present.
  - ``prediction_accuracy`` clamped to ``[0, 1]``.
  - Free-text fields capped to defend against multi-paragraph drift.

Inputs accepted (all required have explicit defaults / sentinels):

* ``premortem`` — full pre-mortem dict from :mod:`premortem` (or its
  inner ``failure_scenarios`` list). Mandatory — the aftermath does
  not make sense without a plan to compare against.
* ``transcript`` — the practice transcript, same shape as the input to
  :func:`backend.core._runtime.debrief.generate_debrief`.
* ``user_goal`` — what the user was trying to achieve. Falls back to
  :data:`DEFAULT_USER_GOAL`.
* ``person_profile`` — full PersonVault dict from Step 06 (or its inner
  ``profile`` block). Optional — grounds ``notes`` and
  ``unforeseen_moments`` against the counterparty's observed habits.
* ``debrief`` — the debrief dict from :mod:`debrief`. Optional. When
  present, the prompt surfaces ``one_fix_next_time`` and headline
  counts so the aftermath can reconcile the post-round coaching read
  with the pre-round predictions.

The five rendered context blocks all degrade to ``(no … provided)``
sentinels when missing so the prompt template is never malformed.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .audio import _extract_text
from .debrief import count_user_turns, format_practice_transcript
from .generation import GenerationConfig, chat
from .parsing import JsonParseError, parse_json
from .persona_sim import (
    DEFAULT_USER_GOAL,
    _normalize_resistance_type,
    format_persona_profile_block,
)
from .premortem import REQUIRED_SCENARIO_COUNT
from .prompts import load_prompt

__all__ = [
    "ALLOWED_GOAL_STATUSES",
    "ALLOWED_MATCH_QUALITIES",
    "ALLOWED_UNFORESEEN_KINDS",
    "AftermathConfig",
    "AftermathError",
    "DEFAULT_AFTERMATH_PROMPT_NAME",
    "RETRY_SAMPLING",
    "build_aftermath_messages",
    "format_debrief_block",
    "format_premortem_block",
    "generate_aftermath",
]

LOG = logging.getLogger(__name__)

DEFAULT_AFTERMATH_PROMPT_NAME = "aftermath"

# Mirrors the enums at data/schemas/aftermath.schema.json. Unit tests
# pin these tuples against the on-disk schema so drift fails loudly.
ALLOWED_MATCH_QUALITIES: tuple[str, ...] = (
    "direct_hit",
    "partial",
    "did_not_occur",
)
ALLOWED_UNFORESEEN_KINDS: tuple[str, ...] = ("risk", "opportunity")
ALLOWED_GOAL_STATUSES: tuple[str, ...] = (
    "achieved",
    "partial",
    "not_achieved",
)

# Near-synonym maps. Same defence-in-depth shape as Step 06's
# ``_COMMUNICATION_STYLE_SYNONYMS`` and Step 07's
# ``_RESISTANCE_TYPE_SYNONYMS``: the model reaches for words near its
# training distribution rather than the exact enum codes. Accept the
# obvious synonyms here; truly unknown labels still return ``None`` so
# the retry path fires.
_MATCH_QUALITY_SYNONYMS: dict[str, str] = {
    # Direct hit cluster
    "direct": "direct_hit",
    "direct hit": "direct_hit",
    "exact": "direct_hit",
    "exact match": "direct_hit",
    "exact_match": "direct_hit",
    "full": "direct_hit",
    "full match": "direct_hit",
    "full_match": "direct_hit",
    "hit": "direct_hit",
    "match": "direct_hit",
    "matched": "direct_hit",
    "yes": "direct_hit",
    # Partial cluster
    "partial match": "partial",
    "partial_match": "partial",
    "partly": "partial",
    "somewhat": "partial",
    "mixed": "partial",
    "close": "partial",
    # Did-not-occur cluster
    "miss": "did_not_occur",
    "missed": "did_not_occur",
    "no match": "did_not_occur",
    "no_match": "did_not_occur",
    "did not occur": "did_not_occur",
    "didnt occur": "did_not_occur",
    "didnt_occur": "did_not_occur",
    "did_not_happen": "did_not_occur",
    "did not happen": "did_not_occur",
    "none": "did_not_occur",
    "no": "did_not_occur",
    "did_not_materialize": "did_not_occur",
    "did not materialize": "did_not_occur",
    "did_not_materialise": "did_not_occur",
    "did not materialise": "did_not_occur",
    "not_materialized": "did_not_occur",
    "not materialized": "did_not_occur",
    "not_materialised": "did_not_occur",
    "not materialised": "did_not_occur",
}

_UNFORESEEN_KIND_SYNONYMS: dict[str, str] = {
    "negative": "risk",
    "threat": "risk",
    "problem": "risk",
    "downside": "risk",
    "hazard": "risk",
    "positive": "opportunity",
    "win": "opportunity",
    "opening": "opportunity",
    "upside": "opportunity",
    "good": "opportunity",
}

_GOAL_STATUS_SYNONYMS: dict[str, str] = {
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
}

# Light sampling for the retry path — same temperature / top_p / top_k
# as the other intelligence runtimes. Pinned by the unit test so drift
# is loud.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}

# Caps on free-text fields. The schema has no length constraint; these
# guard against the model running into multi-paragraph narration
# instead of the requested coaching-read shape.
_MAX_LIST_ITEMS = 8
_MAX_SUMMARY_CHARS = 400
_MAX_EVIDENCE_CHARS = 400
_MAX_NOTES_CHARS = 600
_MAX_DESCRIPTION_CHARS = 400
_MAX_FOCUS_CHARS = 400
_MAX_TITLE_CHARS = 120

_NO_PREMORTEM_SENTINEL = "(no pre-mortem provided)"
_NO_DEBRIEF_SENTINEL = (
    "(no post-round debrief provided — base the aftermath on the "
    "pre-mortem and transcript alone)"
)

# Bracket-prefix render-format leak in ``evidence``. The Step 10 first
# Colab run had both ``enable_thinking`` branches paste the transcript's
# render-format bracket (e.g. ``"Jamie (resistance=deflect,
# escalation=0.60): I told you..."``) into ``evidence``, even with the
# prompt asking for "a quote or a turn pointer". Defence-in-depth:
# strip the prefix in code AS WELL AS in the prompt. Two recognised
# shapes match what :func:`format_practice_transcript` renders:
#
#   * ``Jamie (resistance=deflect, escalation=0.60)`` — persona turn,
#     with or without the surrounding ``[...]``, with or without a
#     trailing colon. ``escalation=...`` is optional because
#     :func:`format_practice_transcript` drops it when the turn has
#     no numeric escalation level.
#   * ``[USER 1]`` / ``[OTHER]`` — bracket-required turns.
#
# The regex only strips when one of those exact shapes is present at
# the start of the string. A natural turn pointer the model wrote in
# its own words (e.g. ``"Jamie's first reply — ..."``) is left
# untouched. Same defence-in-depth pattern as Step 09's
# ``_APOLOGY_CUE_RE``: enforce the system contract in code so a
# future model regression cannot leak the render format back into
# the aftermath payload.
_TRANSCRIPT_META_PREFIX_RE: re.Pattern[str] = re.compile(
    r"""^\s*
    (?:
        # Persona-style: optional outer bracket + name +
        # ( resistance = X [, escalation = Y.YY ] )
        \[?\s*
        [A-Za-z][\w'.\- ]{0,40}?
        \s*
        \(\s* resistance \s* = \s* \w+ \s*
        (?:, \s* escalation \s* = \s* [\d.]+ \s*)?
        \)
        \s*\]?
        |
        # USER N / OTHER form: brackets required so a stray
        # ``USER 2`` in the middle of a quote is not eaten.
        \[ \s* (?: USER \s+ \d+ | OTHER ) \s* \]
    )
    \s* [:\-—]? \s*
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _strip_transcript_meta_prefix(text: str) -> str:
    """Strip a leaked transcript render-format bracket from the start.

    The runtime's :func:`format_practice_transcript` renders each turn
    with a metadata bracket the model occasionally copies verbatim
    into ``evidence``. This helper removes that bracket (and a single
    optional separator after it) when present at the start of the
    string and leaves the rest alone. Returns the cleaned text
    stripped of leading / trailing whitespace.
    """
    cleaned = _TRANSCRIPT_META_PREFIX_RE.sub("", text, count=1).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class AftermathError(ValueError):
    """The model emitted a malformed aftermath payload — wrong scenario
    count, missing required field, unrecognisable ``match_quality``, etc."""


@dataclass
class AftermathConfig:
    """Run-time inputs for :func:`generate_aftermath`.

    ``premortem`` and ``transcript`` are the only fields without
    automatic empty-sentinel rendering — the aftermath does not make
    sense without a plan to compare against and a reality to compare it
    to. Both are checked at :func:`_resolve_instruction` time and raise
    :class:`AftermathError` when missing or unusable.

    ``enable_thinking`` exposes Gemma 4's thinking channel. Defaults to
    ``False`` per the rule in ``knowledge/phases/rules.md``; the open
    hypothesis is that thinking helps on analytical payloads that
    cross-reference multiple speaker roles in one object — which is
    exactly the shape of the aftermath (predicted resistance, actual
    transcript content, the user's response to it). N=3 across two
    task families is currently suggestive; the notebook A/B run on
    Step 10 is the third analytical-task datapoint that promotes or
    demotes ``[[hypothesis-persona-thinking-helps]]``.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from the five
    context blocks. Pass an explicit ``instruction`` only for ablation
    experiments.
    """

    premortem: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    user_goal: str = DEFAULT_USER_GOAL
    person_profile: dict = field(default_factory=dict)
    debrief: dict = field(default_factory=dict)
    max_new_tokens: int = 1024
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_AFTERMATH_PROMPT_NAME
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Helpers — extracting the pre-mortem scenarios in canonical order
# ---------------------------------------------------------------------------


def _extract_premortem_scenarios(premortem: Any) -> list[dict]:
    """Return the pre-mortem's ``failure_scenarios`` list, or ``[]``.

    Accepts either the full pre-mortem dict (with a ``failure_scenarios``
    key) or the raw list. The list is returned in the order the
    pre-mortem emitted it — index 0 is scenario 1, index 1 is scenario
    2, etc. The runtime forces ``scenario_id`` from the index, so the
    caller never has to trust either the pre-mortem's or the model's
    ``scenario_id`` values.
    """
    if isinstance(premortem, list):
        return [s for s in premortem if isinstance(s, dict)]
    if isinstance(premortem, dict):
        raw = premortem.get("failure_scenarios")
        if isinstance(raw, list):
            return [s for s in raw if isinstance(s, dict)]
    return []


def _scenario_title(scenario: dict, *, fallback_id: int) -> str:
    title = scenario.get("title")
    if isinstance(title, str) and title.strip():
        return _cap(title.strip(), limit=_MAX_TITLE_CHARS)
    return f"Scenario {fallback_id}"


def _scenario_predicted_resistance(scenario: dict) -> str:
    """Pull the predicted ``resistance_type`` out of a pre-mortem scenario.

    Looks at ``scenario["simulation_parameters"]["resistance_type"]``
    first (the canonical location in the pre-mortem schema), then
    falls back to ``scenario["resistance_type"]`` for flatter inputs.
    Runs whatever it finds through the persona-sim normaliser so the
    aftermath enum stays a single source of truth across Steps 07 /
    08 / 10. Returns ``"deflect"`` as the safe default if nothing
    parses — the runtime is rejecting the scenario at
    :func:`_resolve_instruction` time if the pre-mortem doesn't carry
    a valid resistance type.
    """
    params = scenario.get("simulation_parameters")
    candidate: Any = None
    if isinstance(params, dict):
        candidate = params.get("resistance_type")
    if not candidate:
        candidate = scenario.get("resistance_type")
    normalised = _normalize_resistance_type(candidate)
    return normalised or "deflect"


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def format_premortem_block(premortem: Any) -> str:
    """Render a pre-mortem dict as a numbered text block for the prompt.

    Each scenario is rendered with its ``scenario_id``, ``title``,
    predicted ``resistance_type`` / ``escalation_ceiling`` /
    ``destabilization_risk``, ``likely_trigger``, the opponent's
    predicted ``opening_move``, and a one-line ``description``. The
    aftermath model needs all of these to decide whether reality
    matched the plan — particularly the ``opening_move`` and
    ``description``, which are the shape-of-the-play the model is
    comparing against the transcript.

    Returns a sentinel ``(no pre-mortem provided)`` string when the
    input is empty so the prompt template is never malformed.
    """
    scenarios = _extract_premortem_scenarios(premortem)
    if not scenarios:
        return _NO_PREMORTEM_SENTINEL

    lines: list[str] = []
    for i, scenario in enumerate(scenarios, start=1):
        title = _scenario_title(scenario, fallback_id=i)
        resistance = _scenario_predicted_resistance(scenario)
        params = scenario.get("simulation_parameters")
        if not isinstance(params, dict):
            params = {}
        ceiling = params.get("escalation_ceiling")
        opening = params.get("opening_move")
        risk = scenario.get("destabilization_risk")
        trigger = scenario.get("likely_trigger")
        description = scenario.get("description")

        ceiling_str = (
            f"{float(ceiling):.2f}"
            if isinstance(ceiling, (int, float))
            else "(unknown)"
        )
        risk_str = (
            f"{float(risk):.2f}"
            if isinstance(risk, (int, float))
            else "(unknown)"
        )

        lines.append(
            f"S{i}. {title} — predicted resistance_type={resistance}, "
            f"escalation_ceiling={ceiling_str}, destabilization_risk={risk_str}"
        )
        if isinstance(description, str) and description.strip():
            lines.append(f"    description: {description.strip()}")
        if isinstance(trigger, str) and trigger.strip():
            lines.append(f"    likely_trigger: {trigger.strip()}")
        if isinstance(opening, str) and opening.strip():
            lines.append(f"    predicted opening_move: {opening.strip()!r}")
    return "\n".join(lines)


def format_debrief_block(debrief: Any) -> str:
    """Render the optional post-round debrief as a compact text block.

    The aftermath compares pre-mortem (plan) to transcript (reality);
    the debrief is the post-round coaching read, which sometimes
    contains observations (``one_fix_next_time``, headline win/loss
    counts) that sharpen the aftermath's ``next_round_focus``. Only
    the high-signal fields are surfaced — the full debrief lists are
    available downstream if the model needs them, and dumping
    everything here would blow up the prompt.

    Returns the ``no debrief`` sentinel when input is empty so the
    prompt template is never malformed on cold-start.
    """
    if not isinstance(debrief, dict) or not debrief:
        return _NO_DEBRIEF_SENTINEL

    one_fix = debrief.get("one_fix_next_time")
    ground_lost = debrief.get("ground_lost")
    over_apologies = debrief.get("over_apologies")
    missed_openings = debrief.get("missed_openings")
    wins = debrief.get("wins")

    def _count(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    lines = [
        f"- one_fix_next_time: {one_fix.strip()!r}"
        if isinstance(one_fix, str) and one_fix.strip()
        else "- one_fix_next_time: (none provided)",
        f"- wins: {_count(wins)}",
        f"- ground_lost: {_count(ground_lost)}",
        f"- over_apologies: {_count(over_apologies)}",
        f"- missed_openings: {_count(missed_openings)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat-template messages
# ---------------------------------------------------------------------------


def build_aftermath_messages(instruction: str) -> list[dict]:
    """Build the OpenAI-style messages list for one aftermath call.

    Single user-side instruction at index 0 — same shape as Steps 04 /
    05 / 06 / 08 / 09. No system role for prompt-based JSON; the
    instruction goes in the user turn so the chat template lays it
    out as a plain request the model answers directly.

    Pure function — kept separate from :func:`generate_aftermath` so
    the message shape can be unit-tested without loading transformers.
    """
    return [{"role": "user", "content": instruction}]


# ---------------------------------------------------------------------------
# Payload coercion
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


def _normalize_enum(value: Any, *, allowed: tuple[str, ...], synonyms: dict[str, str]) -> Optional[str]:
    """Coerce a model-emitted enum value to a canonical entry.

    Handles three sources of drift between model output and the schema
    enum, same shape as :func:`_normalize_resistance_type`:

    * **Casing** — ``"Direct_Hit"`` → ``"direct_hit"``.
    * **Word separator** — ``"direct-hit"`` / ``"direct hit"`` →
      ``"direct_hit"``.
    * **Near-synonyms** — ``"exact match"`` → ``"direct_hit"``,
      ``"missed"`` → ``"did_not_occur"``, etc. via the supplied
      synonym map.

    Returns the canonical enum value, or ``None`` if the input is
    unrecognisable.
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


def _normalize_match_quality(value: Any) -> Optional[str]:
    return _normalize_enum(
        value,
        allowed=ALLOWED_MATCH_QUALITIES,
        synonyms=_MATCH_QUALITY_SYNONYMS,
    )


def _normalize_unforeseen_kind(value: Any) -> Optional[str]:
    return _normalize_enum(
        value,
        allowed=ALLOWED_UNFORESEEN_KINDS,
        synonyms=_UNFORESEEN_KIND_SYNONYMS,
    )


def _normalize_goal_status(value: Any) -> Optional[str]:
    return _normalize_enum(
        value,
        allowed=ALLOWED_GOAL_STATUSES,
        synonyms=_GOAL_STATUS_SYNONYMS,
    )


def _coerce_goal_outcome(raw: Any) -> dict:
    """Coerce the ``goal_outcome`` block.

    Required and non-empty. Status is enum-normalised; summary is
    capped and stripped. Raises :class:`AftermathError` rather than
    swallowing — the goal outcome is the single most important
    sentence in the whole payload.
    """
    if not isinstance(raw, dict):
        raise AftermathError(
            f"goal_outcome must be an object, got {type(raw).__name__}"
        )

    status = _normalize_goal_status(raw.get("status"))
    if status is None:
        raise AftermathError(
            f"goal_outcome.status {raw.get('status')!r} not in enum "
            f"{ALLOWED_GOAL_STATUSES}"
        )

    summary = _opt_str(raw.get("summary"))
    if not summary:
        raise AftermathError("goal_outcome.summary is empty or missing")
    summary = _cap(summary, limit=_MAX_SUMMARY_CHARS)

    return {"status": status, "summary": summary}


def _coerce_scenario_outcome(raw: Any, *, scenario_input: dict, scenario_id: int) -> dict:
    """Coerce one ``scenario_outcomes`` entry against the pre-mortem input.

    ``scenario_id``, ``title``, and ``predicted_resistance_type`` come
    from ``scenario_input`` (the pre-mortem scenario at this index),
    NOT from the model — defence-in-depth against the model
    paraphrasing the title or re-typing the enum. The model only
    contributes ``match_quality``, ``evidence``, and ``notes``.

    ``materialized`` is derived from ``match_quality`` in code, not
    trusted to the model.

    When ``match_quality`` is ``did_not_occur``, ``evidence`` is forced
    to the empty string regardless of what the model emitted —
    invented evidence for a missed prediction is the failure mode the
    prompt explicitly forbids. When ``match_quality`` is
    ``direct_hit`` / ``partial`` and the model leaked an empty
    ``evidence``, the entry is downgraded to ``did_not_occur`` so the
    output stays internally consistent.
    """
    raw_dict = raw if isinstance(raw, dict) else {}

    match_quality = _normalize_match_quality(raw_dict.get("match_quality"))
    if match_quality is None:
        raise AftermathError(
            f"scenario {scenario_id}: match_quality "
            f"{raw_dict.get('match_quality')!r} not in enum "
            f"{ALLOWED_MATCH_QUALITIES}"
        )

    notes = _opt_str(raw_dict.get("notes"))
    if not notes:
        raise AftermathError(f"scenario {scenario_id}: notes is empty or missing")
    notes = _cap(notes, limit=_MAX_NOTES_CHARS)

    evidence_raw = _opt_str(raw_dict.get("evidence"))
    if match_quality == "did_not_occur":
        evidence = ""
    else:
        # Strip any leaked transcript render-format bracket BEFORE the
        # empty-after-strip check — a string like
        # ``"Jamie (resistance=deflect, escalation=0.60):"`` looks
        # non-empty until the prefix strips away to nothing, at which
        # point it's effectively an empty evidence and should downgrade
        # the same way an explicitly empty string does.
        cleaned_evidence: Optional[str] = (
            _strip_transcript_meta_prefix(evidence_raw)
            if evidence_raw
            else None
        )
        if not cleaned_evidence:
            LOG.info(
                "Scenario %d emitted match_quality=%r with empty evidence — "
                "downgrading to did_not_occur",
                scenario_id,
                match_quality,
            )
            match_quality = "did_not_occur"
            evidence = ""
        else:
            evidence = _cap(cleaned_evidence, limit=_MAX_EVIDENCE_CHARS)

    materialized = match_quality != "did_not_occur"

    return {
        "scenario_id": scenario_id,
        "title": _scenario_title(scenario_input, fallback_id=scenario_id),
        "predicted_resistance_type": _scenario_predicted_resistance(scenario_input),
        "match_quality": match_quality,
        "materialized": materialized,
        "evidence": evidence,
        "notes": notes,
    }


def _coerce_unforeseen_moments(raw: Any, *, user_turn_count: int) -> list[dict]:
    """Coerce ``unforeseen_moments``.

    Each entry must carry a recognisable ``kind`` enum value and a
    non-empty ``description``; malformed entries are dropped silently
    rather than failing the whole aftermath (one bad unforeseen entry
    should not lose the rest of the payload). Optional ``turn`` is
    clamped to ``[1, user_turn_count]`` when present and dropped when
    unrecoverable so the schema's ``minimum: 1`` constraint is honoured.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        kind = _normalize_unforeseen_kind(entry.get("kind"))
        description = _opt_str(entry.get("description"))
        if kind is None or not description:
            continue
        item: dict[str, Any] = {
            "kind": kind,
            "description": _cap(description, limit=_MAX_DESCRIPTION_CHARS),
        }
        turn = _coerce_optional_turn(entry.get("turn"), user_turn_count=user_turn_count)
        if turn is not None:
            item["turn"] = turn
        out.append(item)
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _coerce_optional_turn(value: Any, *, user_turn_count: int) -> Optional[int]:
    """Coerce an optional ``turn`` integer to ``[1, user_turn_count]``.

    Returns ``None`` for unrecoverable values (non-numeric, missing,
    NaN) and also when ``user_turn_count`` is 0 (an aftermath on an
    empty transcript is rejected upstream, but the helper stays
    defensive).
    """
    if value is None:
        return None
    if user_turn_count <= 0:
        return None
    try:
        turn = int(value)
    except (TypeError, ValueError):
        try:
            turn = int(float(value))
        except (TypeError, ValueError):
            return None
    if turn < 1:
        turn = 1
    if turn > user_turn_count:
        turn = user_turn_count
    return turn


def _coerce_aftermath_payload(
    payload: Any,
    *,
    premortem_scenarios: list[dict],
    user_turn_count: int,
) -> dict:
    """Turn the model's JSON output into a schema-conforming Aftermath dict.

    System-contract enforcement (in code, not prompt — same rationale
    as Step 04's ``whisper_prompt=None for other`` rule, Step 07's
    ``persona_name`` forcing, Step 08's ``scenario_id`` forcing, and
    Step 09's ``_APOLOGY_CUE_RE``):

    * Top-level payload must be a dict — anything else raises so the
      retry path takes over.
    * ``goal_outcome`` required, non-empty, enum-normalised.
    * ``scenario_outcomes`` must be a list of length
      :data:`REQUIRED_SCENARIO_COUNT`. The runtime forces the canonical
      ``scenario_id`` 1 / 2 / 3 sequence regardless of model output.
    * ``title`` and ``predicted_resistance_type`` per scenario are
      copied verbatim from the pre-mortem input — the model is not
      trusted to re-emit them.
    * ``match_quality`` runs through :func:`_normalize_match_quality`
      so casing / synonym drift is silently corrected.
    * ``materialized`` is derived from ``match_quality`` in code.
    * Evidence consistency: ``did_not_occur`` forces empty evidence;
      missing evidence on a ``direct_hit`` / ``partial`` downgrades the
      entry to ``did_not_occur``.
    * ``unforeseen_moments[].kind`` enum-normalised; optional ``turn``
      clamped to ``[1, user_turn_count]``; malformed entries dropped
      silently.
    * ``prediction_accuracy`` clamped to ``[0, 1]``.
    * ``next_round_focus`` required and non-empty.

    Raises :class:`AftermathError` for any unrepairable input.
    """
    if not isinstance(payload, dict):
        raise AftermathError(
            f"aftermath payload not an object: {type(payload).__name__}"
        )

    if len(premortem_scenarios) != REQUIRED_SCENARIO_COUNT:
        raise AftermathError(
            f"aftermath needs exactly {REQUIRED_SCENARIO_COUNT} pre-mortem "
            f"scenarios; got {len(premortem_scenarios)}"
        )

    goal_outcome = _coerce_goal_outcome(payload.get("goal_outcome"))

    raw_outcomes = payload.get("scenario_outcomes")
    if not isinstance(raw_outcomes, list):
        raise AftermathError(
            "scenario_outcomes must be a list, got "
            f"{type(raw_outcomes).__name__}"
        )
    # We force the count and per-entry shape from the pre-mortem input.
    # If the model emitted fewer entries than scenarios, we still build
    # one outcome per scenario — passing the missing entry as ``{}`` so
    # the per-entry coercer raises a clear error (which the retry path
    # picks up). Extra entries beyond REQUIRED_SCENARIO_COUNT are
    # ignored — the pre-mortem is the source of truth for scenario
    # count.
    scenario_outcomes: list[dict] = []
    for i, scenario_input in enumerate(premortem_scenarios):
        raw_entry = raw_outcomes[i] if i < len(raw_outcomes) else {}
        scenario_outcomes.append(
            _coerce_scenario_outcome(
                raw_entry,
                scenario_input=scenario_input,
                scenario_id=i + 1,
            )
        )

    unforeseen_moments = _coerce_unforeseen_moments(
        payload.get("unforeseen_moments"),
        user_turn_count=user_turn_count,
    )

    prediction_accuracy = _clamp_unit(payload.get("prediction_accuracy"))

    next_round_focus = _opt_str(payload.get("next_round_focus"))
    if not next_round_focus:
        raise AftermathError("next_round_focus is empty or missing")
    next_round_focus = _cap(next_round_focus, limit=_MAX_FOCUS_CHARS)

    return {
        "goal_outcome": goal_outcome,
        "scenario_outcomes": scenario_outcomes,
        "unforeseen_moments": unforeseen_moments,
        "prediction_accuracy": prediction_accuracy,
        "next_round_focus": next_round_focus,
        "aftermath_id": f"aftermath_{uuid.uuid4().hex[:12]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(cfg: AftermathConfig, *, premortem_scenarios: list[dict]) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    if len(premortem_scenarios) != REQUIRED_SCENARIO_COUNT:
        raise AftermathError(
            "AftermathConfig.premortem must carry exactly "
            f"{REQUIRED_SCENARIO_COUNT} failure_scenarios; got "
            f"{len(premortem_scenarios)}"
        )
    if count_user_turns(cfg.transcript) <= 0:
        raise AftermathError(
            "AftermathConfig.transcript has no user turns — aftermath needs "
            "the actual practice round to compare against the plan"
        )
    user_goal = (cfg.user_goal or DEFAULT_USER_GOAL).strip() or DEFAULT_USER_GOAL
    return load_prompt(cfg.prompt_name).render(
        user_goal=user_goal,
        person_profile_block=format_persona_profile_block(cfg.person_profile),
        premortem_block=format_premortem_block(cfg.premortem),
        debrief_block=format_debrief_block(cfg.debrief),
        transcript=format_practice_transcript(cfg.transcript),
    )


def _build_generation_config(
    cfg: AftermathConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def generate_aftermath(
    processor: Any,
    model: Any,
    *,
    cfg: Optional[AftermathConfig] = None,
) -> dict:
    """Produce a plan-vs-reality aftermath for a finished practice round.

    Inference path mirrors the other intelligence components: greedy
    first (deterministic, fastest), then a single retry with light
    sampling (:data:`RETRY_SAMPLING`) on parse / validation failure.
    If both attempts fail, raises :class:`AftermathError` with an
    ``attempts`` attribute carrying each attempt's raw output (truncated
    to 500 chars) for diagnostics.

    Returns a dict matching ``data/schemas/aftermath.schema.json`` plus
    runtime-attached ``aftermath_id`` and ``generated_at`` fields (the
    schema permits additional properties).
    """
    cfg = cfg or AftermathConfig()
    premortem_scenarios = _extract_premortem_scenarios(cfg.premortem)
    user_turn_count = count_user_turns(cfg.transcript)

    instruction = _resolve_instruction(cfg, premortem_scenarios=premortem_scenarios)
    messages = build_aftermath_messages(instruction)

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
            return _coerce_aftermath_payload(
                payload,
                premortem_scenarios=premortem_scenarios,
                user_turn_count=user_turn_count,
            )
        except (JsonParseError, AftermathError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "generate_aftermath attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = AftermathError(
        f"aftermath generation failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err
