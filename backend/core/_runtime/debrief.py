"""Post-Round Debrief Engine — Phase 4 / Step 9 of Tough Talks.

One-shot analytical pass that scores a finished practice round and
returns coaching feedback matching ``data/schemas/debrief.schema.json``:

* ``ground_lost`` — turns where the user gave up leverage.
* ``over_apologies`` — user turns containing unnecessary apology cues.
* ``missed_openings`` — turns where the persona left an opening the
  user did not capitalise on; each entry carries a ``better_line`` the
  user could deliver verbatim next time.
* ``wins`` — turns where the user did something that advanced the goal.
* ``one_fix_next_time`` — single sentence, the most important change.

Architectural shape mirrors Steps 04 / 05 / 06 / 08:

* Text-only Gemma 4 (``LoadConfig(multimodal=False)``).
* Single prompt-based JSON call — no rolling history. The whole
  finished transcript is fed in once.
* Two-shot retry: greedy first, then a single light-sampling pass
  (:data:`RETRY_SAMPLING`) on JSON / validation failure.
* System-contract enforcement in code: ``turn`` indices clamped to the
  user-turn range, required string fields rejected when empty, each
  array capped at :data:`_MAX_LIST_ITEMS`, long free-text fields capped.

Inputs accepted (all optional except ``transcript``):

* ``transcript`` — a list of turn dicts. The two shapes the runtime
  knows about:
  - ``{"speaker": "user", "text": "..."}`` — what the user said.
  - ``{"speaker": "persona", "persona_name": "...", "reply": "...",
    "resistance_type": "...", "escalation_level": ...}`` — exactly the
    shape :mod:`persona_sim` emits. ``"speaker": "other"`` is also
    accepted (e.g. transcribed real-world conversations from Step 03).
* ``user_goal`` — what the USER was trying to achieve. Falls back to
  :data:`DEFAULT_USER_GOAL`.
* ``person_profile`` — full PersonVault dict from Step 06 (or its
  inner ``profile`` block). Grounds ``missed_openings`` and
  ``ground_lost`` against the counterparty's known patterns.
* ``talk_dna_profile`` — TalkDNA dict from Step 05. Surfaces the
  user-side habits the model should specifically check for
  (over-apologising, hedging before vulnerable statements, silence
  under pressure, known escalation triggers).

The two profile blocks render to ``(no … profile provided)`` sentinels
when missing so the prompt template is never malformed on cold-start.
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
from .persona_sim import DEFAULT_USER_GOAL, format_persona_profile_block
from .premortem import format_talk_dna_block
from .prompts import load_prompt

_OTHER_SPEAKER_FALLBACK = "Other"

__all__ = [
    "DEFAULT_DEBRIEF_PROMPT_NAME",
    "DebriefConfig",
    "DebriefError",
    "RETRY_SAMPLING",
    "build_debrief_messages",
    "count_user_turns",
    "format_practice_transcript",
    "generate_debrief",
]

LOG = logging.getLogger(__name__)

DEFAULT_DEBRIEF_PROMPT_NAME = "debrief"

# Light sampling for the retry path — same temperature / top_p / top_k
# as the other intelligence runtimes. Pinned by the unit test so drift
# is loud.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}

# Caps on free-text fields. The schema does not constrain length;
# these guard against the model running into multi-paragraph narration
# instead of the requested coaching-line shape.
_MAX_LIST_ITEMS = 8
_MAX_QUOTE_CHARS = 400
_MAX_REASON_CHARS = 400
_MAX_DESCRIPTION_CHARS = 400
_MAX_BETTER_LINE_CHARS = 400
_MAX_ONE_FIX_CHARS = 400


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class DebriefError(ValueError):
    """The model emitted a malformed debrief payload — missing required
    field, empty string in a place that needs content, etc."""


@dataclass
class DebriefConfig:
    """Run-time inputs for :func:`generate_debrief`.

    ``transcript`` is the only required field. ``user_goal``,
    ``person_profile``, and ``talk_dna_profile`` all have sensible
    empty-sentinel rendering so the debrief works on cold-start
    (transcribed real-world conversation, no prior profiles yet) and
    on fully-loaded mode (Step 6 + Step 5 outputs threaded in).

    ``enable_thinking`` exposes Gemma 4's thinking channel. Defaults to
    ``False`` per the rule in ``knowledge/phases/rules.md``; the open
    hypothesis is that thinking helps on analytical payloads that
    write content for multiple speaker roles in one object — which is
    exactly what the debrief does (user quotes, opponent context,
    better-line in the user's voice). The notebook A/B tests this.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from the four
    context blocks. Pass an explicit ``instruction`` only for ablation
    experiments.
    """

    transcript: list[dict] = field(default_factory=list)
    user_goal: str = DEFAULT_USER_GOAL
    person_profile: dict = field(default_factory=dict)
    talk_dna_profile: dict = field(default_factory=dict)
    max_new_tokens: int = 768
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_DEBRIEF_PROMPT_NAME
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------


def count_user_turns(transcript: Any) -> int:
    """Return the number of ``speaker == "user"`` entries in the transcript.

    Exposed as a public helper so the notebook and unit tests can pin
    the expected ``turn`` ceiling without re-implementing the count.
    """
    if not isinstance(transcript, list):
        return 0
    return sum(1 for t in transcript if isinstance(t, dict) and t.get("speaker") == "user")


def _persona_label(turn: dict) -> str:
    """Render a persona / other turn's bracket label.

    Uses ``persona_name`` when present (Step 07's emit shape), falls
    back to the generic ``OTHER`` for transcribed-conversation turns.
    Appends ``resistance_type`` and ``escalation_level`` when both are
    on the turn, so the model sees the persona's own self-rating
    alongside what it said — useful signal for spotting which
    counterparty moves the user should have countered.
    """
    speaker = turn.get("speaker")
    name = turn.get("persona_name")
    if speaker == "persona":
        base = str(name).strip() if isinstance(name, str) and name.strip() else _OTHER_SPEAKER_FALLBACK
    else:
        base = "OTHER"

    resistance = turn.get("resistance_type")
    raw_esc = turn.get("escalation_level")
    if isinstance(resistance, str) and resistance.strip() and isinstance(raw_esc, (int, float)):
        return f"[{base} (resistance={resistance.strip()}, escalation={float(raw_esc):.2f})]"
    if isinstance(resistance, str) and resistance.strip():
        return f"[{base} (resistance={resistance.strip()})]"
    return f"[{base}]"


def _turn_text(turn: dict) -> str:
    """Return the spoken text for a turn, handling both shapes.

    User turns carry ``text``; persona turns (Step 07's emit shape)
    carry ``reply``. Falls back to whichever is present.
    """
    for key in ("text", "reply"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("\n", " ").strip()
    return ""


def format_practice_transcript(transcript: list[dict]) -> str:
    """Render a finished practice transcript as a numbered text block.

    User turns are labelled ``[USER 1]``, ``[USER 2]``, … so the model
    has unambiguous indices to use in the debrief's ``turn`` field.
    Persona / other turns are labelled with the persona's name (or
    ``OTHER``) and the persona-sim metadata (``resistance_type`` and
    ``escalation_level``) when present, but carry NO numeric index —
    the prompt explicitly tells the model not to cite them by ``turn``.

    Accepts both Step 07's emit shape (``speaker == "persona"`` with
    ``persona_name`` / ``reply`` / ``resistance_type`` /
    ``escalation_level``) and the simpler ``speaker == "user"|"other"``
    / ``text`` shape used by transcribed real-world conversations from
    Step 03.

    Returns a deterministic ``(empty transcript)`` sentinel when the
    input is empty so the prompt template is never malformed.
    """
    if not isinstance(transcript, list) or not transcript:
        return "(empty transcript)"
    lines: list[str] = []
    user_idx = 0
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        speaker = entry.get("speaker")
        text = _turn_text(entry)
        if not text:
            continue
        if speaker == "user":
            user_idx += 1
            lines.append(f"  [USER {user_idx}] {text}")
        else:
            lines.append(f"  {_persona_label(entry)} {text}")
    return "\n".join(lines) if lines else "(empty transcript)"


# ---------------------------------------------------------------------------
# Chat-template messages
# ---------------------------------------------------------------------------


def build_debrief_messages(instruction: str) -> list[dict]:
    """Build the OpenAI-style messages list for one debrief call.

    Single user-side instruction at index 0 — same shape as Steps 04 /
    05 / 06 / 08. No system role for prompt-based JSON; the
    instruction goes in the user turn so the chat template lays it
    out as a plain request the model answers directly.

    Pure function — kept separate from :func:`generate_debrief` so the
    message shape can be unit-tested without loading transformers.
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


def _coerce_turn(value: Any, *, user_turn_count: int) -> Optional[int]:
    """Coerce a model-emitted ``turn`` to an integer in ``[1, user_turn_count]``.

    Returns ``None`` when the value is unrecoverable (non-numeric,
    NaN, etc.) so the caller can drop the entry rather than emit a
    bogus turn index. When ``user_turn_count`` is 0 (debrief on an
    empty transcript is rejected upstream), always returns ``None``.
    """
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


def _coerce_ground_lost(raw: Any, *, user_turn_count: int) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        turn = _coerce_turn(entry.get("turn"), user_turn_count=user_turn_count)
        quote = _opt_str(entry.get("quote"))
        reason = _opt_str(entry.get("reason"))
        if turn is None or not quote or not reason:
            continue
        out.append(
            {
                "turn": turn,
                "quote": _cap(quote, limit=_MAX_QUOTE_CHARS),
                "reason": _cap(reason, limit=_MAX_REASON_CHARS),
            }
        )
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _coerce_over_apologies(raw: Any, *, user_turn_count: int) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        turn = _coerce_turn(entry.get("turn"), user_turn_count=user_turn_count)
        quote = _opt_str(entry.get("quote"))
        if turn is None or not quote:
            continue
        out.append(
            {
                "turn": turn,
                "quote": _cap(quote, limit=_MAX_QUOTE_CHARS),
            }
        )
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _coerce_missed_openings(raw: Any, *, user_turn_count: int) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        turn = _coerce_turn(entry.get("turn"), user_turn_count=user_turn_count)
        description = _opt_str(entry.get("description"))
        better_line = _opt_str(entry.get("better_line"))
        if turn is None or not description or not better_line:
            continue
        out.append(
            {
                "turn": turn,
                "description": _cap(description, limit=_MAX_DESCRIPTION_CHARS),
                "better_line": _cap(better_line, limit=_MAX_BETTER_LINE_CHARS),
            }
        )
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _coerce_wins(raw: Any, *, user_turn_count: int) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        turn = _coerce_turn(entry.get("turn"), user_turn_count=user_turn_count)
        description = _opt_str(entry.get("description"))
        if turn is None or not description:
            continue
        out.append(
            {
                "turn": turn,
                "description": _cap(description, limit=_MAX_DESCRIPTION_CHARS),
            }
        )
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _coerce_debrief_payload(payload: Any, *, user_turn_count: int) -> dict:
    """Turn the model's JSON output into a schema-conforming Debrief dict.

    System-contract enforcement (in code, not prompt — same rationale
    as Step 04's ``whisper_prompt=None for other`` rule and Step 08's
    ``scenario_id`` forcing):

    * Top-level payload must be a dict — anything else raises so the
      retry path takes over.
    * Each of the four arrays is coerced entry-by-entry; malformed
      entries are dropped silently rather than failing the whole
      debrief (a single bad ``ground_lost`` entry should not lose three
      valid ``wins``).
    * Each ``turn`` is clamped to ``[1, user_turn_count]``.
    * Required string fields per entry are rejected when empty.
    * Each list is capped at :data:`_MAX_LIST_ITEMS` items.
    * ``one_fix_next_time`` required and non-empty; raises when missing
      because it's the single most-important coaching takeaway and
      letting it slide produces a useless debrief.
    """
    if not isinstance(payload, dict):
        raise DebriefError(
            f"debrief payload not an object: {type(payload).__name__}"
        )

    one_fix = _opt_str(payload.get("one_fix_next_time"))
    if not one_fix:
        raise DebriefError("one_fix_next_time is empty or missing")
    one_fix = _cap(one_fix, limit=_MAX_ONE_FIX_CHARS)

    return {
        "ground_lost": _coerce_ground_lost(
            payload.get("ground_lost"), user_turn_count=user_turn_count
        ),
        "over_apologies": _coerce_over_apologies(
            payload.get("over_apologies"), user_turn_count=user_turn_count
        ),
        "missed_openings": _coerce_missed_openings(
            payload.get("missed_openings"), user_turn_count=user_turn_count
        ),
        "wins": _coerce_wins(
            payload.get("wins"), user_turn_count=user_turn_count
        ),
        "one_fix_next_time": one_fix,
        "debrief_id": f"debrief_{uuid.uuid4().hex[:12]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(cfg: DebriefConfig) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    user_turn_count = count_user_turns(cfg.transcript)
    if user_turn_count <= 0:
        raise DebriefError(
            "DebriefConfig.transcript has no user turns — debrief needs at "
            "least one USER turn to score"
        )
    user_goal = (cfg.user_goal or DEFAULT_USER_GOAL).strip() or DEFAULT_USER_GOAL
    return load_prompt(cfg.prompt_name).render(
        user_goal=user_goal,
        person_profile_block=format_persona_profile_block(cfg.person_profile),
        talk_dna_block=format_talk_dna_block(cfg.talk_dna_profile),
        transcript=format_practice_transcript(cfg.transcript),
    )


def _build_generation_config(
    cfg: DebriefConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def generate_debrief(
    processor: Any,
    model: Any,
    *,
    cfg: Optional[DebriefConfig] = None,
) -> dict:
    """Produce a coaching debrief for a finished practice transcript.

    Inference path mirrors the other intelligence components: greedy
    first (deterministic, fastest), then a single retry with light
    sampling (:data:`RETRY_SAMPLING`) on parse / validation failure.
    If both attempts fail, raises :class:`DebriefError` with an
    ``attempts`` attribute carrying each attempt's raw output (truncated
    to 500 chars) for diagnostics.

    Returns a dict matching ``data/schemas/debrief.schema.json`` plus
    runtime-attached ``debrief_id`` and ``generated_at`` fields (the
    schema permits additional properties).
    """
    cfg = cfg or DebriefConfig()
    user_turn_count = count_user_turns(cfg.transcript)
    if user_turn_count <= 0:
        raise DebriefError(
            "DebriefConfig.transcript has no user turns — debrief needs at "
            "least one USER turn to score"
        )
    instruction = _resolve_instruction(cfg)
    messages = build_debrief_messages(instruction)

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
            return _coerce_debrief_payload(
                payload, user_turn_count=user_turn_count
            )
        except (JsonParseError, DebriefError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "generate_debrief attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = DebriefError(
        f"debrief generation failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err
