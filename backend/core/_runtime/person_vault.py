"""Person Vault construction — Phase 3 / Step 6 of Tough Talks.

Given a conversation transcript and a counterparty's display name, this
module produces a PersonVault profile matching
``data/schemas/person_vault.schema.json``. The PersonVault is the
*mirror* of TalkDNA: TalkDNA captures how the **user** communicates,
PersonVault captures how a **recurring counterparty** communicates so
later steps (persona simulation, premortem, pulse) can be specific
about who they're modelling.

Hybrid design — same pattern as :mod:`talk_dna`:

* **Deterministic** numerics (``avg_other_turn_words``,
  ``deflection_candidates``, ``interruption_of_user_rate`` when timing
  is available) are computed in code from the transcript so the model
  never has to count.
* **LLM-judged** qualitative fields (``communication_style``,
  ``emotional_triggers``, ``de_escalation_keys``, curated
  ``common_deflections``, ``responds_best_to``, optional
  ``cultural_context``) come from a single prompt-based call to
  Gemma 4 E2B (text-only). Same two-shot retry pattern as
  :func:`analyze_talk_dna`: greedy first, light sampling once on
  parse / validation failure.

Incremental updates: pass an existing PersonVault profile via
``PersonVaultConfig.prior_profile``. The runtime bumps ``version``,
increments ``conversation_count``, inherits ``person_id``, and
**accumulates** the qualitative lists across conversations (deduped,
capped at 8 items). Accumulation is deliberate here — losing
``emotional_triggers`` observed in conversation 1 just because
conversation 2 didn't re-demonstrate them would defeat the point of a
rolling per-person profile.

Input transcript is the same shape as TalkDNA: a list of dicts with
``speaker`` (``"user"`` / ``"other"``), ``text``, and optional
``emotion`` / timing fields. The analysis focuses on the ``other``
speaker's turns; ``user`` turns are kept in the prompt for context
(triggers and de-escalation cues only make sense in dialogue).

``relationship_pulse`` (trend, unresolved items, last positive
exchange) is **not** computed here — it is the responsibility of
Step 11 and the field is omitted from this module's output. The schema
makes it optional so a profile emitted by Step 06 alone still
validates.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .audio import _extract_text
from .generation import GenerationConfig, chat
from .parsing import JsonParseError, parse_json
from .prompts import load_prompt
from .talk_dna import format_transcript

__all__ = [
    "ALLOWED_COMMUNICATION_STYLES",
    "ALLOWED_RELATIONSHIP_TYPES",
    "DEFAULT_PERSON_VAULT_PROMPT_NAME",
    "DEFAULT_PERSON_NAME",
    "DEFAULT_RELATIONSHIP_TYPE",
    "DeterministicPersonMetrics",
    "PersonVaultAnalysisError",
    "PersonVaultConfig",
    "RETRY_SAMPLING",
    "analyze_person_vault",
    "compute_person_metrics",
    "format_person_prior_profile",
]

LOG = logging.getLogger(__name__)

DEFAULT_PERSON_VAULT_PROMPT_NAME = "person_vault"
DEFAULT_PERSON_NAME = "Other"
DEFAULT_RELATIONSHIP_TYPE = "other"

# Mirrors the enums at
# data/schemas/person_vault.schema.json#/properties/relationship_type/enum
# and #/properties/profile/properties/communication_style/enum.
# Unit tests pin these against the on-disk schema so drift fails loudly.
ALLOWED_RELATIONSHIP_TYPES: tuple[str, ...] = (
    "partner",
    "parent",
    "sibling",
    "friend",
    "manager",
    "report",
    "colleague",
    "other",
)

ALLOWED_COMMUNICATION_STYLES: tuple[str, ...] = (
    "direct",
    "indirect",
    "passive_aggressive",
    "avoidant",
    "collaborative",
    "dominant",
    "assertive",
    "empathetic",
    "defensive",
)

# Near-synonym map for ``communication_style``. The LLM tends to reach
# for words that match its training-time distribution rather than our
# enum (e.g. it emits "defensive" / "aggressive" / "evasive" instead of
# the canonical enum codes). Rather than fight the model with prompt
# rules, we accept the obvious synonyms here and map them to the
# closest canonical value. Truly unknown styles still raise.
#
# Keys are space-separated lower-case (so "passive-aggressive" and
# "Passive Aggressive" both normalise to the same key before lookup);
# values must be in :data:`ALLOWED_COMMUNICATION_STYLES`.
_COMMUNICATION_STYLE_SYNONYMS: dict[str, str] = {
    # Confrontational / aggressive cluster
    "aggressive": "dominant",
    "confrontational": "dominant",
    "combative": "dominant",
    "argumentative": "dominant",
    "controlling": "dominant",
    # Avoidant / evasive cluster
    "evasive": "avoidant",
    "withdrawn": "avoidant",
    "stonewalling": "avoidant",
    "deflective": "avoidant",
    # Direct cluster
    "blunt": "direct",
    "frank": "direct",
    "straightforward": "direct",
    # Assertive cluster
    "firm": "assertive",
    # Empathetic / warm cluster
    "warm": "empathetic",
    "supportive": "empathetic",
    "compassionate": "empathetic",
    # Collaborative cluster
    "open": "collaborative",
    "cooperative": "collaborative",
    # Indirect / diplomatic cluster
    "diplomatic": "indirect",
    "tactful": "indirect",
    # Defensive cluster — guarded / protective postures
    "guarded": "defensive",
    "protective": "defensive",
}


def _normalize_communication_style(value: Any) -> Optional[str]:
    """Coerce a model-emitted ``communication_style`` to a canonical
    enum value.

    Handles three sources of drift between model output and the schema
    enum:

    * **Casing** — ``"Direct"`` / ``"DIRECT"`` → ``"direct"``.
    * **Word separator** — ``"passive-aggressive"`` /
      ``"passive aggressive"`` → ``"passive_aggressive"``.
    * **Near-synonyms** — ``"aggressive"`` → ``"dominant"``,
      ``"evasive"`` → ``"avoidant"``, etc. via
      :data:`_COMMUNICATION_STYLE_SYNONYMS`.

    Returns the canonical enum value, or ``None`` if the input is
    unrecognisable. Callers raise a :class:`PersonVaultAnalysisError`
    on ``None`` so the retry path can take over.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace("-", " ")
    if not cleaned:
        return None
    # First try the canonical snake_case form (covers exact enum match
    # plus spelling/spacing variants of the canonical values).
    snake = cleaned.replace(" ", "_")
    if snake in ALLOWED_COMMUNICATION_STYLES:
        return snake
    # Fall back to the synonym map (keyed on the space-separated form).
    return _COMMUNICATION_STYLE_SYNONYMS.get(cleaned)

# Light sampling for the retry path — same rationale as the TalkDNA and
# emotion-radar retries: just enough randomness to escape a single bad
# greedy trajectory without losing too much determinism. Pinned by the
# unit test.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}


# ---------------------------------------------------------------------------
# Deterministic deflection candidates
# ---------------------------------------------------------------------------

# Common English deflection cues used by counterparties to avoid the
# issue, push blame back, or invoke history rather than addressing the
# current point. Phrases match whole (with word boundaries) and are
# case-insensitive. The list stays small on purpose — PersonVault only
# needs CANDIDATES; the LLM curates the final ``common_deflections``
# list and can add culture/relationship-specific deflections from the
# transcript.
#
# Each entry is (label, regex). The label is what's surfaced to the
# prompt — short, generalised — and the regex is the matcher.
_DEFLECTION_CANDIDATES: tuple[tuple[str, re.Pattern], ...] = (
    ("you always", re.compile(r"\byou\s+always\b", re.IGNORECASE)),
    ("you never", re.compile(r"\byou\s+never\b", re.IGNORECASE)),
    ("don't blame me", re.compile(r"\bdon'?t\s+blame\s+me\b", re.IGNORECASE)),
    (
        "don't put this on me",
        re.compile(r"\bdon'?t\s+put\s+this\s+on\s+me\b", re.IGNORECASE),
    ),
    ("not my fault", re.compile(r"\b(?:not|isn'?t|wasn'?t)\s+my\s+fault\b", re.IGNORECASE)),
    ("you weren't there", re.compile(r"\byou\s+weren'?t\s+(?:there|in)\b", re.IGNORECASE)),
    ("you said you would", re.compile(r"\byou\s+said\s+you\s+would\b", re.IGNORECASE)),
    ("i told you", re.compile(r"\bi\s+(?:already\s+)?told\s+you\b", re.IGNORECASE)),
    (
        "as i already said",
        re.compile(r"\bas\s+i\s+(?:already\s+)?(?:said|told\s+you)\b", re.IGNORECASE),
    ),
    (
        "that's not the point",
        re.compile(r"\bthat'?s\s+not\s+(?:the|my)\s+point\b", re.IGNORECASE),
    ),
    (
        "you're missing the point",
        re.compile(r"\byou(?:'re|\s+are)\s+missing\s+(?:the|my)\s+point\b", re.IGNORECASE),
    ),
    ("yes but", re.compile(r"\byes,?\s+but\b", re.IGNORECASE)),
)

_DEFLECTION_MIN_OCCURRENCES = 1  # one is enough — these are strong cues
_MAX_LIST_ITEMS = 8  # safety cap for any returned string list
_MAX_RESPONDS_BEST_TO_CHARS = 280  # one or two sentences
_MAX_CULTURAL_CONTEXT_CHARS = 280


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class PersonVaultAnalysisError(ValueError):
    """The model emitted a malformed PersonVault payload (bad enum
    value, non-dict response, etc.) — or the input transcript is
    unusable for counterparty profiling."""


@dataclass(frozen=True)
class DeterministicPersonMetrics:
    """Numerics computed in code from the transcript.

    Fed into the prompt as ground truth and merged into the final
    profile. ``interruption_of_user_rate`` is ``None`` when no per-turn
    timing data is available — we never fabricate a value.
    """

    other_turn_count: int
    total_turn_count: int
    avg_other_turn_words: float
    deflection_candidates: list[str]
    interruption_of_user_rate: Optional[float]


@dataclass
class PersonVaultConfig:
    """Run-time inputs for :func:`analyze_person_vault`.

    ``name`` is the user-facing display name (e.g. ``"Mom"``,
    ``"Manager Sarah"``) — required input from the caller, never
    inferred from the transcript.

    ``relationship_type`` is one of
    :data:`ALLOWED_RELATIONSHIP_TYPES` or ``None``. When ``None`` and
    a ``prior_profile`` is provided, the prior's value is inherited;
    otherwise it defaults to ``"other"``.

    ``person_id`` is an opaque locally-generated UUID. Pass an existing
    one to keep continuity across calls; leave ``None`` to let the
    runtime mint a fresh id (or inherit one from ``prior_profile``).

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from ``name``,
    ``relationship_type``, the transcript, the deterministic metrics,
    and ``prior_profile``. Pass an explicit ``instruction`` only for
    ablation experiments.

    ``prior_profile`` is an existing PersonVault dict (matching
    ``data/schemas/person_vault.schema.json``). When provided the
    runtime bumps ``version``, increments ``conversation_count``,
    inherits ``person_id``, and **accumulates** the qualitative lists
    across conversations (deduped, capped at 8 items).
    """

    name: str = DEFAULT_PERSON_NAME
    relationship_type: Optional[str] = None
    person_id: Optional[str] = None
    max_new_tokens: int = 768
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_PERSON_VAULT_PROMPT_NAME
    prior_profile: Optional[dict] = None


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


def _avg_other_turn_words(other_turns: list[dict]) -> float:
    if not other_turns:
        return 0.0
    return sum(_word_count(t.get("text", "") or "") for t in other_turns) / len(
        other_turns
    )


def _extract_deflection_candidates(other_turns: list[dict]) -> list[str]:
    if not other_turns:
        return []
    counts: dict[str, int] = {}
    for t in other_turns:
        text = t.get("text", "") or ""
        for label, pattern in _DEFLECTION_CANDIDATES:
            n = len(pattern.findall(text))
            if n:
                counts[label] = counts.get(label, 0) + n
    surfaced = {
        label: n
        for label, n in counts.items()
        if n >= _DEFLECTION_MIN_OCCURRENCES
    }
    return sorted(surfaced, key=lambda c: -surfaced[c])


def _interruption_of_user_rate(turns: list[dict]) -> Optional[float]:
    """Fraction of 'other' turns that begin before the prior 'user'
    turn's end. Mirror of the TalkDNA interruption metric but for the
    counterparty interrupting the user.

    Requires per-turn ``start_seconds`` and ``end_seconds`` on at least
    two adjacent turns. Returns ``None`` when timing data isn't
    present — never fakes a value.
    """
    timed = [
        t for t in turns if "start_seconds" in t and "end_seconds" in t
    ]
    if len(timed) < 2:
        return None
    candidate_count = 0
    interrupted = 0
    for i in range(1, len(timed)):
        cur, prev = timed[i], timed[i - 1]
        if cur.get("speaker") != "other" or prev.get("speaker") != "user":
            continue
        candidate_count += 1
        try:
            if float(cur["start_seconds"]) < float(prev["end_seconds"]):
                interrupted += 1
        except (TypeError, ValueError):
            continue
    if candidate_count == 0:
        return None
    return interrupted / candidate_count


def compute_person_metrics(turns: list[dict]) -> DeterministicPersonMetrics:
    """Compute the deterministic numerics for the counterparty side of
    one conversation.

    Pure function over the input list — no model calls, no I/O. Kept
    public so the notebook can render the code-side numbers without
    invoking the LLM.
    """
    other_turns = [t for t in turns if t.get("speaker") == "other"]
    return DeterministicPersonMetrics(
        other_turn_count=len(other_turns),
        total_turn_count=len(turns),
        avg_other_turn_words=_avg_other_turn_words(other_turns),
        deflection_candidates=_extract_deflection_candidates(other_turns),
        interruption_of_user_rate=_interruption_of_user_rate(turns),
    )


# ---------------------------------------------------------------------------
# Prompt rendering helpers
# ---------------------------------------------------------------------------

_NO_PRIOR_SENTINEL = (
    "(none — this is the first conversation in this person's vault)"
)
_RELATIONSHIP_UNSPECIFIED = "unspecified"


def _format_metrics_block(metrics: DeterministicPersonMetrics) -> str:
    deflections = metrics.deflection_candidates or "none detected"
    lines = [
        f"- other_turn_count: {metrics.other_turn_count} "
        f"(of {metrics.total_turn_count} total turns)",
        f"- avg_other_turn_words: {metrics.avg_other_turn_words:.1f}",
        f"- deflection_phrase_candidates: {deflections}",
    ]
    if metrics.interruption_of_user_rate is None:
        lines.append(
            "- interruption_of_user_rate: unknown (no per-turn timing data)"
        )
    else:
        lines.append(
            f"- interruption_of_user_rate: "
            f"{metrics.interruption_of_user_rate:.2f}"
        )
    return "\n".join(lines)


def format_person_prior_profile(prior: Optional[dict]) -> str:
    """Render an existing PersonVault profile as a text block for the
    prompt.

    Used for incremental updates: the model sees what we've already
    observed about this person and is asked to refine rather than
    restart. Returns a deterministic "no prior" sentinel when ``prior``
    is empty so the prompt template is never malformed.
    """
    if not prior or not isinstance(prior, dict):
        return _NO_PRIOR_SENTINEL
    profile = prior.get("profile") or {}
    lines = [
        f"  version: {prior.get('version', 1)}",
        f"  conversation_count: {prior.get('conversation_count', 0)}",
        f"  profile.communication_style: "
        f"{profile.get('communication_style', 'unknown')}",
        f"  profile.emotional_triggers: "
        f"{profile.get('emotional_triggers', [])}",
        f"  profile.de_escalation_keys: "
        f"{profile.get('de_escalation_keys', [])}",
        f"  profile.common_deflections: "
        f"{profile.get('common_deflections', [])}",
        f"  profile.responds_best_to: "
        f"{profile.get('responds_best_to', 'unknown')}",
    ]
    if profile.get("cultural_context"):
        lines.append(
            f"  profile.cultural_context: {profile['cultural_context']}"
        )
    return "\n".join(lines)


def _resolve_relationship_type(
    cfg: PersonVaultConfig, prior: Optional[dict]
) -> str:
    """Pick the final relationship_type for the emitted profile.

    Order of preference: explicit cfg value (validated against enum) →
    prior profile's value (if valid) → default ``"other"``.
    """
    if cfg.relationship_type:
        if cfg.relationship_type in ALLOWED_RELATIONSHIP_TYPES:
            return cfg.relationship_type
        raise PersonVaultAnalysisError(
            f"relationship_type {cfg.relationship_type!r} not in "
            f"{ALLOWED_RELATIONSHIP_TYPES}"
        )
    if prior:
        prior_rt = prior.get("relationship_type")
        if isinstance(prior_rt, str) and prior_rt in ALLOWED_RELATIONSHIP_TYPES:
            return prior_rt
    return DEFAULT_RELATIONSHIP_TYPE


def _resolve_person_id(
    cfg: PersonVaultConfig, prior: Optional[dict]
) -> str:
    """Pick the final person_id. Explicit > inherited > freshly minted."""
    if cfg.person_id:
        return cfg.person_id
    if prior:
        prior_id = prior.get("person_id")
        if isinstance(prior_id, str) and prior_id:
            return prior_id
    return f"person_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Payload coercion
# ---------------------------------------------------------------------------


def _trim_unique_strings(items: Any, *, limit: int) -> list[str]:
    """Return a deduped list of clean, non-empty string items, capped.

    Comparison key is the lower-cased trimmed form so ``"You Always"``
    and ``"you always"`` collapse to a single entry. Preserves the
    first-seen casing.
    """
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _merge_string_lists(
    prior_items: Any, new_items: list[str], *, limit: int
) -> list[str]:
    """Accumulate prior + new observations into a single deduped list.

    Prior items come first (preserving the order in which we observed
    them); new items are appended. Comparison is case-insensitive on the
    trimmed form. Capped at ``limit`` total entries — when the cap is
    reached, the OLDEST priors are dropped first so the list stays
    responsive to recent conversations.
    """
    cleaned_prior = _trim_unique_strings(prior_items, limit=limit)
    cleaned_new = _trim_unique_strings(new_items, limit=limit)
    combined: list[str] = []
    seen: set[str] = set()
    for item in cleaned_prior + cleaned_new:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(item)
    if len(combined) > limit:
        # Drop oldest priors first — keep the tail.
        combined = combined[-limit:]
    return combined


def _clean_short_text(value: Any, *, limit: int) -> Optional[str]:
    """Trim/cap a free-text field. Returns ``None`` for empty / non-str."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _coerce_person_vault_payload(
    payload: Any,
    *,
    cfg: PersonVaultConfig,
    metrics: DeterministicPersonMetrics,
) -> dict:
    """Turn the model's JSON output into a schema-conforming PersonVault
    dict.

    Validates enums, dedupes / accumulates string lists, merges prior
    observations, and attaches ``person_id``, ``name``,
    ``relationship_type``, ``version``, ``conversation_count``, and
    ``updated_at``. Raises :class:`PersonVaultAnalysisError` for any
    missing/invalid field that can't be repaired without guessing.
    """
    if not isinstance(payload, dict):
        raise PersonVaultAnalysisError(
            f"person_vault payload not an object: {type(payload).__name__}"
        )

    raw_style = payload.get("communication_style")
    style = _normalize_communication_style(raw_style)
    if style is None:
        raise PersonVaultAnalysisError(
            f"communication_style not in enum: {raw_style!r}; "
            f"allowed: {ALLOWED_COMMUNICATION_STYLES}"
        )

    prior = cfg.prior_profile if isinstance(cfg.prior_profile, dict) else {}
    prior_profile_obj = prior.get("profile") or {}
    prior_count = int(prior.get("conversation_count", 0) or 0)
    prior_version = int(prior.get("version", 0) or 0)

    # Curated deflections — merge model output with the deterministic
    # candidates so we don't lose a strong cue the model happened to
    # omit on this pass.
    new_deflections = _trim_unique_strings(
        payload.get("common_deflections"), limit=_MAX_LIST_ITEMS
    )
    if not new_deflections:
        new_deflections = list(metrics.deflection_candidates[:_MAX_LIST_ITEMS])

    emotional_triggers = _merge_string_lists(
        prior_profile_obj.get("emotional_triggers"),
        _trim_unique_strings(
            payload.get("emotional_triggers"), limit=_MAX_LIST_ITEMS
        ),
        limit=_MAX_LIST_ITEMS,
    )
    de_escalation_keys = _merge_string_lists(
        prior_profile_obj.get("de_escalation_keys"),
        _trim_unique_strings(
            payload.get("de_escalation_keys"), limit=_MAX_LIST_ITEMS
        ),
        limit=_MAX_LIST_ITEMS,
    )
    common_deflections = _merge_string_lists(
        prior_profile_obj.get("common_deflections"),
        new_deflections,
        limit=_MAX_LIST_ITEMS,
    )

    responds_best_to = _clean_short_text(
        payload.get("responds_best_to"), limit=_MAX_RESPONDS_BEST_TO_CHARS
    )
    if responds_best_to is None:
        # Fall back to the prior value rather than emitting an empty
        # field — a counterparty's preferred framing shouldn't be lost
        # because this round didn't surface fresh signal.
        responds_best_to = _clean_short_text(
            prior_profile_obj.get("responds_best_to"),
            limit=_MAX_RESPONDS_BEST_TO_CHARS,
        )

    cultural_context = _clean_short_text(
        payload.get("cultural_context"), limit=_MAX_CULTURAL_CONTEXT_CHARS
    )
    if cultural_context is None:
        cultural_context = _clean_short_text(
            prior_profile_obj.get("cultural_context"),
            limit=_MAX_CULTURAL_CONTEXT_CHARS,
        )

    profile: dict[str, Any] = {
        "communication_style": style,
        "emotional_triggers": emotional_triggers,
        "de_escalation_keys": de_escalation_keys,
        "common_deflections": common_deflections,
    }
    if responds_best_to:
        profile["responds_best_to"] = responds_best_to
    if cultural_context:
        profile["cultural_context"] = cultural_context

    relationship_type = _resolve_relationship_type(cfg, prior)
    person_id = _resolve_person_id(cfg, prior)
    name = cfg.name.strip() or prior.get("name") or DEFAULT_PERSON_NAME

    result: dict[str, Any] = {
        "person_id": person_id,
        "name": name,
        "relationship_type": relationship_type,
        "version": prior_version + 1,
        "conversation_count": prior_count + 1,
        "profile": profile,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Preserve any relationship_pulse the caller had attached to the
    # prior — Step 06 doesn't compute it, but we shouldn't drop it
    # silently on an incremental update.
    if isinstance(prior.get("relationship_pulse"), dict):
        result["relationship_pulse"] = prior["relationship_pulse"]
    return result


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(
    cfg: PersonVaultConfig,
    *,
    turns: list[dict],
    metrics: DeterministicPersonMetrics,
) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    rel_type = cfg.relationship_type or (
        cfg.prior_profile.get("relationship_type")
        if isinstance(cfg.prior_profile, dict)
        else None
    )
    rel_block = rel_type if rel_type else _RELATIONSHIP_UNSPECIFIED
    return load_prompt(cfg.prompt_name).render(
        person_name=cfg.name,
        relationship_type_block=rel_block,
        prior_profile_block=format_person_prior_profile(cfg.prior_profile),
        metrics_block=_format_metrics_block(metrics),
        transcript=format_transcript(turns),
    )


def _build_generation_config(
    cfg: PersonVaultConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def analyze_person_vault(
    processor: Any,
    model: Any,
    turns: list[dict],
    *,
    cfg: Optional[PersonVaultConfig] = None,
) -> dict:
    """Compute or update a PersonVault profile from one conversation.

    ``turns`` is the standard transcript shape: list of dicts with at
    least ``speaker`` (``"user"`` / ``"other"``) and ``text``. Optional
    ``emotion`` and ``start_seconds`` / ``end_seconds`` fields are used
    for prompt context and the interruption metric respectively.

    Inference path is two-shot, mirroring :func:`analyze_talk_dna`:
    greedy first (deterministic, fastest), then a single retry with
    light sampling (``RETRY_SAMPLING``) on parse / validation failure.
    If both attempts fail, raises :class:`PersonVaultAnalysisError`
    with an ``attempts`` attribute carrying each attempt's raw output
    (truncated to 500 chars) for diagnostics.

    Returns a dict matching ``data/schemas/person_vault.schema.json``.
    """
    cfg = cfg or PersonVaultConfig()

    if not turns:
        raise PersonVaultAnalysisError("turns is empty — nothing to analyze")
    other_turns = [t for t in turns if t.get("speaker") == "other"]
    if not other_turns:
        raise PersonVaultAnalysisError(
            "no 'other' turns in transcript — PersonVault needs "
            "counterparty dialogue"
        )

    metrics = compute_person_metrics(turns)
    instruction = _resolve_instruction(cfg, turns=turns, metrics=metrics)
    messages: list[dict] = [{"role": "user", "content": instruction}]

    attempts: list[dict] = []
    for sampling in (None, RETRY_SAMPLING):
        gen_cfg = _build_generation_config(cfg, sampling)
        raw = chat(processor, model, messages, cfg=gen_cfg)
        text = _extract_text(processor.parse_response(raw)).strip()
        try:
            payload = parse_json(text)
            return _coerce_person_vault_payload(
                payload, cfg=cfg, metrics=metrics
            )
        except (JsonParseError, PersonVaultAnalysisError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "analyze_person_vault attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = PersonVaultAnalysisError(
        f"person_vault analysis failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err
