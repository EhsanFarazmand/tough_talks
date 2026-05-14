"""Talk DNA pattern detector — Phase 3 / Step 5 of Tough Talks.

Given a single conversation transcript (optionally enriched with the
per-turn emotion data emitted by Step 4) this module produces a TalkDNA
profile matching ``data/schemas/talk_dna.schema.json``.

Hybrid design:

* **Deterministic** numerics (``apology_rate``, ``avg_turn_length_words``,
  ``interruption_rate``, plus a candidate set of habitual filler
  phrases) are computed in code from the transcript — the model never
  has to count. This keeps the precision-sensitive parts honest.
* **LLM-judged** qualitative fields (``sarcasm_frequency``,
  ``silence_under_pressure``, ``escalation_triggers``, curated
  ``filler_phrases``, ``strengths``, ``weaknesses``) come from a single
  prompt-based call to Gemma 4 E2B (text-only — no audio path here).
  Same two-shot retry pattern as :mod:`emotion`: greedy first, light
  sampling once on parse/validation failure.

Incremental updates: pass an existing TalkDNA profile via
``TalkDNAConfig.prior_profile``. The runtime bumps ``version`` and
``conversation_count``, weighted-averages the deterministic numerics
against the priors, and feeds the prior profile into the prompt so the
model evolves the qualitative traits instead of restarting from scratch.

The input transcript is a list of dicts with at least ``speaker``
(``"user"`` or ``"other"``) and ``text`` fields; an optional
``emotion`` field (the inner ``emotions`` object from an EmotionRadar
result) is rendered into the prompt when present, giving the model
prosody-derived signal it could not infer from text alone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .audio import _extract_text
from .generation import GenerationConfig, chat
from .parsing import JsonParseError, parse_json
from .prompts import load_prompt

__all__ = [
    "ALLOWED_SARCASM_FREQUENCIES",
    "ALLOWED_SPEAKERS",
    "DEFAULT_TALK_DNA_PROMPT_NAME",
    "DEFAULT_USER_ID",
    "DeterministicMetrics",
    "RETRY_SAMPLING",
    "TalkDNAAnalysisError",
    "TalkDNAConfig",
    "analyze_talk_dna",
    "compute_deterministic_metrics",
    "format_prior_profile",
    "format_transcript",
]

LOG = logging.getLogger(__name__)

DEFAULT_TALK_DNA_PROMPT_NAME = "talk_dna"
DEFAULT_USER_ID = "local"

ALLOWED_SPEAKERS: tuple[str, ...] = ("user", "other")

# Mirrors the schema enum at
# data/schemas/talk_dna.schema.json#/properties/patterns/properties/sarcasm_frequency/enum.
# The unit test pins this against the on-disk schema so drift fails loudly.
ALLOWED_SARCASM_FREQUENCIES: tuple[str, ...] = (
    "never",
    "rare",
    "low",
    "moderate",
    "high",
)

# Light sampling for the retry path — same rationale as the emotion-radar
# retry: just enough randomness to escape a single bad greedy trajectory
# without losing too much determinism. Pinned by the unit test.
RETRY_SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 0.9, "top_k": 64}


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

# Apology detection is intentionally conservative. We match the
# unmistakable cues only ("sorry", "apologise", "my bad", "my fault",
# "forgive me") and let the model pick up subtler forms ("I shouldn't
# have…") via the qualitative fields. False positives here would
# inflate ``apology_rate`` and miscalibrate every downstream judgement.
_APOLOGY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b(sorry|apolog(?:y|ies|ize|ise|izing|ising))\b", re.IGNORECASE),
    re.compile(r"\b(my\s+bad|my\s+fault|forgive\s+me)\b", re.IGNORECASE),
)

# Common English hedges. Multi-word entries are matched whole (word
# boundaries on either side). The list stays small on purpose — Talk
# DNA only needs to surface candidates; the LLM curates the final list
# and can add culture/language-specific hedges from the transcript.
_FILLER_CANDIDATES: tuple[str, ...] = (
    "just",
    "kind of",
    "sort of",
    "i think",
    "i guess",
    "i mean",
    "i feel like",
    "you know",
    "maybe",
    "perhaps",
    "actually",
    "literally",
    "basically",
)

_FILLER_MIN_OCCURRENCES = 2  # surfaced only when used habitually
_MAX_LIST_ITEMS = 8  # safety cap for any returned string list
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class DeterministicMetrics:
    """Numerics computed in code from the transcript — fed into the
    prompt as ground truth and merged into the final profile.

    ``interruption_rate`` is ``None`` when no per-turn timing data
    (``start_seconds`` / ``end_seconds``) is available on the input
    turns — we never fabricate a value, and the field is simply omitted
    from the emitted profile in that case.
    """

    apology_rate: float
    avg_user_turn_words: float
    filler_candidates: list[str]
    interruption_rate: Optional[float]
    user_turn_count: int
    total_turn_count: int


def _word_count(text: str) -> int:
    return len(text.split())


def _count_apologies(user_turns: list[dict]) -> int:
    return sum(
        1
        for t in user_turns
        if any(p.search(t.get("text", "") or "") for p in _APOLOGY_PATTERNS)
    )


def _avg_user_turn_words(user_turns: list[dict]) -> float:
    if not user_turns:
        return 0.0
    return sum(_word_count(t.get("text", "") or "") for t in user_turns) / len(
        user_turns
    )


def _extract_filler_candidates(user_turns: list[dict]) -> list[str]:
    if not user_turns:
        return []
    counts: dict[str, int] = {}
    for t in user_turns:
        text = (t.get("text", "") or "").lower()
        for candidate in _FILLER_CANDIDATES:
            pattern = re.compile(
                r"\b" + re.escape(candidate) + r"\b", re.IGNORECASE
            )
            n = len(pattern.findall(text))
            if n:
                counts[candidate] = counts.get(candidate, 0) + n
    habitual = {c: n for c, n in counts.items() if n >= _FILLER_MIN_OCCURRENCES}
    return sorted(habitual, key=lambda c: -habitual[c])


def _interruption_rate(turns: list[dict]) -> Optional[float]:
    """Fraction of user turns that begin before the prior 'other' turn's
    end. Requires per-turn ``start_seconds`` and ``end_seconds`` on at
    least two adjacent turns. Returns ``None`` when timing data isn't
    present — never fakes a value."""
    timed = [
        t for t in turns if "start_seconds" in t and "end_seconds" in t
    ]
    if len(timed) < 2:
        return None
    candidate_count = 0
    interrupted = 0
    for i in range(1, len(timed)):
        cur, prev = timed[i], timed[i - 1]
        if cur.get("speaker") != "user" or prev.get("speaker") != "other":
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


def compute_deterministic_metrics(turns: list[dict]) -> DeterministicMetrics:
    """Compute the deterministic numerics for a single conversation.

    Pure function over the input list — no model calls, no I/O. Kept
    public so the notebook (and any future caller that wants the
    transparent numbers without invoking the LLM) can render them.
    """
    user_turns = [t for t in turns if t.get("speaker") == "user"]
    apologies = _count_apologies(user_turns)
    return DeterministicMetrics(
        apology_rate=(apologies / len(user_turns)) if user_turns else 0.0,
        avg_user_turn_words=_avg_user_turn_words(user_turns),
        filler_candidates=_extract_filler_candidates(user_turns),
        interruption_rate=_interruption_rate(turns),
        user_turn_count=len(user_turns),
        total_turn_count=len(turns),
    )


# ---------------------------------------------------------------------------
# Prompt rendering helpers
# ---------------------------------------------------------------------------

_NO_PRIOR_SENTINEL = (
    "(none — this is the first conversation in the user's TalkDNA history)"
)


def format_transcript(turns: list[dict]) -> str:
    """Render a conversation as ``[idx] speaker (emotion): text`` lines.

    Emotion tags are appended when an ``emotion`` dict is present on the
    turn (subset of an EmotionRadar ``emotions`` object). Missing fields
    are tolerated — Talk DNA must work on text-only transcripts too.
    """
    if not turns:
        return "(empty transcript)"
    lines: list[str] = []
    for i, t in enumerate(turns, start=1):
        speaker = t.get("speaker", "?")
        emotion = t.get("emotion") or {}
        primary = emotion.get("primary")
        intensity = emotion.get("intensity")
        if primary and isinstance(intensity, (int, float)):
            tag = f" ({primary}, intensity {float(intensity):.2f})"
        elif primary:
            tag = f" ({primary})"
        else:
            tag = ""
        text = (t.get("text") or "").replace("\n", " ").strip()
        lines.append(f"  [{i}] {speaker}{tag}: {text}")
    return "\n".join(lines)


def _format_metrics_block(metrics: DeterministicMetrics) -> str:
    fillers = metrics.filler_candidates or "none detected"
    lines = [
        f"- apology_rate: {metrics.apology_rate:.2f} "
        f"(across {metrics.user_turn_count} user turns)",
        f"- avg_user_turn_words: {metrics.avg_user_turn_words:.1f}",
        f"- filler_phrase_candidates: {fillers}",
    ]
    if metrics.interruption_rate is None:
        lines.append("- interruption_rate: unknown (no per-turn timing data)")
    else:
        lines.append(f"- interruption_rate: {metrics.interruption_rate:.2f}")
    return "\n".join(lines)


def format_prior_profile(prior: Optional[dict]) -> str:
    """Render an existing TalkDNA profile as a text block for the prompt.

    Used for incremental updates: the model sees what we've already
    observed about the user and is asked to refine rather than restart.
    Returns a deterministic "no prior" sentinel when ``prior`` is empty
    so the prompt template is never malformed.
    """
    if not prior or not isinstance(prior, dict):
        return _NO_PRIOR_SENTINEL
    patterns = prior.get("patterns") or {}
    strengths = prior.get("strengths") or []
    weaknesses = prior.get("weaknesses") or []
    lines = [
        f"  version: {prior.get('version', 1)}",
        f"  conversation_count: {prior.get('conversation_count', 0)}",
        f"  patterns.apology_rate: {patterns.get('apology_rate', 'unknown')}",
        f"  patterns.avg_turn_length_words: "
        f"{patterns.get('avg_turn_length_words', 'unknown')}",
        f"  patterns.filler_phrases: {patterns.get('filler_phrases', [])}",
        f"  patterns.sarcasm_frequency: "
        f"{patterns.get('sarcasm_frequency', 'unknown')}",
        f"  patterns.silence_under_pressure: "
        f"{patterns.get('silence_under_pressure', 'unknown')}",
        f"  patterns.escalation_triggers: "
        f"{patterns.get('escalation_triggers', [])}",
        f"  strengths: {strengths}",
        f"  weaknesses: {weaknesses}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public dataclasses + errors
# ---------------------------------------------------------------------------


class TalkDNAAnalysisError(ValueError):
    """The model emitted a malformed TalkDNA payload (bad enum value,
    non-dict response, etc.) — or the input transcript is unusable."""


@dataclass
class TalkDNAConfig:
    """Run-time inputs for :func:`analyze_talk_dna`.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it from ``user_id``,
    the transcript, the deterministic metrics, and ``prior_profile``.
    Pass an explicit ``instruction`` only for ablation experiments.

    ``prior_profile`` is an existing TalkDNA dict (matching
    ``data/schemas/talk_dna.schema.json``). When provided, the runtime
    bumps ``version``, increments ``conversation_count``, and
    weighted-averages the deterministic numerics against the prior
    values using ``conversation_count`` as the weight.
    """

    user_id: str = DEFAULT_USER_ID
    max_new_tokens: int = 768
    instruction: Optional[str] = None
    prompt_name: str = DEFAULT_TALK_DNA_PROMPT_NAME
    prior_profile: Optional[dict] = None


# ---------------------------------------------------------------------------
# Payload coercion
# ---------------------------------------------------------------------------


def _trim_unique_strings(items: Any, *, limit: int) -> list[str]:
    """Return a deduped list of clean, non-empty string items, capped."""
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


def _normalize_identifier(text: str) -> str:
    """Coerce a model-emitted label to lowercase snake_case."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return cleaned


def _trim_identifier_list(items: Any, *, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        ident = _normalize_identifier(item)
        if not ident or not _IDENTIFIER_RE.match(ident):
            continue
        if ident in seen:
            continue
        seen.add(ident)
        out.append(ident)
        if len(out) >= limit:
            break
    return out


def _merge_average(
    prior_value: Optional[float],
    prior_n: int,
    new_value: float,
) -> float:
    """Weighted average of ``prior_value`` (observed over ``prior_n``
    conversations) and ``new_value`` (this conversation). Falls back to
    ``new_value`` when prior is missing or ``prior_n`` is zero."""
    if prior_value is None or prior_n <= 0:
        return float(new_value)
    try:
        prior_f = float(prior_value)
    except (TypeError, ValueError):
        return float(new_value)
    return (prior_f * prior_n + float(new_value)) / (prior_n + 1)


def _coerce_talk_dna_payload(
    payload: Any,
    *,
    cfg: TalkDNAConfig,
    metrics: DeterministicMetrics,
) -> dict:
    """Turn the model's JSON output into a schema-conforming TalkDNA dict.

    Validates enums, normalises identifiers, merges deterministic
    numerics with priors (weighted by ``conversation_count``), and
    attaches ``user_id``, ``version``, ``conversation_count``, and
    ``updated_at``. Raises :class:`TalkDNAAnalysisError` for any
    missing/invalid field that can't be repaired without guessing.
    """
    if not isinstance(payload, dict):
        raise TalkDNAAnalysisError(
            f"talk_dna payload not an object: {type(payload).__name__}"
        )

    sarcasm = payload.get("sarcasm_frequency")
    if (
        not isinstance(sarcasm, str)
        or sarcasm not in ALLOWED_SARCASM_FREQUENCIES
    ):
        raise TalkDNAAnalysisError(
            f"sarcasm_frequency not in enum: {sarcasm!r}; "
            f"allowed: {ALLOWED_SARCASM_FREQUENCIES}"
        )
    silence = bool(payload.get("silence_under_pressure", False))

    filler_phrases = _trim_unique_strings(
        payload.get("filler_phrases"), limit=_MAX_LIST_ITEMS
    )
    if not filler_phrases:
        # Fall back to the deterministic candidates when the model
        # returned none — better to surface the habitual hedges we
        # already counted than report an empty list.
        filler_phrases = list(metrics.filler_candidates[:_MAX_LIST_ITEMS])

    escalation_triggers = _trim_unique_strings(
        payload.get("escalation_triggers"), limit=_MAX_LIST_ITEMS
    )
    strengths = _trim_identifier_list(
        payload.get("strengths"), limit=_MAX_LIST_ITEMS
    )
    weaknesses = _trim_identifier_list(
        payload.get("weaknesses"), limit=_MAX_LIST_ITEMS
    )

    prior = cfg.prior_profile if isinstance(cfg.prior_profile, dict) else {}
    prior_patterns = prior.get("patterns") or {}
    prior_count = int(prior.get("conversation_count", 0) or 0)
    prior_version = int(prior.get("version", 0) or 0)

    # Preserve prior strengths/weaknesses when the model returned an
    # empty list — losing observed signal across an incremental update
    # would defeat the point of the rolling profile.
    if not strengths and prior.get("strengths"):
        strengths = _trim_identifier_list(
            prior.get("strengths"), limit=_MAX_LIST_ITEMS
        )
    if not weaknesses and prior.get("weaknesses"):
        weaknesses = _trim_identifier_list(
            prior.get("weaknesses"), limit=_MAX_LIST_ITEMS
        )

    apology_rate = _merge_average(
        prior_patterns.get("apology_rate"),
        prior_count,
        metrics.apology_rate,
    )
    avg_words = _merge_average(
        prior_patterns.get("avg_turn_length_words"),
        prior_count,
        metrics.avg_user_turn_words,
    )

    patterns: dict[str, Any] = {
        "filler_phrases": filler_phrases,
        "apology_rate": round(apology_rate, 4),
        "silence_under_pressure": silence,
        "sarcasm_frequency": sarcasm,
        "escalation_triggers": escalation_triggers,
        "avg_turn_length_words": round(avg_words, 2),
    }
    if metrics.interruption_rate is not None:
        merged_interrupt = _merge_average(
            prior_patterns.get("interruption_rate"),
            prior_count,
            float(metrics.interruption_rate),
        )
        patterns["interruption_rate"] = round(merged_interrupt, 4)
    elif "interruption_rate" in prior_patterns:
        # Carry the prior value forward when this conversation had no
        # timing data — we shouldn't lose the observation just because
        # one conversation came in text-only.
        patterns["interruption_rate"] = prior_patterns["interruption_rate"]

    return {
        "user_id": cfg.user_id,
        "version": prior_version + 1,
        "conversation_count": prior_count + 1,
        "patterns": patterns,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def _resolve_instruction(
    cfg: TalkDNAConfig,
    *,
    turns: list[dict],
    metrics: DeterministicMetrics,
) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    return load_prompt(cfg.prompt_name).render(
        user_id=cfg.user_id,
        transcript=format_transcript(turns),
        metrics_block=_format_metrics_block(metrics),
        prior_profile_block=format_prior_profile(cfg.prior_profile),
    )


def _build_generation_config(
    cfg: TalkDNAConfig,
    sampling: Optional[dict[str, Any]],
) -> GenerationConfig:
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    if sampling is not None:
        gen_cfg.do_sample = True
        gen_cfg.temperature = float(sampling["temperature"])
        gen_cfg.top_p = float(sampling["top_p"])
        gen_cfg.top_k = int(sampling["top_k"])
    return gen_cfg


def analyze_talk_dna(
    processor: Any,
    model: Any,
    turns: list[dict],
    *,
    cfg: Optional[TalkDNAConfig] = None,
) -> dict:
    """Compute or update a TalkDNA profile from one conversation.

    ``turns`` is a list of dicts with at least ``speaker`` (``"user"``
    or ``"other"``) and ``text`` fields. An optional ``emotion`` dict
    on a turn (subset of an EmotionRadar ``emotions`` object) is
    rendered into the prompt for additional signal.

    Inference path is two-shot, mirroring :func:`analyze_emotion`:
    greedy first (deterministic, fastest), then a single retry with
    light sampling (``RETRY_SAMPLING``) on parse / validation failure.
    If both attempts fail, raises :class:`TalkDNAAnalysisError` with
    an ``attempts`` attribute carrying each attempt's raw output
    (truncated to 500 chars) for diagnostics.

    Returns a dict matching ``data/schemas/talk_dna.schema.json``.
    """
    cfg = cfg or TalkDNAConfig()

    if not turns:
        raise TalkDNAAnalysisError("turns is empty — nothing to analyze")
    user_turns = [t for t in turns if t.get("speaker") == "user"]
    if not user_turns:
        raise TalkDNAAnalysisError(
            "no user turns in transcript — TalkDNA needs user-side dialogue"
        )

    metrics = compute_deterministic_metrics(turns)
    instruction = _resolve_instruction(cfg, turns=turns, metrics=metrics)
    messages: list[dict] = [{"role": "user", "content": instruction}]

    attempts: list[dict] = []
    for sampling in (None, RETRY_SAMPLING):
        gen_cfg = _build_generation_config(cfg, sampling)
        raw = chat(processor, model, messages, cfg=gen_cfg)
        text = _extract_text(processor.parse_response(raw)).strip()
        try:
            payload = parse_json(text)
            return _coerce_talk_dna_payload(payload, cfg=cfg, metrics=metrics)
        except (JsonParseError, TalkDNAAnalysisError) as exc:
            attempts.append(
                {
                    "sampling": "greedy" if sampling is None else "light",
                    "error": str(exc),
                    "raw_head": raw[:500],
                }
            )
            LOG.warning(
                "analyze_talk_dna attempt %d (%s) failed: %s",
                len(attempts),
                attempts[-1]["sampling"],
                exc,
            )

    err = TalkDNAAnalysisError(
        f"talk_dna analysis failed after {len(attempts)} attempts; "
        f"last error: {attempts[-1]['error']}"
    )
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err
