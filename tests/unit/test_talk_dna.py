"""Unit tests for the Talk DNA runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - deterministic metric computation (apology detection, filler
    extraction, avg user-turn words, interruption rate)
  - transcript and prior-profile formatters used in the prompt
  - ``_coerce_talk_dna_payload`` contract: enum validation,
    identifier normalisation, weighted prior merging, fallback
    behaviour when the model returns empty fields
  - the runtime's ``ALLOWED_SARCASM_FREQUENCIES`` constant matches
    the schema enum on disk (single source of truth check)
  - the default prompt file declares exactly the placeholders the
    runtime renders

The real ``analyze_talk_dna`` call requires a loaded model and is
exercised in ``notebooks/phase3/step05_talk_dna.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.talk_dna import (
    ALLOWED_SARCASM_FREQUENCIES,
    DEFAULT_TALK_DNA_PROMPT_NAME,
    RETRY_SAMPLING,
    DeterministicMetrics,
    TalkDNAAnalysisError,
    TalkDNAConfig,
    _coerce_talk_dna_payload,
    _merge_average,
    _normalize_identifier,
    compute_deterministic_metrics,
    format_prior_profile,
    format_transcript,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "talk_dna.schema.json"
PROMPT_PATH = REPO_ROOT / "data" / "prompts" / f"{DEFAULT_TALK_DNA_PROMPT_NAME}.md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _turns_basic() -> list[dict]:
    """Mixed nine-turn transcript with apologies, hedges, and an 'other'
    party. Used as the canonical small sample for most metric tests."""
    return [
        {"speaker": "user", "text": "I just wanted to ask about the report."},
        {"speaker": "other", "text": "What about it?"},
        {"speaker": "user", "text": "Sorry, I kind of forgot to follow up."},
        {"speaker": "other", "text": "You also missed the meeting."},
        {"speaker": "user", "text": "I think I sort of dropped the ball."},
        {"speaker": "other", "text": "We need to fix this tonight."},
        {"speaker": "user", "text": "I'm sorry, you're right. I'll regroup at six."},
        {"speaker": "other", "text": "Good. Send me the partial numbers."},
        {"speaker": "user", "text": "I will. Apologies again — I just want to fix it."},
    ]


def _minimal_llm_payload(**overrides):
    base = {
        "filler_phrases": ["I just", "kind of"],
        "sarcasm_frequency": "low",
        "silence_under_pressure": False,
        "escalation_triggers": ["being blamed"],
        "strengths": ["acknowledges_mistakes"],
        "weaknesses": ["over_apologizes"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_counts_only_user_turns():
    metrics = compute_deterministic_metrics(_turns_basic())
    assert metrics.user_turn_count == 5
    assert metrics.total_turn_count == 9


def test_compute_metrics_apology_rate_captures_sorry_and_apologies():
    metrics = compute_deterministic_metrics(_turns_basic())
    # User turns 3, 7, 9 contain apologies — 3 of 5.
    assert metrics.apology_rate == pytest.approx(3 / 5)


def test_compute_metrics_apology_rate_zero_when_no_apologies():
    turns = [
        {"speaker": "user", "text": "Just give me the report."},
        {"speaker": "other", "text": "Working on it."},
    ]
    metrics = compute_deterministic_metrics(turns)
    assert metrics.apology_rate == 0.0


def test_compute_metrics_apology_does_not_count_other_speaker():
    """A counterparty saying 'sorry' should not inflate the user's rate."""
    turns = [
        {"speaker": "user", "text": "Where's the data?"},
        {"speaker": "other", "text": "I'm sorry, I'll send it."},
    ]
    metrics = compute_deterministic_metrics(turns)
    assert metrics.apology_rate == 0.0


def test_compute_metrics_apology_handles_my_bad_and_my_fault():
    turns = [
        {"speaker": "user", "text": "My bad, I missed the meeting."},
        {"speaker": "user", "text": "That's my fault — I should've checked."},
    ]
    metrics = compute_deterministic_metrics(turns)
    assert metrics.apology_rate == 1.0


def test_compute_metrics_avg_turn_length_words_uses_user_only():
    turns = [
        {"speaker": "user", "text": "one two three four"},  # 4 words
        {"speaker": "other", "text": "a b c d e f g h"},  # 8 words — ignored
        {"speaker": "user", "text": "one two"},  # 2 words
    ]
    metrics = compute_deterministic_metrics(turns)
    assert metrics.avg_user_turn_words == pytest.approx(3.0)


def test_compute_metrics_filler_candidates_only_surface_repeat_offenders():
    """A filler used once doesn't count — we want habitual patterns only."""
    turns = [
        {"speaker": "user", "text": "I just want it done."},
        {"speaker": "user", "text": "I just need clarity."},  # 2x 'just'
        {"speaker": "user", "text": "Maybe later."},  # 1x 'maybe' — dropped
    ]
    metrics = compute_deterministic_metrics(turns)
    assert "just" in metrics.filler_candidates
    assert "maybe" not in metrics.filler_candidates


def test_compute_metrics_filler_candidates_sorted_by_frequency():
    turns = [
        {"speaker": "user", "text": "I just think I just want it done."},
        {"speaker": "user", "text": "Kind of like that — kind of."},
        {"speaker": "user", "text": "I just kind of agree."},
    ]
    metrics = compute_deterministic_metrics(turns)
    # 'just' appears 3x, 'kind of' appears 3x, 'i think' appears 1x.
    assert metrics.filler_candidates[0] in ("just", "kind of")
    assert "i think" not in metrics.filler_candidates


def test_compute_metrics_filler_candidates_empty_when_no_hedges():
    metrics = compute_deterministic_metrics(
        [{"speaker": "user", "text": "Send the report by noon."}]
    )
    assert metrics.filler_candidates == []


def test_compute_metrics_interruption_rate_none_without_timing():
    metrics = compute_deterministic_metrics(_turns_basic())
    assert metrics.interruption_rate is None


def test_compute_metrics_interruption_rate_detects_overlap():
    """User starts at t=4.5 while 'other' ended at t=5.0 → interruption."""
    turns = [
        {"speaker": "other", "text": "Listen", "start_seconds": 0.0, "end_seconds": 5.0},
        {"speaker": "user", "text": "No.", "start_seconds": 4.5, "end_seconds": 5.5},
        {"speaker": "other", "text": "OK", "start_seconds": 5.6, "end_seconds": 7.0},
        {"speaker": "user", "text": "Fine", "start_seconds": 7.5, "end_seconds": 8.0},
    ]
    metrics = compute_deterministic_metrics(turns)
    # Two user-after-other transitions; one overlaps.
    assert metrics.interruption_rate == pytest.approx(0.5)


def test_compute_metrics_interruption_rate_zero_when_all_clean():
    turns = [
        {"speaker": "other", "text": "A", "start_seconds": 0.0, "end_seconds": 1.0},
        {"speaker": "user", "text": "B", "start_seconds": 1.1, "end_seconds": 2.0},
        {"speaker": "other", "text": "C", "start_seconds": 2.2, "end_seconds": 3.0},
        {"speaker": "user", "text": "D", "start_seconds": 3.1, "end_seconds": 4.0},
    ]
    metrics = compute_deterministic_metrics(turns)
    assert metrics.interruption_rate == 0.0


def test_compute_metrics_handles_empty_transcript():
    metrics = compute_deterministic_metrics([])
    assert metrics.user_turn_count == 0
    assert metrics.apology_rate == 0.0
    assert metrics.avg_user_turn_words == 0.0
    assert metrics.filler_candidates == []
    assert metrics.interruption_rate is None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_format_transcript_includes_index_speaker_and_text():
    text = format_transcript(
        [
            {"speaker": "user", "text": "Hello there"},
            {"speaker": "other", "text": "Hi"},
        ]
    )
    assert "[1] user" in text
    assert "Hello there" in text
    assert "[2] other" in text
    assert "Hi" in text


def test_format_transcript_renders_emotion_tag_when_present():
    text = format_transcript(
        [
            {
                "speaker": "user",
                "text": "Fine",
                "emotion": {"primary": "frustration", "intensity": 0.75},
            }
        ]
    )
    assert "frustration" in text
    assert "0.75" in text


def test_format_transcript_omits_emotion_tag_when_missing():
    text = format_transcript([{"speaker": "user", "text": "Hi"}])
    assert "(" not in text  # no emotion parens


def test_format_transcript_empty_returns_sentinel():
    text = format_transcript([])
    assert "empty" in text.lower()


def test_format_prior_profile_empty_returns_sentinel():
    text = format_prior_profile(None)
    assert "first" in text.lower()


def test_format_prior_profile_empty_dict_returns_sentinel():
    text = format_prior_profile({})
    assert "first" in text.lower()


def test_format_prior_profile_includes_patterns_and_lists():
    prior = {
        "version": 2,
        "conversation_count": 3,
        "patterns": {
            "apology_rate": 0.35,
            "filler_phrases": ["just", "kind of"],
            "sarcasm_frequency": "rare",
            "silence_under_pressure": False,
            "escalation_triggers": ["being blamed"],
        },
        "strengths": ["acknowledges_mistakes"],
        "weaknesses": ["over_apologizes"],
    }
    text = format_prior_profile(prior)
    assert "version: 2" in text
    assert "conversation_count: 3" in text
    assert "0.35" in text
    assert "just" in text
    assert "rare" in text
    assert "being blamed" in text
    assert "acknowledges_mistakes" in text
    assert "over_apologizes" in text


# ---------------------------------------------------------------------------
# _normalize_identifier and _merge_average helpers
# ---------------------------------------------------------------------------


def test_normalize_identifier_handles_spaces_and_punctuation():
    assert _normalize_identifier("Over Apologizes!") == "over_apologizes"
    assert _normalize_identifier("clear-problem-statement") == "clear_problem_statement"
    assert _normalize_identifier("LISTENS_ACTIVELY") == "listens_actively"


def test_normalize_identifier_strips_leading_trailing_underscores():
    assert _normalize_identifier("___hedges___") == "hedges"


def test_merge_average_falls_back_to_new_when_no_prior():
    assert _merge_average(None, 0, 0.5) == 0.5
    assert _merge_average(0.7, 0, 0.5) == 0.5  # prior_n = 0 → ignore prior


def test_merge_average_weighted_by_prior_count():
    # prior 0.4 over 3 conversations, new 0.8 → (0.4*3 + 0.8) / 4 = 0.5
    assert _merge_average(0.4, 3, 0.8) == pytest.approx(0.5)


def test_merge_average_handles_unparseable_prior():
    """A prior dict from an older schema version might have a non-numeric
    value — we should fall back gracefully, not crash."""
    assert _merge_average("unknown", 3, 0.4) == 0.4


# ---------------------------------------------------------------------------
# _coerce_talk_dna_payload: happy path
# ---------------------------------------------------------------------------


def _basic_metrics(**overrides) -> DeterministicMetrics:
    base = {
        "apology_rate": 0.4,
        "avg_user_turn_words": 10.0,
        "filler_candidates": ["just", "kind of"],
        "interruption_rate": None,
        "user_turn_count": 5,
        "total_turn_count": 9,
    }
    base.update(overrides)
    return DeterministicMetrics(**base)


def test_coerce_returns_required_top_level_fields():
    result = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(),
        metrics=_basic_metrics(),
    )
    for key in ("user_id", "version", "patterns", "strengths", "weaknesses", "updated_at"):
        assert key in result, f"missing required key: {key}"


def test_coerce_uses_cfg_user_id():
    result = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(user_id="alice"),
        metrics=_basic_metrics(),
    )
    assert result["user_id"] == "alice"


def test_coerce_bumps_version_to_one_for_first_conversation():
    result = _coerce_talk_dna_payload(
        _minimal_llm_payload(), cfg=TalkDNAConfig(), metrics=_basic_metrics()
    )
    assert result["version"] == 1
    assert result["conversation_count"] == 1


def test_coerce_bumps_version_off_prior_profile():
    prior = {
        "version": 4,
        "conversation_count": 4,
        "patterns": {"apology_rate": 0.2, "avg_turn_length_words": 8.0},
        "strengths": [],
        "weaknesses": [],
    }
    result = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(prior_profile=prior),
        metrics=_basic_metrics(),
    )
    assert result["version"] == 5
    assert result["conversation_count"] == 5


def test_coerce_weighted_averages_apology_rate_with_prior():
    """Prior 0.2 over 4 conversations + new 0.4 → (0.2*4 + 0.4)/5 = 0.24."""
    prior = {
        "version": 4,
        "conversation_count": 4,
        "patterns": {"apology_rate": 0.2, "avg_turn_length_words": 8.0},
    }
    result = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(prior_profile=prior),
        metrics=_basic_metrics(apology_rate=0.4),
    )
    assert result["patterns"]["apology_rate"] == pytest.approx(0.24)


def test_coerce_includes_interruption_rate_only_when_metrics_have_it():
    no_timing = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(),
        metrics=_basic_metrics(interruption_rate=None),
    )
    assert "interruption_rate" not in no_timing["patterns"]

    with_timing = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(),
        metrics=_basic_metrics(interruption_rate=0.25),
    )
    assert with_timing["patterns"]["interruption_rate"] == pytest.approx(0.25)


def test_coerce_carries_prior_interruption_when_new_metrics_have_none():
    """A text-only follow-up conversation shouldn't lose an interruption
    rate already observed on the audio-derived prior profile."""
    prior = {
        "version": 1,
        "conversation_count": 1,
        "patterns": {
            "apology_rate": 0.1,
            "avg_turn_length_words": 9.0,
            "interruption_rate": 0.3,
        },
    }
    result = _coerce_talk_dna_payload(
        _minimal_llm_payload(),
        cfg=TalkDNAConfig(prior_profile=prior),
        metrics=_basic_metrics(interruption_rate=None),
    )
    assert result["patterns"]["interruption_rate"] == 0.3


def test_coerce_normalizes_strength_identifiers_to_snake_case():
    payload = _minimal_llm_payload(
        strengths=["Acknowledges Mistakes", "clear-problem-statement"]
    )
    result = _coerce_talk_dna_payload(
        payload, cfg=TalkDNAConfig(), metrics=_basic_metrics()
    )
    assert "acknowledges_mistakes" in result["strengths"]
    assert "clear_problem_statement" in result["strengths"]


def test_coerce_dedupes_strengths():
    payload = _minimal_llm_payload(
        strengths=["listens_actively", "Listens Actively", "LISTENS_ACTIVELY"]
    )
    result = _coerce_talk_dna_payload(
        payload, cfg=TalkDNAConfig(), metrics=_basic_metrics()
    )
    assert result["strengths"] == ["listens_actively"]


def test_coerce_falls_back_to_deterministic_filler_candidates_when_llm_empty():
    payload = _minimal_llm_payload(filler_phrases=[])
    result = _coerce_talk_dna_payload(
        payload,
        cfg=TalkDNAConfig(),
        metrics=_basic_metrics(filler_candidates=["just", "kind of"]),
    )
    assert "just" in result["patterns"]["filler_phrases"]
    assert "kind of" in result["patterns"]["filler_phrases"]


def test_coerce_preserves_prior_strengths_when_model_returns_empty():
    """An incremental update where this conversation revealed no new
    strengths shouldn't blow away strengths the model surfaced on prior
    conversations."""
    prior = {
        "version": 2,
        "conversation_count": 2,
        "patterns": {"apology_rate": 0.2, "avg_turn_length_words": 9.0},
        "strengths": ["listens_actively"],
        "weaknesses": ["over_apologizes"],
    }
    payload = _minimal_llm_payload(strengths=[], weaknesses=[])
    result = _coerce_talk_dna_payload(
        payload,
        cfg=TalkDNAConfig(prior_profile=prior),
        metrics=_basic_metrics(),
    )
    assert result["strengths"] == ["listens_actively"]
    assert result["weaknesses"] == ["over_apologizes"]


# ---------------------------------------------------------------------------
# _coerce_talk_dna_payload: error paths
# ---------------------------------------------------------------------------


def test_coerce_rejects_non_dict_payload():
    with pytest.raises(TalkDNAAnalysisError):
        _coerce_talk_dna_payload(
            "not a dict", cfg=TalkDNAConfig(), metrics=_basic_metrics()
        )


def test_coerce_rejects_invalid_sarcasm_frequency():
    payload = _minimal_llm_payload(sarcasm_frequency="medium")
    with pytest.raises(TalkDNAAnalysisError):
        _coerce_talk_dna_payload(
            payload, cfg=TalkDNAConfig(), metrics=_basic_metrics()
        )


def test_coerce_rejects_missing_sarcasm_frequency():
    payload = _minimal_llm_payload()
    del payload["sarcasm_frequency"]
    with pytest.raises(TalkDNAAnalysisError):
        _coerce_talk_dna_payload(
            payload, cfg=TalkDNAConfig(), metrics=_basic_metrics()
        )


def test_coerce_drops_non_identifier_strengths_silently():
    """A label with spaces gets normalised; a label that is empty after
    normalisation is dropped (no crash, no garbage)."""
    payload = _minimal_llm_payload(strengths=["!!!", "", "good_pacing"])
    result = _coerce_talk_dna_payload(
        payload, cfg=TalkDNAConfig(), metrics=_basic_metrics()
    )
    assert result["strengths"] == ["good_pacing"]


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_talk_dna_schema_required_fields_match_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "TalkDNA"
    for field in ("user_id", "version", "patterns", "strengths", "weaknesses", "updated_at"):
        assert field in schema["required"], f"{field!r} must be required"


def test_allowed_sarcasm_frequencies_match_schema_enum():
    """The runtime constant must equal the on-disk enum. Drift here means
    the model would be allowed to emit a label the schema rejects."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(
        schema["properties"]["patterns"]["properties"]["sarcasm_frequency"]["enum"]
    )
    assert set(ALLOWED_SARCASM_FREQUENCIES) == enum


def test_talk_dna_prompt_has_all_placeholders():
    """The default prompt file must exist and accept exactly the four
    placeholders the runtime renders. Drift breaks ``Prompt.render``."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in ("$user_id", "$transcript", "$metrics_block", "$prior_profile_block"):
        assert placeholder in text, f"prompt missing {placeholder}"


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Mirror the rationale from emotion radar: just enough randomness to
    escape a single bad greedy trajectory. Pinned so it doesn't drift."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1
