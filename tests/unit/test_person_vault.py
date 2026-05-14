"""Unit tests for the Person Vault runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - deterministic metric computation (avg counterparty turn words,
    deflection candidate extraction, interruption-of-user rate)
  - prior-profile formatter used in the prompt
  - ``_coerce_person_vault_payload`` contract: enum validation,
    list merging across conversations, fallback behaviour when the
    model returns empty fields, person_id / name / relationship_type
    resolution, version bumping
  - the runtime's ``ALLOWED_COMMUNICATION_STYLES`` and
    ``ALLOWED_RELATIONSHIP_TYPES`` constants match the schema enums
    on disk (single source of truth check)
  - the default prompt file declares exactly the placeholders the
    runtime renders

The real ``analyze_person_vault`` call requires a loaded model and is
exercised in ``notebooks/phase3/step06_person_vault.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.person_vault import (
    ALLOWED_COMMUNICATION_STYLES,
    ALLOWED_RELATIONSHIP_TYPES,
    DEFAULT_PERSON_VAULT_PROMPT_NAME,
    RETRY_SAMPLING,
    DeterministicPersonMetrics,
    PersonVaultAnalysisError,
    PersonVaultConfig,
    _coerce_person_vault_payload,
    _merge_string_lists,
    _resolve_person_id,
    _resolve_relationship_type,
    compute_person_metrics,
    format_person_prior_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "person_vault.schema.json"
PROMPT_PATH = (
    REPO_ROOT / "data" / "prompts" / f"{DEFAULT_PERSON_VAULT_PROMPT_NAME}.md"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _turns_basic() -> list[dict]:
    """Mixed nine-turn transcript where ``other`` deflects and pushes
    blame back. Used as the canonical small sample for most metric
    tests."""
    return [
        {"speaker": "user", "text": "The report was due Monday."},
        {
            "speaker": "other",
            "text": "I told you the data team hadn't delivered.",
        },
        {"speaker": "user", "text": "Sorry, I kind of forgot."},
        {
            "speaker": "other",
            "text": "Don't blame me for your missed message.",
        },
        {"speaker": "user", "text": "I can't be in every meeting."},
        {
            "speaker": "other",
            "text": "Don't put this on me. You weren't there.",
        },
        {"speaker": "user", "text": "Okay, fair, my bad."},
        {
            "speaker": "other",
            "text": "We can present staging numbers and flag the gaps.",
        },
        {"speaker": "user", "text": "Good. Let's regroup at six."},
    ]


def _minimal_llm_payload(**overrides):
    base = {
        "communication_style": "direct",
        "emotional_triggers": ["being blamed", "missed messages"],
        "de_escalation_keys": ["specific next step", "shared plan"],
        "common_deflections": ["don't blame me", "I told you"],
        "responds_best_to": (
            "Concrete numbers and a single decision. Loses patience "
            "with hypotheticals."
        ),
        "cultural_context": None,
    }
    base.update(overrides)
    return base


def _basic_metrics(**overrides) -> DeterministicPersonMetrics:
    base = {
        "other_turn_count": 4,
        "total_turn_count": 9,
        "avg_other_turn_words": 9.0,
        "deflection_candidates": ["don't blame me", "i told you"],
        "interruption_of_user_rate": None,
    }
    base.update(overrides)
    return DeterministicPersonMetrics(**base)


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_counts_only_other_turns():
    metrics = compute_person_metrics(_turns_basic())
    assert metrics.other_turn_count == 4
    assert metrics.total_turn_count == 9


def test_compute_metrics_avg_other_turn_words_uses_other_only():
    turns = [
        {"speaker": "user", "text": "this should not count one two three"},
        {"speaker": "other", "text": "one two three four"},  # 4 words
        {"speaker": "other", "text": "one two"},  # 2 words
    ]
    metrics = compute_person_metrics(turns)
    assert metrics.avg_other_turn_words == pytest.approx(3.0)


def test_compute_metrics_deflection_candidates_picked_up_from_other():
    metrics = compute_person_metrics(_turns_basic())
    # 'don't blame me', 'i told you', 'you weren't there',
    # "don't put this on me" should all surface.
    surfaced = set(metrics.deflection_candidates)
    assert "don't blame me" in surfaced
    assert "i told you" in surfaced
    assert "you weren't there" in surfaced
    assert "don't put this on me" in surfaced


def test_compute_metrics_deflection_ignores_user_phrases():
    """A user saying a deflection-shaped phrase should not credit the
    counterparty."""
    turns = [
        {"speaker": "user", "text": "Don't blame me — I tried."},
        {"speaker": "other", "text": "OK, let's move on."},
    ]
    metrics = compute_person_metrics(turns)
    assert metrics.deflection_candidates == []


def test_compute_metrics_deflection_candidates_sorted_by_frequency():
    turns = [
        {"speaker": "other", "text": "I told you. I told you twice. As I already said."},
        {"speaker": "other", "text": "You always do this."},
    ]
    metrics = compute_person_metrics(turns)
    # 'i told you' fires both in turn 1 (twice — including the 'as i
    # already said' fallback regex on 'said'); 'you always' fires once.
    assert metrics.deflection_candidates[0] == "i told you"


def test_compute_metrics_deflection_empty_when_no_cues():
    turns = [
        {"speaker": "other", "text": "Let me check the numbers and get back to you."},
    ]
    metrics = compute_person_metrics(turns)
    assert metrics.deflection_candidates == []


def test_compute_metrics_interruption_of_user_rate_none_without_timing():
    metrics = compute_person_metrics(_turns_basic())
    assert metrics.interruption_of_user_rate is None


def test_compute_metrics_interruption_of_user_detects_overlap():
    """Other starts at t=4.5 while user ended at t=5.0 → interruption."""
    turns = [
        {"speaker": "user", "text": "I think", "start_seconds": 0.0, "end_seconds": 5.0},
        {"speaker": "other", "text": "No", "start_seconds": 4.5, "end_seconds": 5.5},
        {"speaker": "user", "text": "Let me", "start_seconds": 5.6, "end_seconds": 7.0},
        {"speaker": "other", "text": "OK", "start_seconds": 7.5, "end_seconds": 8.0},
    ]
    metrics = compute_person_metrics(turns)
    # Two other-after-user transitions; one overlaps.
    assert metrics.interruption_of_user_rate == pytest.approx(0.5)


def test_compute_metrics_handles_empty_transcript():
    metrics = compute_person_metrics([])
    assert metrics.other_turn_count == 0
    assert metrics.avg_other_turn_words == 0.0
    assert metrics.deflection_candidates == []
    assert metrics.interruption_of_user_rate is None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_format_person_prior_profile_empty_returns_sentinel():
    text = format_person_prior_profile(None)
    assert "first" in text.lower()


def test_format_person_prior_profile_empty_dict_returns_sentinel():
    text = format_person_prior_profile({})
    assert "first" in text.lower()


def test_format_person_prior_profile_includes_fields():
    prior = {
        "version": 2,
        "conversation_count": 3,
        "profile": {
            "communication_style": "passive_aggressive",
            "emotional_triggers": ["being blamed"],
            "de_escalation_keys": ["concrete plan"],
            "common_deflections": ["I told you", "you weren't there"],
            "responds_best_to": "Concrete numbers and a single decision.",
            "cultural_context": "Engineering-lead context — values directness.",
        },
    }
    text = format_person_prior_profile(prior)
    assert "version: 2" in text
    assert "conversation_count: 3" in text
    assert "passive_aggressive" in text
    assert "being blamed" in text
    assert "concrete plan" in text
    assert "I told you" in text
    assert "Concrete numbers" in text
    assert "Engineering-lead" in text


# ---------------------------------------------------------------------------
# _merge_string_lists and helpers
# ---------------------------------------------------------------------------


def test_merge_string_lists_dedupes_case_insensitively():
    merged = _merge_string_lists(["Being Blamed"], ["being blamed"], limit=8)
    assert merged == ["Being Blamed"]


def test_merge_string_lists_appends_new_items():
    merged = _merge_string_lists(
        ["being blamed"], ["missed deadlines"], limit=8
    )
    assert merged == ["being blamed", "missed deadlines"]


def test_merge_string_lists_caps_total_length():
    prior = [f"old_{i}" for i in range(6)]
    new = [f"new_{i}" for i in range(6)]
    merged = _merge_string_lists(prior, new, limit=5)
    # Cap is enforced — only the tail survives so recent observations win.
    assert len(merged) == 5
    assert "new_5" in merged


def test_merge_string_lists_handles_non_list_prior():
    merged = _merge_string_lists(None, ["being blamed"], limit=8)
    assert merged == ["being blamed"]


def test_resolve_relationship_type_uses_cfg_when_valid():
    rt = _resolve_relationship_type(
        PersonVaultConfig(name="Mom", relationship_type="parent"), prior=None
    )
    assert rt == "parent"


def test_resolve_relationship_type_rejects_invalid_cfg():
    with pytest.raises(PersonVaultAnalysisError):
        _resolve_relationship_type(
            PersonVaultConfig(name="X", relationship_type="frenemy"),
            prior=None,
        )


def test_resolve_relationship_type_inherits_from_prior():
    rt = _resolve_relationship_type(
        PersonVaultConfig(name="Mom"),
        prior={"relationship_type": "parent"},
    )
    assert rt == "parent"


def test_resolve_relationship_type_defaults_to_other():
    rt = _resolve_relationship_type(
        PersonVaultConfig(name="X"), prior=None
    )
    assert rt == "other"


def test_resolve_person_id_prefers_cfg():
    pid = _resolve_person_id(
        PersonVaultConfig(name="X", person_id="explicit_id"),
        prior={"person_id": "prior_id"},
    )
    assert pid == "explicit_id"


def test_resolve_person_id_inherits_from_prior():
    pid = _resolve_person_id(
        PersonVaultConfig(name="X"),
        prior={"person_id": "prior_id"},
    )
    assert pid == "prior_id"


def test_resolve_person_id_generates_when_missing():
    pid = _resolve_person_id(PersonVaultConfig(name="X"), prior=None)
    assert pid.startswith("person_")
    assert len(pid) > len("person_")


# ---------------------------------------------------------------------------
# _coerce_person_vault_payload: happy path
# ---------------------------------------------------------------------------


def test_coerce_returns_required_top_level_fields():
    result = _coerce_person_vault_payload(
        _minimal_llm_payload(),
        cfg=PersonVaultConfig(name="Jamie"),
        metrics=_basic_metrics(),
    )
    for key in (
        "person_id",
        "name",
        "version",
        "conversation_count",
        "profile",
        "updated_at",
    ):
        assert key in result, f"missing required key: {key}"


def test_coerce_uses_cfg_name():
    result = _coerce_person_vault_payload(
        _minimal_llm_payload(),
        cfg=PersonVaultConfig(name="Manager Sarah"),
        metrics=_basic_metrics(),
    )
    assert result["name"] == "Manager Sarah"


def test_coerce_bumps_version_to_one_for_first_conversation():
    result = _coerce_person_vault_payload(
        _minimal_llm_payload(),
        cfg=PersonVaultConfig(name="Jamie"),
        metrics=_basic_metrics(),
    )
    assert result["version"] == 1
    assert result["conversation_count"] == 1


def test_coerce_bumps_version_off_prior_profile():
    prior = {
        "person_id": "person_abc",
        "name": "Jamie",
        "relationship_type": "colleague",
        "version": 4,
        "conversation_count": 4,
        "profile": {
            "communication_style": "direct",
            "emotional_triggers": ["being blamed"],
            "de_escalation_keys": ["concrete plan"],
            "common_deflections": ["I told you"],
        },
    }
    result = _coerce_person_vault_payload(
        _minimal_llm_payload(),
        cfg=PersonVaultConfig(name="Jamie", prior_profile=prior),
        metrics=_basic_metrics(),
    )
    assert result["version"] == 5
    assert result["conversation_count"] == 5
    # person_id inherited from prior.
    assert result["person_id"] == "person_abc"


def test_coerce_accumulates_emotional_triggers_across_conversations():
    prior = {
        "person_id": "person_abc",
        "name": "Jamie",
        "version": 1,
        "conversation_count": 1,
        "profile": {
            "communication_style": "direct",
            "emotional_triggers": ["being blamed"],
            "de_escalation_keys": [],
            "common_deflections": [],
        },
    }
    payload = _minimal_llm_payload(
        emotional_triggers=["missed deadlines"],
    )
    result = _coerce_person_vault_payload(
        payload,
        cfg=PersonVaultConfig(name="Jamie", prior_profile=prior),
        metrics=_basic_metrics(),
    )
    triggers = result["profile"]["emotional_triggers"]
    assert "being blamed" in triggers
    assert "missed deadlines" in triggers


def test_coerce_falls_back_to_deterministic_deflections_when_llm_empty():
    payload = _minimal_llm_payload(common_deflections=[])
    result = _coerce_person_vault_payload(
        payload,
        cfg=PersonVaultConfig(name="Jamie"),
        metrics=_basic_metrics(
            deflection_candidates=["don't blame me", "i told you"]
        ),
    )
    deflections = result["profile"]["common_deflections"]
    assert "don't blame me" in deflections
    assert "i told you" in deflections


def test_coerce_inherits_responds_best_to_when_model_returns_empty():
    prior = {
        "person_id": "p1",
        "name": "Jamie",
        "version": 1,
        "conversation_count": 1,
        "profile": {
            "communication_style": "direct",
            "responds_best_to": "Concrete numbers and a single decision.",
        },
    }
    payload = _minimal_llm_payload(responds_best_to="")
    result = _coerce_person_vault_payload(
        payload,
        cfg=PersonVaultConfig(name="Jamie", prior_profile=prior),
        metrics=_basic_metrics(),
    )
    assert (
        result["profile"]["responds_best_to"]
        == "Concrete numbers and a single decision."
    )


def test_coerce_preserves_prior_relationship_pulse():
    """Step 06 doesn't compute pulse, but if a caller had Step 11 fill
    one in earlier, the incremental update mustn't drop it."""
    prior = {
        "person_id": "p1",
        "name": "Jamie",
        "version": 1,
        "conversation_count": 1,
        "profile": {"communication_style": "direct"},
        "relationship_pulse": {
            "trend": "improving",
            "unresolved_items": ["missed deadline"],
        },
    }
    result = _coerce_person_vault_payload(
        _minimal_llm_payload(),
        cfg=PersonVaultConfig(name="Jamie", prior_profile=prior),
        metrics=_basic_metrics(),
    )
    assert result["relationship_pulse"]["trend"] == "improving"
    assert "missed deadline" in result["relationship_pulse"]["unresolved_items"]


def test_coerce_clamps_overly_long_responds_best_to():
    long_text = "x " * 200  # 400 chars
    payload = _minimal_llm_payload(responds_best_to=long_text)
    result = _coerce_person_vault_payload(
        payload,
        cfg=PersonVaultConfig(name="Jamie"),
        metrics=_basic_metrics(),
    )
    assert len(result["profile"]["responds_best_to"]) <= 280


# ---------------------------------------------------------------------------
# _coerce_person_vault_payload: error paths
# ---------------------------------------------------------------------------


def test_coerce_rejects_non_dict_payload():
    with pytest.raises(PersonVaultAnalysisError):
        _coerce_person_vault_payload(
            "not a dict",
            cfg=PersonVaultConfig(name="X"),
            metrics=_basic_metrics(),
        )


def test_coerce_rejects_invalid_communication_style():
    payload = _minimal_llm_payload(communication_style="unknown_style")
    with pytest.raises(PersonVaultAnalysisError):
        _coerce_person_vault_payload(
            payload,
            cfg=PersonVaultConfig(name="X"),
            metrics=_basic_metrics(),
        )


def test_coerce_rejects_missing_communication_style():
    payload = _minimal_llm_payload()
    del payload["communication_style"]
    with pytest.raises(PersonVaultAnalysisError):
        _coerce_person_vault_payload(
            payload,
            cfg=PersonVaultConfig(name="X"),
            metrics=_basic_metrics(),
        )


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_person_vault_schema_required_fields_match_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "PersonVault"
    for field in (
        "person_id",
        "name",
        "version",
        "profile",
        "conversation_count",
        "updated_at",
    ):
        assert field in schema["required"], f"{field!r} must be required"


def test_allowed_communication_styles_match_schema_enum():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(
        schema["properties"]["profile"]["properties"]["communication_style"][
            "enum"
        ]
    )
    assert set(ALLOWED_COMMUNICATION_STYLES) == enum


def test_allowed_relationship_types_match_schema_enum():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(schema["properties"]["relationship_type"]["enum"])
    assert set(ALLOWED_RELATIONSHIP_TYPES) == enum


def test_person_vault_prompt_has_all_placeholders():
    """The default prompt file must exist and accept exactly the five
    placeholders the runtime renders. Drift breaks ``Prompt.render``."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "$person_name",
        "$relationship_type_block",
        "$prior_profile_block",
        "$metrics_block",
        "$transcript",
    ):
        assert placeholder in text, f"prompt missing {placeholder}"


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Mirror the rationale from talk_dna / emotion: just enough
    randomness to escape a single bad greedy trajectory. Pinned so it
    doesn't drift."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1
