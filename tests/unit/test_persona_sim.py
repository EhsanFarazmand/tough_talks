"""Unit tests for the Adversarial Persona Simulator runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - ``resistance_type`` normalisation: casing, separators, synonyms
  - ``_coerce_persona_reply_payload`` contract: enum validation,
    persona_name forced from cfg, escalation clamping, reply
    non-empty / cap
  - ``build_practice_messages`` shape: system at index 0, alternating
    user / assistant, prior persona turns serialised back to JSON
  - profile-block formatting accepts both full PersonVault dicts and
    inner profile dicts
  - the runtime's ``ALLOWED_RESISTANCE_TYPES`` matches the schema enum
    on disk (single source of truth check)
  - the default prompt file declares exactly the placeholders the
    runtime renders

The real ``generate_persona_reply`` call requires a loaded model and is
exercised in ``notebooks/phase3/step07_persona_sim.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.persona_sim import (
    ALLOWED_RESISTANCE_TYPES,
    DEFAULT_PERSONA_PROMPT_NAME,
    RETRY_SAMPLING,
    PersonaReplyError,
    PersonaSimConfig,
    _RESISTANCE_TYPE_SYNONYMS,
    _clamp_unit,
    _coerce_persona_reply_payload,
    _normalize_resistance_type,
    _resolve_inner_profile,
    _resolve_persona_name,
    _resolve_relationship_block,
    build_practice_messages,
    format_persona_profile_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "persona_reply.schema.json"
PROMPT_PATH = (
    REPO_ROOT / "data" / "prompts" / f"{DEFAULT_PERSONA_PROMPT_NAME}.md"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_person_vault(**overrides) -> dict:
    """A canonical PersonVault wrapper dict — what Step 06 emits.

    Used to verify that the persona runtime accepts the full dict
    (with the top-level ``name`` / ``relationship_type`` / inner
    ``profile``) and not just the inner profile object.
    """
    base = {
        "person_id": "person_test",
        "name": "Jamie",
        "relationship_type": "colleague",
        "version": 2,
        "conversation_count": 2,
        "profile": {
            "communication_style": "defensive",
            "emotional_triggers": ["citing past commitments", "missed messages"],
            "de_escalation_keys": [
                "explicitly disowning blame",
                "concrete next step",
            ],
            "common_deflections": [
                "I told you",
                "Don't blame me",
                "Don't put this on me",
            ],
            "responds_best_to": (
                "Concrete numbers and a single decision. Loses patience "
                "with hypotheticals."
            ),
        },
    }
    base.update(overrides)
    return base


def _minimal_llm_payload(**overrides) -> dict:
    base = {
        "persona_name": "Jamie",
        "reply": "I told you on Tuesday the data team hadn't delivered.",
        "resistance_type": "deflect",
        "escalation_level": 0.55,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# resistance_type normalisation
# ---------------------------------------------------------------------------


def test_normalize_resistance_accepts_canonical_values():
    for rt in ALLOWED_RESISTANCE_TYPES:
        assert _normalize_resistance_type(rt) == rt


def test_normalize_resistance_handles_casing():
    assert _normalize_resistance_type("Deflect") == "deflect"
    assert _normalize_resistance_type("DEFLECT") == "deflect"
    assert _normalize_resistance_type("Counter_Attack") == "counter_attack"


def test_normalize_resistance_handles_hyphen_separator():
    assert _normalize_resistance_type("counter-attack") == "counter_attack"
    assert _normalize_resistance_type("guilt-trip") == "guilt_trip"


def test_normalize_resistance_handles_space_separator():
    assert _normalize_resistance_type("counter attack") == "counter_attack"
    assert _normalize_resistance_type("Guilt Trip") == "guilt_trip"


def test_normalize_resistance_maps_denial_to_deny():
    assert _normalize_resistance_type("denial") == "deny"


def test_normalize_resistance_maps_counterattack_to_counter_attack():
    """The model frequently emits the unhyphenated single-word form."""
    assert _normalize_resistance_type("counterattack") == "counter_attack"


def test_normalize_resistance_maps_silence_to_silent():
    assert _normalize_resistance_type("silence") == "silent"
    assert _normalize_resistance_type("stonewalling") == "silent"


def test_normalize_resistance_maps_concession_to_concede():
    assert _normalize_resistance_type("concession") == "concede"
    assert _normalize_resistance_type("agreement") == "concede"


def test_normalize_resistance_maps_deflection_to_deflect():
    assert _normalize_resistance_type("deflection") == "deflect"
    assert _normalize_resistance_type("evasion") == "deflect"


def test_normalize_resistance_returns_none_for_unrecognised():
    assert _normalize_resistance_type("transcendence") is None
    assert _normalize_resistance_type("") is None
    assert _normalize_resistance_type(None) is None
    assert _normalize_resistance_type(42) is None


def test_synonym_map_values_all_in_enum():
    """Every synonym must point at a canonical enum value — otherwise
    the normaliser could return a string the coercer then rejects.
    Mirrors ``test_synonym_map_values_all_in_enum`` in Step 06."""
    for synonym, canonical in _RESISTANCE_TYPE_SYNONYMS.items():
        assert canonical in ALLOWED_RESISTANCE_TYPES, (
            f"synonym {synonym!r} maps to {canonical!r}, "
            f"which is not in ALLOWED_RESISTANCE_TYPES"
        )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def test_clamp_unit_within_range():
    assert _clamp_unit(0.0) == 0.0
    assert _clamp_unit(0.5) == 0.5
    assert _clamp_unit(1.0) == 1.0


def test_clamp_unit_clamps_below_zero():
    assert _clamp_unit(-0.4) == 0.0


def test_clamp_unit_clamps_above_one():
    assert _clamp_unit(1.7) == 1.0


def test_clamp_unit_handles_non_numeric():
    assert _clamp_unit("nope") == 0.0
    assert _clamp_unit(None) == 0.0


# ---------------------------------------------------------------------------
# Profile resolution + block formatting
# ---------------------------------------------------------------------------


def test_resolve_inner_profile_unwraps_full_dict():
    inner = _resolve_inner_profile(_full_person_vault())
    assert inner["communication_style"] == "defensive"
    assert "emotional_triggers" in inner


def test_resolve_inner_profile_accepts_inner_dict_directly():
    inner_only = _full_person_vault()["profile"]
    inner = _resolve_inner_profile(inner_only)
    assert inner["communication_style"] == "defensive"


def test_resolve_inner_profile_returns_empty_for_non_dict():
    assert _resolve_inner_profile(None) == {}
    assert _resolve_inner_profile("not a dict") == {}


def test_resolve_persona_name_uses_top_level_name():
    assert _resolve_persona_name(_full_person_vault()) == "Jamie"


def test_resolve_persona_name_falls_back_when_missing():
    assert _resolve_persona_name({}) == "Other"
    assert _resolve_persona_name(None) == "Other"


def test_resolve_relationship_block_uses_top_level_relationship():
    assert _resolve_relationship_block(_full_person_vault()) == "colleague"


def test_resolve_relationship_block_defaults_to_unspecified():
    assert _resolve_relationship_block({}) == "unspecified"
    assert _resolve_relationship_block(None) == "unspecified"


def test_format_profile_block_full_dict_includes_all_fields():
    text = format_persona_profile_block(_full_person_vault())
    assert "communication_style: defensive" in text
    assert "citing past commitments" in text
    assert "explicitly disowning blame" in text
    assert "I told you" in text
    assert "Concrete numbers" in text


def test_format_profile_block_accepts_inner_dict():
    inner_only = _full_person_vault()["profile"]
    text = format_persona_profile_block(inner_only)
    assert "communication_style: defensive" in text


def test_format_profile_block_renders_empty_lists_as_sentinel():
    profile = _full_person_vault(profile={"communication_style": "direct"})
    text = format_persona_profile_block(profile)
    # All four list fields should fall through to the "(none observed)"
    # sentinel rather than disappearing — the model gets a stable
    # structure each turn.
    assert text.count("(none observed)") >= 3


def test_format_profile_block_includes_cultural_context_when_present():
    profile = _full_person_vault()
    profile["profile"]["cultural_context"] = "Engineering-lead — values directness."
    text = format_persona_profile_block(profile)
    assert "cultural_context: Engineering-lead" in text


def test_format_profile_block_omits_cultural_context_when_absent():
    text = format_persona_profile_block(_full_person_vault())
    assert "cultural_context" not in text


def test_format_profile_block_returns_sentinel_for_empty():
    assert "(no PersonVault profile" in format_persona_profile_block(None)
    assert "(no PersonVault profile" in format_persona_profile_block({})


# ---------------------------------------------------------------------------
# build_practice_messages
# ---------------------------------------------------------------------------


def test_build_practice_messages_system_at_index_zero():
    msgs = build_practice_messages("SYSTEM", [], "hello")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYSTEM"


def test_build_practice_messages_appends_new_user_message_last():
    msgs = build_practice_messages("SYSTEM", [], "hello jamie")
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "hello jamie"


def test_build_practice_messages_renders_history_alternating():
    history = [
        {"speaker": "user", "text": "What about the report?"},
        {
            "speaker": "persona",
            "persona_name": "Jamie",
            "reply": "I told you on Tuesday.",
            "resistance_type": "deflect",
            "escalation_level": 0.5,
        },
    ]
    msgs = build_practice_messages("SYSTEM", history, "Right, but Wednesday EOD?")
    # Expected shape: system, user, assistant, user
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_build_practice_messages_assistant_content_is_valid_json():
    """The assistant content fed back to the chat template is JSON-
    serialised so the model sees its own structured trajectory."""
    history = [
        {"speaker": "user", "text": "What about the report?"},
        {
            "speaker": "persona",
            "persona_name": "Jamie",
            "reply": "I told you on Tuesday.",
            "resistance_type": "deflect",
            "escalation_level": 0.5,
        },
    ]
    msgs = build_practice_messages("SYSTEM", history, "x")
    assistant_msg = msgs[2]
    parsed = json.loads(assistant_msg["content"])
    assert parsed["persona_name"] == "Jamie"
    assert parsed["reply"] == "I told you on Tuesday."
    assert parsed["resistance_type"] == "deflect"
    assert parsed["escalation_level"] == 0.5


def test_build_practice_messages_skips_unknown_speakers():
    """A malformed history entry shouldn't corrupt chat-template alignment."""
    history = [
        {"speaker": "ghost", "text": "boo"},
        {"speaker": "user", "text": "real turn"},
    ]
    msgs = build_practice_messages("SYSTEM", history, "x")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "user"]


# ---------------------------------------------------------------------------
# _coerce_persona_reply_payload: happy path
# ---------------------------------------------------------------------------


def test_coerce_returns_required_top_level_fields():
    result = _coerce_persona_reply_payload(
        _minimal_llm_payload(),
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    for key in ("persona_name", "reply", "resistance_type", "escalation_level"):
        assert key in result, f"missing required key: {key}"


def test_coerce_attaches_turn_id_and_timestamp():
    result = _coerce_persona_reply_payload(
        _minimal_llm_payload(),
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert result["turn_id"].startswith("persona_turn_")
    assert "T" in result["timestamp"]  # ISO 8601 marker


def test_coerce_forces_persona_name_from_cfg():
    """The model's emitted persona_name is ignored — runtime forces
    the configured name. Same defence-in-depth pattern as Step 04's
    ``whisper_prompt=None for other`` rule."""
    payload = _minimal_llm_payload(persona_name="SomeoneElse")
    result = _coerce_persona_reply_payload(
        payload,
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert result["persona_name"] == "Jamie"


def test_coerce_falls_back_to_default_persona_name_when_cfg_empty():
    result = _coerce_persona_reply_payload(
        _minimal_llm_payload(persona_name="Anything"),
        cfg=PersonaSimConfig(persona_profile={}),
    )
    assert result["persona_name"] == "Other"


def test_coerce_normalises_resistance_synonyms():
    payload = _minimal_llm_payload(resistance_type="counterattack")
    result = _coerce_persona_reply_payload(
        payload,
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert result["resistance_type"] == "counter_attack"


def test_coerce_normalises_resistance_casing():
    payload = _minimal_llm_payload(resistance_type="Counter_Attack")
    result = _coerce_persona_reply_payload(
        payload,
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert result["resistance_type"] == "counter_attack"


def test_coerce_clamps_escalation_level_above_one():
    payload = _minimal_llm_payload(escalation_level=1.5)
    result = _coerce_persona_reply_payload(
        payload,
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert result["escalation_level"] == 1.0


def test_coerce_clamps_escalation_level_below_zero():
    payload = _minimal_llm_payload(escalation_level=-0.3)
    result = _coerce_persona_reply_payload(
        payload,
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert result["escalation_level"] == 0.0


def test_coerce_trims_overly_long_reply():
    long_reply = "x " * 600  # 1200 chars
    payload = _minimal_llm_payload(reply=long_reply)
    result = _coerce_persona_reply_payload(
        payload,
        cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
    )
    assert len(result["reply"]) <= 800


# ---------------------------------------------------------------------------
# _coerce_persona_reply_payload: error paths
# ---------------------------------------------------------------------------


def test_coerce_rejects_non_dict_payload():
    with pytest.raises(PersonaReplyError):
        _coerce_persona_reply_payload(
            "not a dict",
            cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
        )


def test_coerce_rejects_missing_reply():
    payload = _minimal_llm_payload()
    del payload["reply"]
    with pytest.raises(PersonaReplyError):
        _coerce_persona_reply_payload(
            payload,
            cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
        )


def test_coerce_rejects_empty_reply():
    payload = _minimal_llm_payload(reply="   ")
    with pytest.raises(PersonaReplyError):
        _coerce_persona_reply_payload(
            payload,
            cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
        )


def test_coerce_rejects_unknown_resistance_type():
    payload = _minimal_llm_payload(resistance_type="bargaining")
    with pytest.raises(PersonaReplyError):
        _coerce_persona_reply_payload(
            payload,
            cfg=PersonaSimConfig(persona_profile=_full_person_vault()),
        )


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_persona_reply_schema_required_fields_match_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "PersonaReply"
    for field in ("persona_name", "reply", "resistance_type", "escalation_level"):
        assert field in schema["required"], f"{field!r} must be required"


def test_allowed_resistance_types_match_schema_enum():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(schema["properties"]["resistance_type"]["enum"])
    assert set(ALLOWED_RESISTANCE_TYPES) == enum


def test_persona_prompt_has_all_placeholders():
    """The default prompt file must exist and accept exactly the four
    placeholders the runtime renders. Drift breaks ``Prompt.render``."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "$persona_name",
        "$relationship_block",
        "$profile_block",
        "$user_goal",
    ):
        assert placeholder in text, f"prompt missing {placeholder}"


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Mirror the rationale from talk_dna / person_vault / emotion: just
    enough randomness to escape a single bad greedy trajectory."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1
