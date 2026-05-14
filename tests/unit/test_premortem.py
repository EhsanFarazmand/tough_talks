"""Unit tests for the Pre-Mortem Generator runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - ``_coerce_premortem_payload`` contract: scenario count, scenario_id
    forced to 1/2/3, resistance synonym normalisation, numeric
    clamping, required string fields rejected when empty
  - ``_coerce_scenario`` and ``_coerce_simulation_parameters`` raise
    on each missing-required-field shape
  - ``format_talk_dna_block`` accepts the full Step 05 dict, partial
    profiles, and empty input
  - ``build_premortem_messages`` shape (single user message at index 0)
  - schema-vs-runtime drift checks: ``REQUIRED_SCENARIO_COUNT`` matches
    the schema's ``minItems``/``maxItems``, the schema's
    ``resistance_type`` enum matches :data:`ALLOWED_RESISTANCE_TYPES`
    from the persona-sim runtime (single source of truth across
    Steps 07 and 08)
  - the default prompt file declares exactly the four placeholders
    the runtime renders

The real ``generate_premortem`` call requires a loaded model and is
exercised in ``notebooks/phase3/step08_premortem.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.persona_sim import ALLOWED_RESISTANCE_TYPES
from backend.core._runtime.premortem import (
    DEFAULT_PREMORTEM_PROMPT_NAME,
    REQUIRED_SCENARIO_COUNT,
    RETRY_SAMPLING,
    PremortemConfig,
    PremortemError,
    _clamp_unit,
    _coerce_premortem_payload,
    _coerce_scenario,
    _coerce_simulation_parameters,
    build_premortem_messages,
    format_talk_dna_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "premortem.schema.json"
PROMPT_PATH = (
    REPO_ROOT / "data" / "prompts" / f"{DEFAULT_PREMORTEM_PROMPT_NAME}.md"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_sim_params(**overrides) -> dict:
    base = {
        "resistance_type": "deflect",
        "escalation_ceiling": 0.7,
        "opening_move": "I told you on Tuesday — staging tables aren't done.",
    }
    base.update(overrides)
    return base


def _minimal_scenario(**overrides) -> dict:
    base = {
        "scenario_id": 1,
        "title": "Deflection via past commitments",
        "description": (
            "Jamie cites the missed escalation thread and reframes the "
            "conversation as a communication failure on the user's side."
        ),
        "likely_trigger": "User mentions the missed Wednesday deadline directly.",
        "destabilization_risk": 0.55,
        "simulation_parameters": _minimal_sim_params(),
    }
    base.update(overrides)
    return base


def _full_premortem_payload(**overrides) -> dict:
    base = {
        "goal": "Get Jamie to commit to a clear escalation channel.",
        "failure_scenarios": [
            _minimal_scenario(scenario_id=1),
            _minimal_scenario(
                scenario_id=2,
                title="Counter-attack on the user's record",
                description="Jamie pivots to past missed deadlines on the user's side.",
                likely_trigger="User asks for a firm commitment without offering support.",
                destabilization_risk=0.75,
                simulation_parameters=_minimal_sim_params(
                    resistance_type="counterattack",
                    escalation_ceiling=0.85,
                    opening_move="You weren't there when I flagged this — don't lecture me now.",
                ),
            ),
            _minimal_scenario(
                scenario_id=3,
                title="Stonewall withdrawal",
                description="Jamie shuts down and refuses to engage when pressed.",
                likely_trigger="Repeated questions after a deflection.",
                destabilization_risk=0.4,
                simulation_parameters=_minimal_sim_params(
                    resistance_type="silence",
                    escalation_ceiling=0.45,
                    opening_move="Fine.",
                ),
            ),
        ],
    }
    base.update(overrides)
    return base


def _talk_dna_v2(**overrides) -> dict:
    """Representative TalkDNA shape from Step 05's v2 output."""
    base = {
        "user_id": "local",
        "version": 2,
        "conversation_count": 2,
        "patterns": {
            "filler_phrases": ["I just feel like", "kind of"],
            "apology_rate": 0.57,
            "silence_under_pressure": True,
            "sarcasm_frequency": "low",
            "escalation_triggers": [
                "feeling dismissed",
                "being told to wait",
            ],
            "avg_turn_length_words": 15.7,
        },
        "strengths": ["clear_problem_statement", "acknowledges_mistakes"],
        "weaknesses": ["over_apologizes", "hedges_before_vulnerable_statements"],
        "updated_at": "2026-05-14T00:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# format_talk_dna_block
# ---------------------------------------------------------------------------


def test_format_talk_dna_block_renders_full_profile():
    text = format_talk_dna_block(_talk_dna_v2())
    assert "over_apologizes" in text
    assert "clear_problem_statement" in text
    assert "feeling dismissed" in text
    assert "0.57" in text
    assert "true" in text
    assert "low" in text


def test_format_talk_dna_block_returns_sentinel_for_empty():
    assert "(no Talk DNA profile" in format_talk_dna_block(None)
    assert "(no Talk DNA profile" in format_talk_dna_block({})


def test_format_talk_dna_block_handles_missing_patterns_block():
    profile = {
        "user_id": "local",
        "version": 1,
        "strengths": ["listens_actively"],
        "weaknesses": [],
    }
    text = format_talk_dna_block(profile)
    assert "listens_actively" in text
    assert "(none observed)" in text  # weaknesses + escalation_triggers fall through
    assert "(unknown)" in text  # apology_rate, sarcasm absent


def test_format_talk_dna_block_handles_missing_pattern_fields():
    profile = {
        "patterns": {"sarcasm_frequency": "moderate"},
        "strengths": [],
        "weaknesses": [],
    }
    text = format_talk_dna_block(profile)
    assert "moderate" in text
    assert "(unknown)" in text  # apology_rate, silence absent


# ---------------------------------------------------------------------------
# build_premortem_messages
# ---------------------------------------------------------------------------


def test_build_premortem_messages_single_user_turn():
    msgs = build_premortem_messages("INSTRUCTION")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "INSTRUCTION"


# ---------------------------------------------------------------------------
# Numeric helper
# ---------------------------------------------------------------------------


def test_clamp_unit_clamps_to_unit_range():
    assert _clamp_unit(-0.5) == 0.0
    assert _clamp_unit(0.4) == 0.4
    assert _clamp_unit(1.7) == 1.0
    assert _clamp_unit("nonsense") == 0.0
    assert _clamp_unit(None) == 0.0


# ---------------------------------------------------------------------------
# _coerce_simulation_parameters
# ---------------------------------------------------------------------------


def test_coerce_sim_params_happy_path():
    result = _coerce_simulation_parameters(_minimal_sim_params(), scenario_id=1)
    assert result["resistance_type"] == "deflect"
    assert result["escalation_ceiling"] == 0.7
    assert result["opening_move"].startswith("I told you")


def test_coerce_sim_params_normalises_resistance_synonym():
    raw = _minimal_sim_params(resistance_type="counterattack")
    result = _coerce_simulation_parameters(raw, scenario_id=2)
    assert result["resistance_type"] == "counter_attack"


def test_coerce_sim_params_clamps_escalation_ceiling():
    raw = _minimal_sim_params(escalation_ceiling=1.4)
    result = _coerce_simulation_parameters(raw, scenario_id=1)
    assert result["escalation_ceiling"] == 1.0


def test_coerce_sim_params_rejects_unknown_resistance():
    raw = _minimal_sim_params(resistance_type="bargaining")
    with pytest.raises(PremortemError, match="resistance_type"):
        _coerce_simulation_parameters(raw, scenario_id=1)


def test_coerce_sim_params_rejects_empty_opening_move():
    raw = _minimal_sim_params(opening_move="   ")
    with pytest.raises(PremortemError, match="opening_move"):
        _coerce_simulation_parameters(raw, scenario_id=2)


def test_coerce_sim_params_rejects_non_dict():
    with pytest.raises(PremortemError, match="simulation_parameters"):
        _coerce_simulation_parameters("not a dict", scenario_id=1)


# ---------------------------------------------------------------------------
# _coerce_scenario
# ---------------------------------------------------------------------------


def test_coerce_scenario_happy_path():
    result = _coerce_scenario(_minimal_scenario(), scenario_id=1)
    assert result["scenario_id"] == 1
    assert result["title"]
    assert result["description"]
    assert result["likely_trigger"]
    assert 0.0 <= result["destabilization_risk"] <= 1.0
    assert "resistance_type" in result["simulation_parameters"]


def test_coerce_scenario_forces_scenario_id_from_arg():
    """Even if the model emits scenario_id=99, the runtime forces the
    canonical value passed in by the caller. Same defence-in-depth as
    persona-sim's persona_name forcing."""
    raw = _minimal_scenario(scenario_id=99)
    result = _coerce_scenario(raw, scenario_id=2)
    assert result["scenario_id"] == 2


def test_coerce_scenario_rejects_empty_title():
    raw = _minimal_scenario(title="   ")
    with pytest.raises(PremortemError, match="title"):
        _coerce_scenario(raw, scenario_id=1)


def test_coerce_scenario_rejects_empty_description():
    raw = _minimal_scenario(description="")
    with pytest.raises(PremortemError, match="description"):
        _coerce_scenario(raw, scenario_id=1)


def test_coerce_scenario_rejects_empty_likely_trigger():
    raw = _minimal_scenario(likely_trigger=None)
    with pytest.raises(PremortemError, match="likely_trigger"):
        _coerce_scenario(raw, scenario_id=1)


def test_coerce_scenario_clamps_destabilization_risk():
    raw = _minimal_scenario(destabilization_risk=2.0)
    result = _coerce_scenario(raw, scenario_id=1)
    assert result["destabilization_risk"] == 1.0


def test_coerce_scenario_rejects_non_dict():
    with pytest.raises(PremortemError, match="payload not an object"):
        _coerce_scenario(["not a dict"], scenario_id=1)


# ---------------------------------------------------------------------------
# _coerce_premortem_payload
# ---------------------------------------------------------------------------


def test_coerce_payload_happy_path():
    cfg = PremortemConfig(conversation_description="x", user_goal="ship safely")
    result = _coerce_premortem_payload(_full_premortem_payload(), cfg=cfg)
    assert result["goal"]
    assert len(result["failure_scenarios"]) == REQUIRED_SCENARIO_COUNT
    # scenario_ids should be 1, 2, 3 in order
    assert [s["scenario_id"] for s in result["failure_scenarios"]] == [1, 2, 3]
    assert result["premortem_id"].startswith("premortem_")
    assert "T" in result["generated_at"]  # ISO 8601 marker


def test_coerce_payload_renumbers_scenarios_in_order():
    """Even if the model emits scenarios out of order or with bad ids,
    the runtime forces 1, 2, 3 in list order."""
    payload = _full_premortem_payload()
    payload["failure_scenarios"][0]["scenario_id"] = 7
    payload["failure_scenarios"][1]["scenario_id"] = 99
    payload["failure_scenarios"][2]["scenario_id"] = -1
    cfg = PremortemConfig(conversation_description="x")
    result = _coerce_premortem_payload(payload, cfg=cfg)
    assert [s["scenario_id"] for s in result["failure_scenarios"]] == [1, 2, 3]


def test_coerce_payload_falls_back_to_cfg_user_goal_when_empty():
    payload = _full_premortem_payload(goal="")
    cfg = PremortemConfig(
        conversation_description="x", user_goal="recover the relationship"
    )
    result = _coerce_premortem_payload(payload, cfg=cfg)
    assert result["goal"] == "recover the relationship"


def test_coerce_payload_rejects_empty_goal_and_user_goal():
    payload = _full_premortem_payload(goal=None)
    cfg = PremortemConfig(conversation_description="x", user_goal="")
    with pytest.raises(PremortemError, match="goal"):
        _coerce_premortem_payload(payload, cfg=cfg)


def test_coerce_payload_rejects_wrong_scenario_count_too_few():
    payload = _full_premortem_payload()
    payload["failure_scenarios"] = payload["failure_scenarios"][:2]
    cfg = PremortemConfig(conversation_description="x")
    with pytest.raises(PremortemError, match="exactly 3"):
        _coerce_premortem_payload(payload, cfg=cfg)


def test_coerce_payload_rejects_wrong_scenario_count_too_many():
    payload = _full_premortem_payload()
    payload["failure_scenarios"].append(_minimal_scenario(scenario_id=4))
    cfg = PremortemConfig(conversation_description="x")
    with pytest.raises(PremortemError, match="exactly 3"):
        _coerce_premortem_payload(payload, cfg=cfg)


def test_coerce_payload_rejects_failure_scenarios_not_list():
    payload = _full_premortem_payload()
    payload["failure_scenarios"] = {"oops": "dict"}
    cfg = PremortemConfig(conversation_description="x")
    with pytest.raises(PremortemError, match="must be a list"):
        _coerce_premortem_payload(payload, cfg=cfg)


def test_coerce_payload_rejects_non_dict_top_level():
    cfg = PremortemConfig(conversation_description="x")
    with pytest.raises(PremortemError, match="not an object"):
        _coerce_premortem_payload(["not", "a", "dict"], cfg=cfg)


def test_coerce_payload_normalises_resistance_synonyms_in_each_scenario():
    """The synonym normaliser fires per-scenario — confirms it's applied
    inside the loop, not just on the first scenario."""
    payload = _full_premortem_payload()
    # scenario 2 already uses 'counterattack', scenario 3 uses 'silence'
    cfg = PremortemConfig(conversation_description="x")
    result = _coerce_premortem_payload(payload, cfg=cfg)
    rts = [
        s["simulation_parameters"]["resistance_type"]
        for s in result["failure_scenarios"]
    ]
    assert rts == ["deflect", "counter_attack", "silent"]


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_premortem_schema_required_scenario_count_matches_runtime():
    """``REQUIRED_SCENARIO_COUNT`` must match the schema's hard-pinned
    ``minItems`` / ``maxItems``. Drift breaks ``_coerce_premortem_payload``
    and the schema validation in the notebook results table."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fs = schema["properties"]["failure_scenarios"]
    assert fs["minItems"] == REQUIRED_SCENARIO_COUNT
    assert fs["maxItems"] == REQUIRED_SCENARIO_COUNT


def test_premortem_schema_resistance_enum_matches_persona_sim_runtime():
    """The premortem schema and the persona-sim runtime share the same
    closed enum — a scenario's ``simulation_parameters.resistance_type``
    is meant to be runnable as a ``PersonaSimConfig`` input. Drift here
    means a generated pre-mortem could yield a scenario that the persona
    simulator silently rejects."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    sim_params = schema["properties"]["failure_scenarios"]["items"][
        "properties"
    ]["simulation_parameters"]
    enum = set(sim_params["properties"]["resistance_type"]["enum"])
    assert enum == set(ALLOWED_RESISTANCE_TYPES)


def test_premortem_schema_required_top_level_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "PreMortem"
    for field in ("goal", "failure_scenarios"):
        assert field in schema["required"]


def test_premortem_prompt_has_all_placeholders():
    """The default prompt file must exist and accept exactly the four
    placeholders the runtime renders. Drift breaks ``Prompt.render``."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "$conversation_description",
        "$user_goal",
        "$person_profile_block",
        "$talk_dna_block",
    ):
        assert placeholder in text, f"prompt missing {placeholder}"


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Mirror the rationale from the other intelligence runtimes — just
    enough randomness to escape a single bad greedy trajectory."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1
