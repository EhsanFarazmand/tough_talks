"""Unit tests for the Aftermath (plan-vs-reality) runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - ``_coerce_aftermath_payload`` contract: required top-level fields,
    pre-mortem-driven enforcement of scenario_id / title /
    predicted_resistance_type, derivation of ``materialized`` from
    ``match_quality``, evidence consistency, count clamping
  - per-section coercers (``_coerce_goal_outcome``,
    ``_coerce_scenario_outcome``, ``_coerce_unforeseen_moments``) drop
    or fix malformed entries
  - enum normalisers (``_normalize_match_quality``,
    ``_normalize_unforeseen_kind``, ``_normalize_goal_status``) accept
    documented synonyms
  - ``format_premortem_block`` and ``format_debrief_block`` rendering:
    sentinels on empty input, all scenario fields surfaced
  - ``build_aftermath_messages`` shape (single user message at index 0)
  - schema-vs-runtime contract: the schema's enums match the runtime
    tuples; the default prompt declares the five placeholders the
    runtime renders

The real ``generate_aftermath`` call requires a loaded model and is
exercised in ``notebooks/phase4/step10_aftermath.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.aftermath import (
    ALLOWED_GOAL_STATUSES,
    ALLOWED_MATCH_QUALITIES,
    ALLOWED_UNFORESEEN_KINDS,
    AftermathConfig,
    AftermathError,
    DEFAULT_AFTERMATH_PROMPT_NAME,
    RETRY_SAMPLING,
    _coerce_aftermath_payload,
    _coerce_goal_outcome,
    _coerce_scenario_outcome,
    _coerce_unforeseen_moments,
    _extract_premortem_scenarios,
    _normalize_goal_status,
    _normalize_match_quality,
    _normalize_unforeseen_kind,
    _scenario_predicted_resistance,
    _scenario_title,
    _strip_transcript_meta_prefix,
    build_aftermath_messages,
    format_debrief_block,
    format_premortem_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "aftermath.schema.json"
PROMPT_PATH = (
    REPO_ROOT / "data" / "prompts" / f"{DEFAULT_AFTERMATH_PROMPT_NAME}.md"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _three_scenarios() -> list[dict]:
    """Pre-mortem-shaped input: three scenarios with distinct resistance
    types. Matches the shape ``generate_premortem()`` emits, including
    nested ``simulation_parameters``."""
    return [
        {
            "scenario_id": 1,
            "title": "The 'I told you' blame redirect",
            "description": "Jamie redirects to past communications.",
            "likely_trigger": "User cites the missed Tuesday deadline directly",
            "destabilization_risk": 0.55,
            "simulation_parameters": {
                "resistance_type": "deflect",
                "escalation_ceiling": 0.65,
                "opening_move": "I told you the staging tables weren't done.",
            },
        },
        {
            "scenario_id": 2,
            "title": "Counter-attack on the new deadline",
            "description": "Jamie pushes back on Wednesday EOD as overreach.",
            "likely_trigger": "User sets a hard date without naming the blockers",
            "destabilization_risk": 0.75,
            "simulation_parameters": {
                "resistance_type": "counter_attack",
                "escalation_ceiling": 0.85,
                "opening_move": "Wednesday EOD is unrealistic — name the blockers first.",
            },
        },
        {
            "scenario_id": 3,
            "title": "'Don't put this on me' guilt-trip",
            "description": "Jamie flips the missed-escalation point into a complaint.",
            "likely_trigger": "User implies the escalation failure is one-sided",
            "destabilization_risk": 0.60,
            "simulation_parameters": {
                "resistance_type": "guilt_trip",
                "escalation_ceiling": 0.70,
                "opening_move": "Don't put this on me. I escalated twice last week.",
            },
        },
    ]


def _full_premortem() -> dict:
    return {
        "goal": "Get Jamie to commit to Wednesday EOD.",
        "failure_scenarios": _three_scenarios(),
        "premortem_id": "premortem_demo_8a1b2c3d4e5f",
        "generated_at": "2026-05-14T15:00:00+00:00",
    }


def _full_payload(**overrides) -> dict:
    """A complete, valid model-emitted aftermath payload."""
    base = {
        "goal_outcome": {
            "status": "partial",
            "summary": (
                "You got the Wednesday EOD commitment but Jamie didn't fully "
                "acknowledge the escalation problem."
            ),
        },
        "scenario_outcomes": [
            {
                "scenario_id": 1,
                "title": "The 'I told you' blame redirect",
                "predicted_resistance_type": "deflect",
                "match_quality": "direct_hit",
                "materialized": True,
                "evidence": "Jamie opened with 'I told you we were short on time' at her first turn.",
                "notes": "S1 landed exactly as predicted — Jamie ran the 'I told you' play on turn one.",
            },
            {
                "scenario_id": 2,
                "title": "Counter-attack on the new deadline",
                "predicted_resistance_type": "counter_attack",
                "match_quality": "partial",
                "materialized": True,
                "evidence": "Jamie demanded the user spell out blockers before committing.",
                "notes": "S2 partly landed — softer than the full counter-attack we predicted.",
            },
            {
                "scenario_id": 3,
                "title": "'Don't put this on me' guilt-trip",
                "predicted_resistance_type": "guilt_trip",
                "match_quality": "did_not_occur",
                "materialized": False,
                "evidence": "",
                "notes": "Jamie used 'don't put this on me' but you reframed and she conceded.",
            },
        ],
        "unforeseen_moments": [
            {
                "kind": "opportunity",
                "turn": 4,
                "description": "Daily Slack check-in landed as a genuine concession Jamie accepted.",
            }
        ],
        "prediction_accuracy": 0.6,
        "next_round_focus": "Practice handling the soft 'manage expectations' pushback.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Enum normalisers
# ---------------------------------------------------------------------------


def test_normalize_match_quality_accepts_canonical_values():
    for value in ALLOWED_MATCH_QUALITIES:
        assert _normalize_match_quality(value) == value


def test_normalize_match_quality_normalises_casing_and_separator():
    assert _normalize_match_quality("Direct_Hit") == "direct_hit"
    assert _normalize_match_quality("direct-hit") == "direct_hit"
    assert _normalize_match_quality("DIRECT HIT") == "direct_hit"
    assert _normalize_match_quality("did not occur") == "did_not_occur"


def test_normalize_match_quality_accepts_documented_synonyms():
    """Model frequently reaches for words near its training distribution.
    Defence-in-depth: accept the obvious synonyms and map them back to
    canonical enum codes; truly unknown labels return ``None`` so the
    retry path fires."""
    cases = {
        "exact": "direct_hit",
        "exact match": "direct_hit",
        "matched": "direct_hit",
        "hit": "direct_hit",
        "partly": "partial",
        "mixed": "partial",
        "miss": "did_not_occur",
        "missed": "did_not_occur",
        "no match": "did_not_occur",
        "did not happen": "did_not_occur",
        "did_not_materialize": "did_not_occur",
        "not materialised": "did_not_occur",
    }
    for raw, expected in cases.items():
        assert _normalize_match_quality(raw) == expected, raw


def test_normalize_match_quality_returns_none_for_unknown():
    assert _normalize_match_quality("ambiguous") is None
    assert _normalize_match_quality(None) is None
    assert _normalize_match_quality("") is None
    assert _normalize_match_quality(42) is None


def test_normalize_unforeseen_kind_accepts_synonyms():
    assert _normalize_unforeseen_kind("risk") == "risk"
    assert _normalize_unforeseen_kind("Opportunity") == "opportunity"
    assert _normalize_unforeseen_kind("threat") == "risk"
    assert _normalize_unforeseen_kind("opening") == "opportunity"
    assert _normalize_unforeseen_kind("unknown") is None


def test_normalize_goal_status_accepts_synonyms():
    assert _normalize_goal_status("achieved") == "achieved"
    assert _normalize_goal_status("success") == "achieved"
    assert _normalize_goal_status("partly") == "partial"
    assert _normalize_goal_status("failed") == "not_achieved"
    assert _normalize_goal_status("did not achieve") == "not_achieved"
    assert _normalize_goal_status("unknown") is None


# ---------------------------------------------------------------------------
# Pre-mortem scenario extraction
# ---------------------------------------------------------------------------


def test_extract_premortem_scenarios_handles_dict_and_list():
    full = _full_premortem()
    assert _extract_premortem_scenarios(full) == _three_scenarios()
    # Bare list is also accepted (for callers that pre-extract).
    assert _extract_premortem_scenarios(_three_scenarios()) == _three_scenarios()


def test_extract_premortem_scenarios_returns_empty_on_garbage():
    assert _extract_premortem_scenarios({}) == []
    assert _extract_premortem_scenarios({"failure_scenarios": "not a list"}) == []
    assert _extract_premortem_scenarios(None) == []
    assert _extract_premortem_scenarios("not a dict") == []


def test_extract_premortem_scenarios_filters_non_dict_entries():
    raw = [{"scenario_id": 1, "title": "a"}, "garbage", None]
    assert _extract_premortem_scenarios(raw) == [{"scenario_id": 1, "title": "a"}]


def test_scenario_predicted_resistance_pulls_nested_value():
    scenario = _three_scenarios()[1]
    assert _scenario_predicted_resistance(scenario) == "counter_attack"


def test_scenario_predicted_resistance_normalises_synonyms():
    """Defence-in-depth: even though premortem.py already normalises
    resistance_type before persisting, the aftermath runtime
    re-normalises so a hand-edited or legacy premortem dict can still
    flow through correctly."""
    scenario = {"simulation_parameters": {"resistance_type": "denial"}}
    assert _scenario_predicted_resistance(scenario) == "deny"
    scenario = {"simulation_parameters": {"resistance_type": "Counter-Attack"}}
    assert _scenario_predicted_resistance(scenario) == "counter_attack"


def test_scenario_predicted_resistance_falls_back_to_flat_field():
    """Flatter shapes (e.g. tests, ad-hoc callers) without
    ``simulation_parameters`` should still resolve."""
    assert _scenario_predicted_resistance({"resistance_type": "deflect"}) == "deflect"


def test_scenario_title_falls_back_when_missing():
    assert _scenario_title({}, fallback_id=2) == "Scenario 2"
    assert _scenario_title({"title": "   "}, fallback_id=3) == "Scenario 3"


# ---------------------------------------------------------------------------
# format_premortem_block / format_debrief_block
# ---------------------------------------------------------------------------


def test_format_premortem_block_renders_all_scenarios_in_order():
    text = format_premortem_block(_full_premortem())
    for i in range(1, 4):
        assert f"S{i}." in text
    assert "deflect" in text
    assert "counter_attack" in text
    assert "guilt_trip" in text
    # opening_move quoted dialogue should survive into the block.
    assert "I told you the staging tables weren't done." in text


def test_format_premortem_block_returns_sentinel_when_empty():
    assert "no pre-mortem" in format_premortem_block({})
    assert "no pre-mortem" in format_premortem_block(None)
    assert "no pre-mortem" in format_premortem_block({"failure_scenarios": []})


def test_format_debrief_block_renders_headline_counts():
    debrief = {
        "ground_lost": [{}, {}],
        "over_apologies": [],
        "missed_openings": [{}],
        "wins": [{}, {}, {}],
        "one_fix_next_time": "Slow down before responding to deflections.",
    }
    text = format_debrief_block(debrief)
    assert "one_fix_next_time" in text
    assert "Slow down before responding to deflections." in text
    # Bare-count rendering for each array.
    assert "ground_lost: 2" in text
    assert "over_apologies: 0" in text
    assert "missed_openings: 1" in text
    assert "wins: 3" in text


def test_format_debrief_block_returns_sentinel_on_empty():
    assert "no post-round debrief" in format_debrief_block({})
    assert "no post-round debrief" in format_debrief_block(None)


# ---------------------------------------------------------------------------
# build_aftermath_messages
# ---------------------------------------------------------------------------


def test_build_aftermath_messages_single_user_turn():
    msgs = build_aftermath_messages("INSTRUCTION")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "INSTRUCTION"


# ---------------------------------------------------------------------------
# _coerce_goal_outcome
# ---------------------------------------------------------------------------


def test_coerce_goal_outcome_happy_path():
    raw = {"status": "achieved", "summary": "You got what you came for."}
    result = _coerce_goal_outcome(raw)
    assert result == {"status": "achieved", "summary": "You got what you came for."}


def test_coerce_goal_outcome_normalises_status_synonym():
    raw = {"status": "Success", "summary": "All good."}
    assert _coerce_goal_outcome(raw)["status"] == "achieved"


def test_coerce_goal_outcome_rejects_empty_summary():
    with pytest.raises(AftermathError, match="summary"):
        _coerce_goal_outcome({"status": "partial", "summary": "   "})


def test_coerce_goal_outcome_rejects_unknown_status():
    with pytest.raises(AftermathError, match="status"):
        _coerce_goal_outcome({"status": "unknown_status", "summary": "Hi."})


def test_coerce_goal_outcome_rejects_non_dict():
    with pytest.raises(AftermathError, match="object"):
        _coerce_goal_outcome("not a dict")


# ---------------------------------------------------------------------------
# _coerce_scenario_outcome
# ---------------------------------------------------------------------------


def test_coerce_scenario_outcome_forces_title_and_resistance_from_input():
    """Defence-in-depth: title and predicted_resistance_type come from
    the premortem input, not the model. The model could paraphrase the
    title or drop the snake_case enum — we don't trust either."""
    scenario_input = _three_scenarios()[0]
    raw = {
        "scenario_id": 99,  # wrong; runtime forces 1
        "title": "Some paraphrased title the model made up",  # wrong
        "predicted_resistance_type": "denial",  # wrong AND a near-synonym
        "match_quality": "direct_hit",
        "materialized": True,
        "evidence": "Jamie opened with the predicted deflection.",
        "notes": "Direct hit.",
    }
    out = _coerce_scenario_outcome(raw, scenario_input=scenario_input, scenario_id=1)
    assert out["scenario_id"] == 1
    assert out["title"] == "The 'I told you' blame redirect"
    assert out["predicted_resistance_type"] == "deflect"
    assert out["match_quality"] == "direct_hit"
    assert out["materialized"] is True
    assert out["evidence"].startswith("Jamie opened")


def test_coerce_scenario_outcome_derives_materialized_from_match_quality():
    """Model's boolean is discarded — match_quality is the source of
    truth. ``direct_hit`` / ``partial`` → True, ``did_not_occur`` →
    False, regardless of what the model emitted."""
    scenario_input = _three_scenarios()[0]
    # Model emits inconsistent True for a did_not_occur:
    raw = {
        "match_quality": "did_not_occur",
        "materialized": True,  # wrong
        "evidence": "any string",  # should be wiped
        "notes": "Did not occur.",
    }
    out = _coerce_scenario_outcome(raw, scenario_input=scenario_input, scenario_id=1)
    assert out["materialized"] is False
    assert out["evidence"] == ""

    # And the inverse: model emits inconsistent False for a direct_hit.
    raw2 = {
        "match_quality": "direct_hit",
        "materialized": False,  # wrong
        "evidence": "Jamie ran the play exactly.",
        "notes": "Direct hit.",
    }
    out2 = _coerce_scenario_outcome(raw2, scenario_input=scenario_input, scenario_id=1)
    assert out2["materialized"] is True


def test_coerce_scenario_outcome_did_not_occur_forces_empty_evidence():
    """The prompt forbids invented evidence for a missed prediction.
    Defence-in-depth: enforce in code even when the model leaks a
    non-empty evidence string into a did_not_occur entry."""
    scenario_input = _three_scenarios()[0]
    raw = {
        "match_quality": "did_not_occur",
        "materialized": False,
        "evidence": "Made-up evidence for a missed prediction.",
        "notes": "The scenario did not occur.",
    }
    out = _coerce_scenario_outcome(raw, scenario_input=scenario_input, scenario_id=1)
    assert out["evidence"] == ""


def test_coerce_scenario_outcome_downgrades_empty_evidence_to_did_not_occur():
    """A direct_hit or partial with empty evidence is internally
    inconsistent — there's no evidence to ground the claim. Downgrade
    to did_not_occur silently so the output stays valid; the retry
    path will already have run if this leaked through both attempts."""
    scenario_input = _three_scenarios()[0]
    raw = {
        "match_quality": "direct_hit",
        "materialized": True,
        "evidence": "   ",  # empty after strip
        "notes": "Claimed a direct hit with nothing to back it up.",
    }
    out = _coerce_scenario_outcome(raw, scenario_input=scenario_input, scenario_id=1)
    assert out["match_quality"] == "did_not_occur"
    assert out["materialized"] is False
    assert out["evidence"] == ""


# ---------------------------------------------------------------------------
# Transcript-meta-prefix strip
# ---------------------------------------------------------------------------


def test_strip_transcript_meta_prefix_removes_persona_bracket():
    """Step 10 Run 1: both ``enable_thinking`` branches pasted the
    transcript's render-format bracket into ``evidence``. Defence-in-
    depth strip removes the prefix on coercion."""
    raw = (
        "Jamie (resistance=deflect, escalation=0.60): "
        "I told you we were short on time."
    )
    assert _strip_transcript_meta_prefix(raw) == "I told you we were short on time."


def test_strip_transcript_meta_prefix_handles_outer_bracket_form():
    """The runtime renders persona turns as
    ``[Jamie (resistance=deflect, escalation=0.60)]`` — the model
    occasionally copies the brackets along with the metadata."""
    raw = (
        "[Jamie (resistance=guilt_trip, escalation=0.70)] "
        "Don't put this on me."
    )
    assert _strip_transcript_meta_prefix(raw) == "Don't put this on me."


def test_strip_transcript_meta_prefix_handles_no_colon_no_outer_bracket():
    """The two real Step 10 outputs differed in whether they kept the
    outer ``[...]`` and whether they added a trailing colon. Both
    shapes must strip."""
    raw = (
        "Jamie (resistance=counter_attack, escalation=0.75) "
        "Wednesday EOD is tight."
    )
    assert _strip_transcript_meta_prefix(raw) == "Wednesday EOD is tight."


def test_strip_transcript_meta_prefix_handles_persona_only_resistance():
    """``format_practice_transcript`` drops ``escalation=...`` when the
    turn has no numeric escalation level. The regex still recognises
    the resistance-only form."""
    raw = "Jamie (resistance=deflect): I told you we were short on time."
    assert _strip_transcript_meta_prefix(raw) == "I told you we were short on time."


def test_strip_transcript_meta_prefix_handles_user_bracket():
    raw = "[USER 2] Can we agree the staging data lands by Wednesday EOD?"
    assert (
        _strip_transcript_meta_prefix(raw)
        == "Can we agree the staging data lands by Wednesday EOD?"
    )


def test_strip_transcript_meta_prefix_handles_other_bracket():
    raw = "[OTHER] Some transcribed reply."
    assert _strip_transcript_meta_prefix(raw) == "Some transcribed reply."


def test_strip_transcript_meta_prefix_preserves_natural_turn_pointers():
    """A natural turn pointer the model wrote in its own words is not
    a render-format bracket and must NOT be stripped. The prompt
    explicitly encourages this shape; stripping it would silently
    eat the model's good-shape evidence."""
    raw = (
        "Jamie's first reply — 'I told you we were short on time.' "
        "She ran the predicted deflection."
    )
    assert _strip_transcript_meta_prefix(raw) == raw

    raw2 = (
        "USER 2 followed up with the hard deadline ask; Jamie's reply "
        "began 'Wednesday EOD is tight'."
    )
    assert _strip_transcript_meta_prefix(raw2) == raw2


def test_strip_transcript_meta_prefix_preserves_bare_quotes():
    """Pure quote evidence without any prefix is the simplest valid
    shape and must pass through unchanged."""
    raw = "I told you we were short on time."
    assert _strip_transcript_meta_prefix(raw) == raw


def test_strip_transcript_meta_prefix_returns_empty_when_only_prefix():
    """If the model emits ONLY the bracket prefix (no spoken content),
    the strip leaves an empty string — which the coercer treats as
    'no real evidence' and downgrades the scenario to did_not_occur."""
    raw = "Jamie (resistance=deflect, escalation=0.60):"
    assert _strip_transcript_meta_prefix(raw) == ""


def test_coerce_scenario_outcome_strips_transcript_prefix_from_evidence():
    """End-to-end: the prefix-leak shape from Step 10 Run 1 is cleaned
    by ``_coerce_scenario_outcome`` without changing match_quality
    when real content follows the bracket."""
    scenario_input = _three_scenarios()[0]
    raw = {
        "match_quality": "direct_hit",
        "materialized": True,
        "evidence": (
            "Jamie (resistance=deflect, escalation=0.60): "
            "I told you we were short on time. It's not a personal failing."
        ),
        "notes": "Direct hit on the predicted deflection.",
    }
    out = _coerce_scenario_outcome(raw, scenario_input=scenario_input, scenario_id=1)
    assert out["match_quality"] == "direct_hit"
    assert out["materialized"] is True
    assert out["evidence"].startswith("I told you we were short on time.")
    assert "resistance=" not in out["evidence"]
    assert "escalation=" not in out["evidence"]


def test_coerce_scenario_outcome_downgrades_when_evidence_is_only_prefix():
    """If the model emits a direct_hit with evidence that is ONLY the
    bracket prefix (no actual spoken content), the strip leaves the
    empty string and the entry downgrades to did_not_occur — same
    path as a completely empty evidence string."""
    scenario_input = _three_scenarios()[0]
    raw = {
        "match_quality": "direct_hit",
        "materialized": True,
        "evidence": "Jamie (resistance=deflect, escalation=0.60):",
        "notes": "Tried to claim a hit but had nothing to back it up.",
    }
    out = _coerce_scenario_outcome(raw, scenario_input=scenario_input, scenario_id=1)
    assert out["match_quality"] == "did_not_occur"
    assert out["materialized"] is False
    assert out["evidence"] == ""


def test_coerce_scenario_outcome_rejects_unrecognisable_match_quality():
    scenario_input = _three_scenarios()[0]
    with pytest.raises(AftermathError, match="match_quality"):
        _coerce_scenario_outcome(
            {"match_quality": "kinda?", "notes": "n"},
            scenario_input=scenario_input,
            scenario_id=1,
        )


def test_coerce_scenario_outcome_rejects_empty_notes():
    scenario_input = _three_scenarios()[0]
    with pytest.raises(AftermathError, match="notes"):
        _coerce_scenario_outcome(
            {
                "match_quality": "did_not_occur",
                "materialized": False,
                "evidence": "",
                "notes": "   ",
            },
            scenario_input=scenario_input,
            scenario_id=1,
        )


def test_coerce_scenario_outcome_handles_non_dict_input():
    """A garbage entry (None, a list, etc.) is treated as an empty
    dict so the standard required-field checks raise — letting the
    retry path catch a model that emitted scenario_outcomes as a list
    of strings."""
    scenario_input = _three_scenarios()[0]
    with pytest.raises(AftermathError, match="match_quality"):
        _coerce_scenario_outcome(
            None, scenario_input=scenario_input, scenario_id=1
        )


# ---------------------------------------------------------------------------
# _coerce_unforeseen_moments
# ---------------------------------------------------------------------------


def test_coerce_unforeseen_moments_keeps_valid_entries():
    raw = [
        {"kind": "risk", "turn": 2, "description": "Jamie escalated unexpectedly."},
        {"kind": "opportunity", "description": "User found an opening at the end."},
    ]
    out = _coerce_unforeseen_moments(raw, user_turn_count=5)
    assert len(out) == 2
    assert out[0]["turn"] == 2
    # Optional turn is omitted when not provided.
    assert "turn" not in out[1]


def test_coerce_unforeseen_moments_normalises_kind_synonyms():
    raw = [
        {"kind": "threat", "description": "An unexpected pushback."},
        {"kind": "opening", "description": "A chance the user found."},
    ]
    out = _coerce_unforeseen_moments(raw, user_turn_count=5)
    assert [m["kind"] for m in out] == ["risk", "opportunity"]


def test_coerce_unforeseen_moments_drops_unknown_kinds_and_empty_descriptions():
    raw = [
        {"kind": "vibes", "description": "Something happened."},
        {"kind": "risk", "description": "   "},
        {"description": "Missing kind."},
    ]
    assert _coerce_unforeseen_moments(raw, user_turn_count=5) == []


def test_coerce_unforeseen_moments_clamps_turn_into_range():
    raw = [
        {"kind": "risk", "turn": 99, "description": "out of range turn"},
        {"kind": "risk", "turn": 0, "description": "below minimum"},
        {"kind": "risk", "turn": "garbage", "description": "unrecoverable turn"},
    ]
    out = _coerce_unforeseen_moments(raw, user_turn_count=5)
    assert out[0]["turn"] == 5
    assert out[1]["turn"] == 1
    # Unrecoverable turn is dropped from the optional field, but the
    # entry survives (kind + description were valid).
    assert "turn" not in out[2]


def test_coerce_unforeseen_moments_returns_empty_on_non_list():
    assert _coerce_unforeseen_moments(None, user_turn_count=5) == []
    assert _coerce_unforeseen_moments("string", user_turn_count=5) == []
    assert _coerce_unforeseen_moments({"oops": "dict"}, user_turn_count=5) == []


def test_coerce_unforeseen_moments_caps_at_max_list_items():
    raw = [
        {"kind": "risk", "description": f"item {i}"}
        for i in range(20)
    ]
    out = _coerce_unforeseen_moments(raw, user_turn_count=5)
    assert len(out) == 8


# ---------------------------------------------------------------------------
# _coerce_aftermath_payload
# ---------------------------------------------------------------------------


def test_coerce_payload_happy_path():
    result = _coerce_aftermath_payload(
        _full_payload(),
        premortem_scenarios=_three_scenarios(),
        user_turn_count=5,
    )
    assert result["goal_outcome"]["status"] == "partial"
    assert len(result["scenario_outcomes"]) == 3
    assert [o["scenario_id"] for o in result["scenario_outcomes"]] == [1, 2, 3]
    # Titles + predicted_resistance_types are forced from the premortem input.
    titles = [o["title"] for o in result["scenario_outcomes"]]
    assert titles == [s["title"] for s in _three_scenarios()]
    assert [o["predicted_resistance_type"] for o in result["scenario_outcomes"]] == [
        "deflect",
        "counter_attack",
        "guilt_trip",
    ]
    assert result["prediction_accuracy"] == 0.6
    assert result["aftermath_id"].startswith("aftermath_")
    assert "T" in result["generated_at"]


def test_coerce_payload_forces_three_scenarios_even_if_model_emits_two():
    """The pre-mortem is the source of truth for scenario count. If the
    model emits two entries, the third must come from somewhere — and
    the per-entry coercer will raise a clean error for the missing
    third, which the retry path catches."""
    payload = _full_payload()
    payload["scenario_outcomes"] = payload["scenario_outcomes"][:2]
    with pytest.raises(AftermathError, match="match_quality"):
        # The third (missing) entry is passed as {} which fails the
        # required-field check on match_quality. That's the desired
        # behaviour — surface the failure so the retry path runs.
        _coerce_aftermath_payload(
            payload,
            premortem_scenarios=_three_scenarios(),
            user_turn_count=5,
        )


def test_coerce_payload_ignores_extra_scenarios_beyond_count():
    """If the model emits four outcomes, we keep the first three — the
    pre-mortem only has three scenarios, and there is no fourth scenario
    to attribute the extra entry to."""
    payload = _full_payload()
    extra = dict(payload["scenario_outcomes"][0])
    extra["scenario_id"] = 4
    payload["scenario_outcomes"].append(extra)
    result = _coerce_aftermath_payload(
        payload,
        premortem_scenarios=_three_scenarios(),
        user_turn_count=5,
    )
    assert len(result["scenario_outcomes"]) == 3
    assert [o["scenario_id"] for o in result["scenario_outcomes"]] == [1, 2, 3]


def test_coerce_payload_rejects_wrong_premortem_count():
    """Aftermath is meaningless without exactly three pre-mortem
    scenarios — the schema's minItems == maxItems == 3 contract."""
    with pytest.raises(AftermathError, match="exactly 3 pre-mortem"):
        _coerce_aftermath_payload(
            _full_payload(),
            premortem_scenarios=_three_scenarios()[:2],
            user_turn_count=5,
        )


def test_coerce_payload_clamps_prediction_accuracy():
    payload_high = _full_payload(prediction_accuracy=1.5)
    payload_low = _full_payload(prediction_accuracy=-0.3)
    payload_garbage = _full_payload(prediction_accuracy="not a number")
    assert _coerce_aftermath_payload(
        payload_high,
        premortem_scenarios=_three_scenarios(),
        user_turn_count=5,
    )["prediction_accuracy"] == 1.0
    assert _coerce_aftermath_payload(
        payload_low,
        premortem_scenarios=_three_scenarios(),
        user_turn_count=5,
    )["prediction_accuracy"] == 0.0
    assert _coerce_aftermath_payload(
        payload_garbage,
        premortem_scenarios=_three_scenarios(),
        user_turn_count=5,
    )["prediction_accuracy"] == 0.0


def test_coerce_payload_rejects_empty_next_round_focus():
    with pytest.raises(AftermathError, match="next_round_focus"):
        _coerce_aftermath_payload(
            _full_payload(next_round_focus="   "),
            premortem_scenarios=_three_scenarios(),
            user_turn_count=5,
        )


def test_coerce_payload_rejects_non_dict_top_level():
    with pytest.raises(AftermathError, match="not an object"):
        _coerce_aftermath_payload(
            ["not", "a", "dict"],
            premortem_scenarios=_three_scenarios(),
            user_turn_count=5,
        )


def test_coerce_payload_rejects_non_list_scenario_outcomes():
    payload = _full_payload(scenario_outcomes="not a list")
    with pytest.raises(AftermathError, match="must be a list"):
        _coerce_aftermath_payload(
            payload,
            premortem_scenarios=_three_scenarios(),
            user_turn_count=5,
        )


def test_coerce_payload_drops_invalid_unforeseen_silently():
    """A single malformed unforeseen_moments entry should not lose the
    rest of the payload — same defence-in-depth as debrief's per-array
    drop-bad-entry behaviour."""
    payload = _full_payload(
        unforeseen_moments=[
            {"kind": "risk", "description": "valid entry"},
            {"kind": "vibes", "description": "unknown kind — dropped"},
            "garbage — dropped",
        ]
    )
    result = _coerce_aftermath_payload(
        payload,
        premortem_scenarios=_three_scenarios(),
        user_turn_count=5,
    )
    assert len(result["unforeseen_moments"]) == 1
    assert result["unforeseen_moments"][0]["kind"] == "risk"


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_aftermath_schema_match_quality_enum_matches_runtime():
    """Drift between the schema's enum and the runtime tuple means
    coerced outputs could fail downstream validators silently."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_path = (
        schema["properties"]["scenario_outcomes"]["items"]
        ["properties"]["match_quality"]["enum"]
    )
    assert tuple(enum_path) == ALLOWED_MATCH_QUALITIES


def test_aftermath_schema_unforeseen_kind_enum_matches_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_path = (
        schema["properties"]["unforeseen_moments"]["items"]
        ["properties"]["kind"]["enum"]
    )
    assert tuple(enum_path) == ALLOWED_UNFORESEEN_KINDS


def test_aftermath_schema_goal_status_enum_matches_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_path = (
        schema["properties"]["goal_outcome"]["properties"]["status"]["enum"]
    )
    assert tuple(enum_path) == ALLOWED_GOAL_STATUSES


def test_aftermath_schema_required_top_level_fields_match_runtime_output():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "Aftermath"
    runtime_required = {
        "goal_outcome",
        "scenario_outcomes",
        "unforeseen_moments",
        "prediction_accuracy",
        "next_round_focus",
    }
    assert set(schema["required"]) == runtime_required


def test_aftermath_schema_scenario_count_pinned_at_three():
    """Premortem and aftermath share the three-scenario contract.
    Drift in either schema is silently corrupting the handoff."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    sc = schema["properties"]["scenario_outcomes"]
    assert sc["minItems"] == 3
    assert sc["maxItems"] == 3


def test_aftermath_prompt_has_all_placeholders():
    """The default prompt file must exist and declare exactly the five
    placeholders the runtime renders. Drift breaks ``Prompt.render``."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "$user_goal",
        "$person_profile_block",
        "$premortem_block",
        "$debrief_block",
        "$transcript",
    ):
        assert placeholder in text, f"prompt missing {placeholder}"


def test_aftermath_config_defaults_match_documented_runtime():
    """Defaults are part of the public contract — guard against
    accidental flips to ``enable_thinking=True`` or a non-default
    prompt name."""
    cfg = AftermathConfig()
    assert cfg.prompt_name == DEFAULT_AFTERMATH_PROMPT_NAME
    assert cfg.enable_thinking is False
    assert cfg.transcript == []
    assert cfg.premortem == {}
    # JSON body + room for unforeseen moments.
    assert cfg.max_new_tokens >= 768


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Mirror the rationale from the other intelligence runtimes — just
    enough randomness to escape a single bad greedy trajectory."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1
