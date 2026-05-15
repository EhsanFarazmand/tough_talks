"""Unit tests for the Relationship Pulse Tracker runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - ``_coerce_pulse_payload`` contract: required top-level fields,
    person_id forcing, version bumping (cold-start vs prior), round_count
    derivation, round_summaries shape (one per input round, in order,
    round_id / started_at forced verbatim from input), all-unknown
    override of ``health_trend`` to ``insufficient_data``.
  - Per-section coercers
    (``_coerce_round_summary``, ``_coerce_recurring_patterns``,
    ``_coerce_emerging_concerns``, ``_coerce_relationship_wins``) drop
    or fix malformed entries — including the schema's
    ``minItems: 2`` rule on ``evidence_round_ids`` (an entry that
    cannot cite two known rounds is not "recurring").
  - Accumulation: ``recurring_patterns`` and ``relationship_wins``
    merge prior + new (deduped, capped at ``_MAX_LIST_ITEMS``); same
    shape as PersonVault's qualitative lists.
  - Enum normalisers (``_normalize_health_trend``,
    ``_normalize_round_goal_status``) accept documented synonyms and
    handle casing / hyphen drift.
  - person_id / person_name / relationship_type resolvers prefer the
    cfg-level inputs over prior pulse fields.
  - ``format_rounds_block`` and ``format_prior_pulse_block`` rendering:
    sentinels on empty input, aftermath / debrief headlines surfaced.
  - ``build_pulse_messages`` shape (single user message at index 0).
  - schema-vs-runtime contract: schema enums match runtime tuples; the
    default prompt declares the five placeholders the runtime renders;
    PulseConfig defaults are the documented public contract.

The real ``generate_pulse`` call requires a loaded model and is
exercised in ``notebooks/phase4/step11_pulse.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.pulse import (
    ALLOWED_HEALTH_TRENDS,
    ALLOWED_ROUND_GOAL_STATUSES,
    DEFAULT_PULSE_PROMPT_NAME,
    PulseConfig,
    PulseError,
    RETRY_SAMPLING,
    ROUND_COUNT_MIN,
    _coerce_emerging_concerns,
    _coerce_pulse_payload,
    _coerce_recurring_patterns,
    _coerce_relationship_wins,
    _coerce_round_summary,
    _normalize_health_trend,
    _normalize_round_goal_status,
    _resolve_person_id,
    _resolve_person_name,
    _resolve_relationship_type,
    _round_goal_status_from_aftermath,
    _round_prediction_accuracy_from_aftermath,
    build_pulse_messages,
    format_prior_pulse_block,
    format_rounds_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "pulse.schema.json"
PROMPT_PATH = REPO_ROOT / "data" / "prompts" / f"{DEFAULT_PULSE_PROMPT_NAME}.md"

JAMIE_PERSON_ID = "person_cf528d57b789"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _person_profile() -> dict:
    return {
        "person_id": JAMIE_PERSON_ID,
        "name": "Jamie",
        "relationship_type": "colleague",
        "version": 2,
        "conversation_count": 2,
        "profile": {
            "communication_style": "defensive",
            "emotional_triggers": ["citing past commitments"],
            "de_escalation_keys": ["reframing as joint problem-solving"],
            "common_deflections": ["I told you"],
        },
    }


def _round_with_aftermath(
    *,
    round_id: str,
    started_at: str,
    goal_status: str = "partial",
    prediction_accuracy: float = 0.6,
) -> dict:
    return {
        "round_id": round_id,
        "started_at": started_at,
        "user_goal": "Get Jamie to commit to Wednesday EOD.",
        "aftermath": {
            "goal_outcome": {
                "status": goal_status,
                "summary": "Mostly held the line on the deadline.",
            },
            "scenario_outcomes": [
                {
                    "scenario_id": 1,
                    "title": "The 'I told you' blame redirect",
                    "predicted_resistance_type": "deflect",
                    "match_quality": "direct_hit",
                    "materialized": True,
                    "evidence": "Jamie said 'I told you' on turn 1.",
                    "notes": "Played out as predicted.",
                },
            ],
            "unforeseen_moments": [],
            "prediction_accuracy": prediction_accuracy,
            "next_round_focus": "Name the deflection out loud earlier.",
        },
        "debrief": {
            "wins": [{"turn": 2, "description": "Held the deadline."}],
            "ground_lost": [],
            "over_apologies": [],
            "missed_openings": [],
            "one_fix_next_time": "Move past blame to the next step.",
        },
    }


def _round_without_aftermath(
    *,
    round_id: str,
    started_at: str,
) -> dict:
    return {
        "round_id": round_id,
        "started_at": started_at,
        "user_goal": "Get Jamie to commit to Wednesday EOD.",
    }


def _two_rounds() -> list[dict]:
    return [
        _round_with_aftermath(
            round_id="round_a",
            started_at="2026-04-30T16:00:00+00:00",
            goal_status="not_achieved",
            prediction_accuracy=0.85,
        ),
        _round_with_aftermath(
            round_id="round_b",
            started_at="2026-05-14T15:30:00+00:00",
            goal_status="partial",
            prediction_accuracy=0.75,
        ),
    ]


def _full_payload(**overrides) -> dict:
    """A complete, valid model-emitted pulse payload over two rounds."""
    base = {
        "health_score": 0.55,
        "health_trend": "improving",
        "trajectory_summary": (
            "Round A was rough — Jamie ran the 'I told you' deflection and "
            "you apologised through it. Round B you named the move and "
            "moved her toward a concrete commitment."
        ),
        "round_summaries": [
            {
                "round_id": "round_a",
                "started_at": "2026-04-30T16:00:00+00:00",
                "goal_status": "not_achieved",
                "prediction_accuracy": 0.85,
                "headline": "Jamie deflected and you apologised through it.",
            },
            {
                "round_id": "round_b",
                "started_at": "2026-05-14T15:30:00+00:00",
                "goal_status": "partial",
                "prediction_accuracy": 0.75,
                "headline": "You held the deadline and reframed the escalation as a joint problem.",
            },
        ],
        "recurring_patterns": [
            {
                "pattern": "Jamie opens with 'I told you' when you cite past commitments.",
                "evidence_round_ids": ["round_a", "round_b"],
            }
        ],
        "emerging_concerns": [
            {
                "concern": "You softened the ask after the guilt-trip — watch for that fold next time.",
                "round_id": "round_b",
            }
        ],
        "relationship_wins": [
            {
                "win": "Reframing the escalation as a joint problem calms Jamie.",
                "round_id": "round_b",
            }
        ],
        "next_step_recommendation": (
            "Name the 'I told you' deflection out loud the first time it appears next round."
        ),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Enum normalisers
# ---------------------------------------------------------------------------


def test_normalize_health_trend_accepts_canonical_values():
    for value in ALLOWED_HEALTH_TRENDS:
        assert _normalize_health_trend(value) == value


def test_normalize_health_trend_handles_casing_and_separator():
    assert _normalize_health_trend("Improving") == "improving"
    assert _normalize_health_trend("INSUFFICIENT_DATA") == "insufficient_data"
    assert _normalize_health_trend("insufficient-data") == "insufficient_data"
    assert _normalize_health_trend("insufficient data") == "insufficient_data"


def test_normalize_health_trend_accepts_documented_synonyms():
    cases = {
        "worsening": "declining",
        "getting worse": "declining",
        "downward": "declining",
        "better": "improving",
        "getting better": "improving",
        "upward": "improving",
        "flat": "stable",
        "steady": "stable",
        "unstable": "volatile",
        "erratic": "volatile",
        "swinging": "volatile",
        "n/a": "insufficient_data",
        "unknown": "insufficient_data",
        "not enough data": "insufficient_data",
    }
    for raw, expected in cases.items():
        assert _normalize_health_trend(raw) == expected, raw


def test_normalize_health_trend_returns_none_on_unknown():
    assert _normalize_health_trend("vibes") is None
    assert _normalize_health_trend("") is None
    assert _normalize_health_trend(None) is None
    assert _normalize_health_trend(123) is None


def test_normalize_round_goal_status_accepts_canonical_values():
    for value in ALLOWED_ROUND_GOAL_STATUSES:
        assert _normalize_round_goal_status(value) == value


def test_normalize_round_goal_status_accepts_documented_synonyms():
    cases = {
        "yes": "achieved",
        "success": "achieved",
        "completed": "achieved",
        "partly": "partial",
        "halfway": "partial",
        "no": "not_achieved",
        "failed": "not_achieved",
        "missed": "not_achieved",
        "n/a": "unknown",
        "no_aftermath": "unknown",
        "tbd": "unknown",
        "pending": "unknown",
    }
    for raw, expected in cases.items():
        assert _normalize_round_goal_status(raw) == expected, raw


# ---------------------------------------------------------------------------
# Per-round artifact extraction
# ---------------------------------------------------------------------------


def test_round_goal_status_from_aftermath_reads_canonical():
    r = _round_with_aftermath(
        round_id="x", started_at="t", goal_status="achieved"
    )
    assert _round_goal_status_from_aftermath(r) == "achieved"


def test_round_goal_status_from_aftermath_returns_unknown_when_missing():
    r = _round_without_aftermath(round_id="x", started_at="t")
    assert _round_goal_status_from_aftermath(r) == "unknown"


def test_round_goal_status_from_aftermath_returns_unknown_on_garbage():
    r = {"aftermath": {"goal_outcome": {"status": "definitely not a status"}}}
    assert _round_goal_status_from_aftermath(r) == "unknown"


def test_round_prediction_accuracy_from_aftermath_clamps_and_defaults():
    r_high = {"aftermath": {"prediction_accuracy": 1.5}}
    r_low = {"aftermath": {"prediction_accuracy": -0.2}}
    r_none = {}
    r_garbage = {"aftermath": {"prediction_accuracy": "not a number"}}
    assert _round_prediction_accuracy_from_aftermath(r_high) == 1.0
    assert _round_prediction_accuracy_from_aftermath(r_low) == 0.0
    assert _round_prediction_accuracy_from_aftermath(r_none) == 0.0
    assert _round_prediction_accuracy_from_aftermath(r_garbage) == 0.0


# ---------------------------------------------------------------------------
# _coerce_round_summary
# ---------------------------------------------------------------------------


def test_coerce_round_summary_forces_round_id_and_started_at_from_input():
    """Model output is NOT trusted on round metadata — defence-in-depth.
    Mirrors Step 10's title / predicted_resistance_type forcing."""
    round_record = _round_with_aftermath(
        round_id="canonical_id",
        started_at="2026-04-30T16:00:00+00:00",
        goal_status="partial",
        prediction_accuracy=0.6,
    )
    raw = {
        "round_id": "the model made this up",
        "started_at": "yesterday",
        "goal_status": "achieved",  # disagrees with aftermath
        "prediction_accuracy": 0.9,
        "headline": "Held the line.",
    }
    result = _coerce_round_summary(raw, round_record=round_record, index=0)
    assert result["round_id"] == "canonical_id"
    assert result["started_at"] == "2026-04-30T16:00:00+00:00"
    # Aftermath wins on goal_status when the aftermath has signal.
    assert result["goal_status"] == "partial"


def test_coerce_round_summary_uses_model_goal_status_when_aftermath_is_unknown():
    round_record = _round_without_aftermath(
        round_id="r1", started_at="2026-05-01T00:00:00+00:00"
    )
    raw = {
        "goal_status": "achieved",
        "prediction_accuracy": 0.5,
        "headline": "Some headline.",
    }
    result = _coerce_round_summary(raw, round_record=round_record, index=0)
    assert result["goal_status"] == "achieved"


def test_coerce_round_summary_defaults_to_unknown_when_both_absent():
    round_record = _round_without_aftermath(
        round_id="r1", started_at="2026-05-01T00:00:00+00:00"
    )
    raw = {
        "goal_status": "definitely garbage",
        "headline": "Some headline.",
    }
    result = _coerce_round_summary(raw, round_record=round_record, index=0)
    assert result["goal_status"] == "unknown"


def test_coerce_round_summary_falls_back_to_aftermath_accuracy_on_garbage():
    round_record = _round_with_aftermath(
        round_id="r1",
        started_at="2026-05-01T00:00:00+00:00",
        prediction_accuracy=0.42,
    )
    raw = {
        "goal_status": "partial",
        "prediction_accuracy": "not a number",
        "headline": "Some headline.",
    }
    result = _coerce_round_summary(raw, round_record=round_record, index=0)
    assert result["prediction_accuracy"] == 0.42


def test_coerce_round_summary_clamps_model_accuracy_to_unit():
    round_record = _round_with_aftermath(
        round_id="r1",
        started_at="2026-05-01T00:00:00+00:00",
        prediction_accuracy=0.5,
    )
    raw_high = {"goal_status": "partial", "prediction_accuracy": 1.7, "headline": "x"}
    raw_low = {"goal_status": "partial", "prediction_accuracy": -0.3, "headline": "x"}
    assert _coerce_round_summary(raw_high, round_record=round_record, index=0)["prediction_accuracy"] == 1.0
    assert _coerce_round_summary(raw_low, round_record=round_record, index=0)["prediction_accuracy"] == 0.0


def test_coerce_round_summary_rejects_empty_headline():
    round_record = _round_with_aftermath(
        round_id="r1", started_at="t"
    )
    with pytest.raises(PulseError, match="headline"):
        _coerce_round_summary(
            {"goal_status": "partial", "headline": "   "},
            round_record=round_record,
            index=0,
        )


# ---------------------------------------------------------------------------
# _coerce_recurring_patterns
# ---------------------------------------------------------------------------


def test_coerce_recurring_patterns_drops_entries_with_fewer_than_two_known_ids():
    """The schema's ``minItems: 2`` on ``evidence_round_ids`` IS the
    definition of recurring. An entry that can't cite two known rounds
    isn't recurring — it gets dropped silently same as Step 10's
    ``did_not_occur`` forcing empty evidence."""
    raw = [
        {
            "pattern": "Jamie deflects on past commitments.",
            "evidence_round_ids": ["round_a", "round_b"],
        },
        {
            "pattern": "One-round observation, not recurring.",
            "evidence_round_ids": ["round_a"],
        },
        {
            "pattern": "Cites only unknown ids.",
            "evidence_round_ids": ["nonexistent_1", "nonexistent_2"],
        },
    ]
    result = _coerce_recurring_patterns(
        raw,
        known_round_ids=["round_a", "round_b"],
        prior_patterns=None,
    )
    assert len(result) == 1
    assert result[0]["pattern"].startswith("Jamie deflects")
    assert result[0]["evidence_round_ids"] == ["round_a", "round_b"]


def test_coerce_recurring_patterns_filters_unknown_ids_within_entry():
    raw = [
        {
            "pattern": "Mixed known and unknown ids.",
            "evidence_round_ids": ["round_a", "ghost", "round_b", "phantom"],
        },
    ]
    result = _coerce_recurring_patterns(
        raw,
        known_round_ids=["round_a", "round_b"],
        prior_patterns=None,
    )
    assert len(result) == 1
    assert result[0]["evidence_round_ids"] == ["round_a", "round_b"]


def test_coerce_recurring_patterns_dedupes_repeated_evidence_ids():
    raw = [
        {
            "pattern": "Same id repeated cannot satisfy minItems 2.",
            "evidence_round_ids": ["round_a", "round_a"],
        },
    ]
    result = _coerce_recurring_patterns(
        raw,
        known_round_ids=["round_a", "round_b"],
        prior_patterns=None,
    )
    # Only one distinct id after dedup -> < 2 -> dropped.
    assert result == []


def test_coerce_recurring_patterns_accumulates_prior_and_new():
    """Same accumulation semantics as PersonVault — prior first
    (oldest), new appended, deduped on pattern text (case-insensitive)."""
    prior = [
        {
            "pattern": "Jamie deflects with 'I told you'.",
            "evidence_round_ids": ["round_a", "round_b"],
        },
    ]
    new = [
        # Same pattern as prior, case-mismatched — should dedupe.
        {
            "pattern": "jamie deflects with 'i told you'.",
            "evidence_round_ids": ["round_a", "round_b"],
        },
        {
            "pattern": "Brand new pattern.",
            "evidence_round_ids": ["round_a", "round_b"],
        },
    ]
    result = _coerce_recurring_patterns(
        new,
        known_round_ids=["round_a", "round_b"],
        prior_patterns=prior,
    )
    assert len(result) == 2
    # Prior preserved at index 0 (the older observation casing).
    assert result[0]["pattern"].startswith("Jamie")
    assert result[1]["pattern"] == "Brand new pattern."


def test_coerce_recurring_patterns_caps_combined_list():
    """When prior + new exceeds _MAX_LIST_ITEMS, oldest priors are
    dropped first."""
    prior = [
        {
            "pattern": f"old_pattern_{i}",
            "evidence_round_ids": ["round_a", "round_b"],
        }
        for i in range(6)
    ]
    new = [
        {
            "pattern": f"new_pattern_{i}",
            "evidence_round_ids": ["round_a", "round_b"],
        }
        for i in range(5)
    ]
    result = _coerce_recurring_patterns(
        new,
        known_round_ids=["round_a", "round_b"],
        prior_patterns=prior,
    )
    assert len(result) == 8
    # Most recent additions (new_pattern_4 last) survive; oldest priors dropped.
    assert any(e["pattern"] == "new_pattern_4" for e in result)
    assert not any(e["pattern"] == "old_pattern_0" for e in result)


# ---------------------------------------------------------------------------
# _coerce_emerging_concerns
# ---------------------------------------------------------------------------


def test_coerce_emerging_concerns_drops_unknown_round_id_silently():
    raw = [
        {"concern": "Real concern attached to a real round.", "round_id": "round_b"},
        {"concern": "Real concern with bogus round_id.", "round_id": "ghost"},
        {"concern": "Real concern without a round_id."},
    ]
    result = _coerce_emerging_concerns(
        raw, known_round_ids=["round_a", "round_b"]
    )
    assert len(result) == 3
    assert result[0]["round_id"] == "round_b"
    # Unknown round_id dropped; entry kept.
    assert "round_id" not in result[1]
    # No round_id provided; entry kept.
    assert "round_id" not in result[2]


def test_coerce_emerging_concerns_drops_empty_concern():
    raw = [
        {"concern": "  ", "round_id": "round_a"},
        {"concern": "valid", "round_id": "round_b"},
    ]
    result = _coerce_emerging_concerns(
        raw, known_round_ids=["round_a", "round_b"]
    )
    assert len(result) == 1
    assert result[0]["concern"] == "valid"


def test_coerce_emerging_concerns_returns_empty_on_non_list():
    assert _coerce_emerging_concerns(
        "not a list", known_round_ids=["round_a"]
    ) == []
    assert _coerce_emerging_concerns(None, known_round_ids=["round_a"]) == []


# ---------------------------------------------------------------------------
# _coerce_relationship_wins
# ---------------------------------------------------------------------------


def test_coerce_relationship_wins_accumulates_prior_and_new():
    prior = [{"win": "Naming the deflection works.", "round_id": "round_a"}]
    new = [
        {"win": "Naming the deflection works.", "round_id": "round_b"},  # dup
        {"win": "Reframing as joint problem-solving calms her.", "round_id": "round_b"},
    ]
    result = _coerce_relationship_wins(
        new,
        known_round_ids=["round_a", "round_b"],
        prior_wins=prior,
    )
    assert len(result) == 2
    # Prior preserved at index 0, new appended.
    assert result[0]["win"] == "Naming the deflection works."
    assert result[1]["win"] == "Reframing as joint problem-solving calms her."


def test_coerce_relationship_wins_drops_unknown_round_ids_silently():
    raw = [{"win": "Real win.", "round_id": "ghost"}]
    result = _coerce_relationship_wins(
        raw, known_round_ids=["round_a", "round_b"], prior_wins=None
    )
    assert len(result) == 1
    assert "round_id" not in result[0]


# ---------------------------------------------------------------------------
# _resolve_person_id / _resolve_person_name / _resolve_relationship_type
# ---------------------------------------------------------------------------


def test_resolve_person_id_prefers_cfg_then_profile_then_prior():
    cfg = PulseConfig(
        rounds=[],
        person_profile={"person_id": "from_profile"},
        person_id="from_cfg",
    )
    assert _resolve_person_id(cfg, {"person_id": "from_prior"}) == "from_cfg"

    cfg_no_explicit = PulseConfig(
        rounds=[],
        person_profile={"person_id": "from_profile"},
    )
    assert _resolve_person_id(cfg_no_explicit, {"person_id": "from_prior"}) == "from_profile"

    cfg_prior_only = PulseConfig(rounds=[], person_profile={})
    assert _resolve_person_id(cfg_prior_only, {"person_id": "from_prior"}) == "from_prior"


def test_resolve_person_id_raises_when_nothing_available():
    cfg = PulseConfig(rounds=[], person_profile={})
    with pytest.raises(PulseError, match="person_id"):
        _resolve_person_id(cfg, {})


def test_resolve_relationship_type_normalises_and_falls_back():
    cfg = PulseConfig(
        rounds=[],
        person_profile={"relationship_type": "COLLEAGUE"},
    )
    assert _resolve_relationship_type(cfg, {}) == "colleague"

    cfg_unknown = PulseConfig(
        rounds=[],
        person_profile={"relationship_type": "former_roommate"},
    )
    # Unknown -> default.
    assert _resolve_relationship_type(cfg_unknown, {}) == "other"


def test_resolve_person_name_prefers_profile_then_prior_then_default():
    cfg = PulseConfig(rounds=[], person_profile={"name": "Jamie"})
    assert _resolve_person_name(cfg, {"person_name": "old name"}) == "Jamie"

    cfg_no_name = PulseConfig(rounds=[], person_profile={})
    assert _resolve_person_name(cfg_no_name, {"person_name": "old name"}) == "old name"

    cfg_empty = PulseConfig(rounds=[], person_profile={})
    assert _resolve_person_name(cfg_empty, {}) == "Other"


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def test_format_rounds_block_surfaces_aftermath_headlines():
    block = format_rounds_block(_two_rounds())
    assert "round_a" in block
    assert "round_b" in block
    assert "goal_outcome" in block
    assert "prediction_accuracy" in block
    assert "scenario_outcomes" in block
    # Each round renders a separate block (joined with blank line).
    assert "Round 1" in block
    assert "Round 2" in block


def test_format_rounds_block_handles_missing_aftermath_with_sentinel():
    rounds = [
        _round_without_aftermath(round_id="r1", started_at="2026-05-01T00:00:00+00:00"),
        _round_without_aftermath(round_id="r2", started_at="2026-05-02T00:00:00+00:00"),
    ]
    block = format_rounds_block(rounds)
    assert "no aftermath" in block
    assert "round goal_status is unknown" in block


def test_format_rounds_block_empty_input():
    assert format_rounds_block([]) == ""
    assert format_rounds_block(None) == ""


def test_format_prior_pulse_block_sentinel_on_empty():
    assert "no prior pulse" in format_prior_pulse_block(None)
    assert "no prior pulse" in format_prior_pulse_block({})


def test_format_prior_pulse_block_renders_carryforward_signals():
    prior = {
        "version": 1,
        "round_count": 2,
        "health_score": 0.55,
        "health_trend": "improving",
        "recurring_patterns": [
            {
                "pattern": "Jamie deflects on past commitments.",
                "evidence_round_ids": ["round_a", "round_b"],
            }
        ],
        "relationship_wins": [{"win": "Naming the deflection works."}],
    }
    block = format_prior_pulse_block(prior)
    assert "prior_version: 1" in block
    assert "0.55" in block
    assert "improving" in block
    assert "Jamie deflects on past commitments." in block
    assert "Naming the deflection works." in block


# ---------------------------------------------------------------------------
# build_pulse_messages
# ---------------------------------------------------------------------------


def test_build_pulse_messages_is_single_user_message():
    msgs = build_pulse_messages("hello")
    assert msgs == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# _coerce_pulse_payload — top-level contract
# ---------------------------------------------------------------------------


def test_coerce_payload_happy_path():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    result = _coerce_pulse_payload(_full_payload(), cfg=cfg, prior_pulse={})
    assert result["person_id"] == JAMIE_PERSON_ID
    assert result["person_name"] == "Jamie"
    assert result["relationship_type"] == "colleague"
    assert result["version"] == 1
    assert result["round_count"] == 2
    assert result["health_trend"] == "improving"
    assert 0.0 <= result["health_score"] <= 1.0
    assert [s["round_id"] for s in result["round_summaries"]] == ["round_a", "round_b"]
    # round_id / started_at copied verbatim from input.
    assert [s["started_at"] for s in result["round_summaries"]] == [
        "2026-04-30T16:00:00+00:00",
        "2026-05-14T15:30:00+00:00",
    ]
    assert result["pulse_id"].startswith("pulse_")
    assert "T" in result["updated_at"]


def test_coerce_payload_bumps_version_from_prior():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    prior = {"version": 3}
    result = _coerce_pulse_payload(_full_payload(), cfg=cfg, prior_pulse=prior)
    assert result["version"] == 4


def test_coerce_payload_round_count_derived_from_input_not_payload():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    payload = _full_payload()
    # Even if the model emits a wrong round_count (we don't read it),
    # the runtime derives from len(round_summaries) which it builds
    # from input length.
    result = _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})
    assert result["round_count"] == 2


def test_coerce_payload_rejects_too_few_rounds():
    cfg = PulseConfig(rounds=_two_rounds()[:1], person_profile=_person_profile())
    with pytest.raises(PulseError, match="at least 2"):
        _coerce_pulse_payload(_full_payload(), cfg=cfg, prior_pulse={})


def test_coerce_payload_rejects_non_dict_top_level():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    with pytest.raises(PulseError, match="not an object"):
        _coerce_pulse_payload(["not", "a", "dict"], cfg=cfg, prior_pulse={})


def test_coerce_payload_rejects_non_list_round_summaries():
    payload = _full_payload(round_summaries="not a list")
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    with pytest.raises(PulseError, match="round_summaries must be a list"):
        _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})


def test_coerce_payload_rejects_empty_trajectory_summary():
    payload = _full_payload(trajectory_summary="   ")
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    with pytest.raises(PulseError, match="trajectory_summary"):
        _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})


def test_coerce_payload_rejects_empty_next_step():
    payload = _full_payload(next_step_recommendation="   ")
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    with pytest.raises(PulseError, match="next_step_recommendation"):
        _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})


def test_coerce_payload_clamps_health_score():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    assert (
        _coerce_pulse_payload(_full_payload(health_score=1.7), cfg=cfg, prior_pulse={})[
            "health_score"
        ]
        == 1.0
    )
    assert (
        _coerce_pulse_payload(_full_payload(health_score=-0.4), cfg=cfg, prior_pulse={})[
            "health_score"
        ]
        == 0.0
    )
    assert (
        _coerce_pulse_payload(
            _full_payload(health_score="garbage"), cfg=cfg, prior_pulse={}
        )["health_score"]
        == 0.0
    )


def test_coerce_payload_forces_insufficient_data_when_all_rounds_unknown():
    """When every round has no aftermath signal, the runtime overrides
    whatever health_trend the model picked. Same defence-in-depth shape
    as Step 09's _APOLOGY_CUE_RE — system contract enforced in code,
    not just in the prompt."""
    rounds_no_aftermath = [
        _round_without_aftermath(round_id="round_a", started_at="t1"),
        _round_without_aftermath(round_id="round_b", started_at="t2"),
    ]
    cfg = PulseConfig(
        rounds=rounds_no_aftermath, person_profile=_person_profile()
    )
    # Model emits 'stable' but every round is unknown.
    payload = _full_payload(
        health_trend="stable",
        round_summaries=[
            {
                "round_id": "round_a",
                "started_at": "t1",
                "goal_status": "unknown",
                "prediction_accuracy": 0.0,
                "headline": "Nothing to score.",
            },
            {
                "round_id": "round_b",
                "started_at": "t2",
                "goal_status": "unknown",
                "prediction_accuracy": 0.0,
                "headline": "Still nothing.",
            },
        ],
    )
    result = _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})
    assert result["health_trend"] == "insufficient_data"


def test_coerce_payload_rejects_invalid_health_trend_when_rounds_have_signal():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    payload = _full_payload(health_trend="totally made up")
    with pytest.raises(PulseError, match="health_trend"):
        _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})


def test_coerce_payload_drops_recurring_patterns_with_one_id_silently():
    """Same shape as Step 10's evidence-downgrade behaviour: malformed
    entries don't break the whole pulse; they get filtered out."""
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    payload = _full_payload(
        recurring_patterns=[
            {"pattern": "Valid recurring pattern.", "evidence_round_ids": ["round_a", "round_b"]},
            {"pattern": "Not recurring.", "evidence_round_ids": ["round_a"]},
        ],
    )
    result = _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})
    assert len(result["recurring_patterns"]) == 1
    assert result["recurring_patterns"][0]["pattern"] == "Valid recurring pattern."


def test_coerce_payload_round_summaries_in_input_order_even_if_model_scrambles():
    cfg = PulseConfig(rounds=_two_rounds(), person_profile=_person_profile())
    payload = _full_payload()
    # Swap the model's output entries; runtime must still produce
    # input-order output (round_a first, then round_b).
    payload["round_summaries"] = list(reversed(payload["round_summaries"]))
    result = _coerce_pulse_payload(payload, cfg=cfg, prior_pulse={})
    assert [s["round_id"] for s in result["round_summaries"]] == ["round_a", "round_b"]
    assert [s["started_at"] for s in result["round_summaries"]] == [
        "2026-04-30T16:00:00+00:00",
        "2026-05-14T15:30:00+00:00",
    ]


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_pulse_schema_health_trend_enum_matches_runtime():
    """Drift between schema enum and runtime tuple silently corrupts
    downstream validators. Same shape as Step 10's enum-parity check."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_path = schema["properties"]["health_trend"]["enum"]
    assert tuple(enum_path) == ALLOWED_HEALTH_TRENDS


def test_pulse_schema_goal_status_enum_matches_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_path = (
        schema["properties"]["round_summaries"]["items"]["properties"]["goal_status"]["enum"]
    )
    assert tuple(enum_path) == ALLOWED_ROUND_GOAL_STATUSES


def test_pulse_schema_required_top_level_fields_match_runtime_output():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "RelationshipPulse"
    runtime_required = {
        "person_id",
        "version",
        "round_count",
        "health_score",
        "health_trend",
        "trajectory_summary",
        "round_summaries",
        "recurring_patterns",
        "emerging_concerns",
        "relationship_wins",
        "next_step_recommendation",
        "updated_at",
    }
    assert set(schema["required"]) == runtime_required


def test_pulse_schema_round_count_minimum_matches_runtime():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["round_count"]["minimum"] == ROUND_COUNT_MIN
    assert schema["properties"]["round_summaries"]["minItems"] == ROUND_COUNT_MIN


def test_pulse_schema_evidence_round_ids_min_items_is_two():
    """The schema's ``minItems: 2`` is the source of truth for
    ``recurring_patterns`` semantics; ``_coerce_recurring_patterns``
    drops entries that violate it. Drift in either direction is
    silently corrupting the contract."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_schema = (
        schema["properties"]["recurring_patterns"]["items"]
        ["properties"]["evidence_round_ids"]
    )
    assert evidence_schema["minItems"] == 2


def test_pulse_prompt_has_all_placeholders():
    """The default prompt file must exist and declare exactly the five
    placeholders the runtime renders. Drift breaks Prompt.render."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "$person_name",
        "$relationship_type",
        "$person_profile_block",
        "$rounds_block",
        "$prior_pulse_block",
    ):
        assert placeholder in text, f"prompt missing {placeholder}"


def test_pulse_config_defaults_match_documented_runtime():
    """Defaults are part of the public contract — guard against
    accidental flips. enable_thinking=True is the documented default
    per the rule in knowledge/phases/rules.md."""
    cfg = PulseConfig()
    assert cfg.prompt_name == DEFAULT_PULSE_PROMPT_NAME
    assert cfg.enable_thinking is True
    assert cfg.rounds == []
    assert cfg.person_profile == {}
    assert cfg.prior_pulse is None
    # Token budget bumped for the largest-input Phase 4 task; covers
    # the 2-3 round case comfortably per the rule's addendum.
    assert cfg.max_new_tokens >= 4096


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Same rationale as the other intelligence runtimes — just enough
    randomness to escape a single bad greedy trajectory."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
