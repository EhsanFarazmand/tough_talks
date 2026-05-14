"""Unit tests for the Post-Round Debrief runtime.

Covers everything that can be validated without loading transformers
or running a real model:

  - ``_coerce_debrief_payload`` contract: required ``one_fix_next_time``,
    silent dropping of malformed array entries, turn clamping, list cap
  - per-array coercers (``_coerce_ground_lost``, ``_coerce_over_apologies``,
    ``_coerce_missed_openings``, ``_coerce_wins``) drop bad entries and
    keep the good ones
  - ``count_user_turns`` and ``format_practice_transcript`` shape: user
    turns numbered ``[USER 1]``, ``[USER 2]``, …; persona turns labelled
    by name with resistance/escalation metadata when present; ``other``
    turns labelled ``OTHER``
  - ``build_debrief_messages`` shape (single user message at index 0)
  - schema-vs-runtime contract: the schema's top-level ``required``
    list contains the five fields the runtime emits, and the default
    prompt declares the four placeholders the runtime renders

The real ``generate_debrief`` call requires a loaded model and is
exercised in ``notebooks/phase4/step09_debrief.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.debrief import (
    DEFAULT_DEBRIEF_PROMPT_NAME,
    RETRY_SAMPLING,
    DebriefConfig,
    DebriefError,
    _APOLOGY_CUE_RE,
    _coerce_debrief_payload,
    _coerce_ground_lost,
    _coerce_missed_openings,
    _coerce_over_apologies,
    _coerce_turn,
    _coerce_wins,
    build_debrief_messages,
    count_user_turns,
    format_practice_transcript,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "debrief.schema.json"
PROMPT_PATH = (
    REPO_ROOT / "data" / "prompts" / f"{DEFAULT_DEBRIEF_PROMPT_NAME}.md"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _five_turn_transcript() -> list[dict]:
    """Step-07-shaped transcript: 5 user turns alternating with 5 persona
    turns. Mirrors the practice round step07_persona_sim.ipynb produces."""
    user_lines = [
        "Hey Jamie, I want to talk about what happened with the report last week.",
        "Can we agree the staging data lands by Wednesday EOD this time?",
        "I'm not saying you didn't escalate — the escalation has to land.",
        "What if we set up a five-minute Slack check-in each morning?",
        "Thank you, that means a lot.",
    ]
    persona_lines = [
        ("I told you we were short on time.", "deflect", 0.60),
        ("Wednesday EOD is tight.", "counter_attack", 0.75),
        ("I told you, don't put this on me.", "guilt_trip", 0.70),
        ("A daily check-in sounds like a lot of overhead.", "deflect", 0.50),
        ("I can do that.", "concede", 0.40),
    ]
    transcript: list[dict] = []
    for user_text, (persona_text, rt, esc) in zip(user_lines, persona_lines):
        transcript.append({"speaker": "user", "text": user_text})
        transcript.append(
            {
                "speaker": "persona",
                "persona_name": "Jamie",
                "reply": persona_text,
                "resistance_type": rt,
                "escalation_level": esc,
            }
        )
    return transcript


def _full_debrief_payload(**overrides) -> dict:
    base = {
        "ground_lost": [
            {
                "turn": 3,
                "quote": "I'm not saying you didn't escalate — the escalation has to land.",
                "reason": "Softened the original ask before re-anchoring on the missed deadline.",
            },
        ],
        "over_apologies": [],
        "missed_openings": [
            {
                "turn": 2,
                "description": "Did not name Jamie's 'I told you' deflection when she used it.",
                "better_line": "Jamie, 'I told you' is the same deflection we keep hitting — can we set that aside and lock the Wednesday date?",
            },
        ],
        "wins": [
            {
                "turn": 4,
                "description": "Proposed a concrete next-step matching Jamie's responds_best_to pattern.",
            },
        ],
        "one_fix_next_time": "Re-anchor on the original commitment every time Jamie pivots to past communication.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# count_user_turns
# ---------------------------------------------------------------------------


def test_count_user_turns_counts_only_user_speaker():
    transcript = _five_turn_transcript()
    assert count_user_turns(transcript) == 5


def test_count_user_turns_handles_empty_and_non_list():
    assert count_user_turns([]) == 0
    assert count_user_turns(None) == 0
    assert count_user_turns("not a list") == 0


def test_count_user_turns_ignores_other_speakers():
    transcript = [
        {"speaker": "user", "text": "hi"},
        {"speaker": "other", "text": "hello"},
        {"speaker": "persona", "reply": "hmm"},
        {"speaker": "user", "text": "two"},
    ]
    assert count_user_turns(transcript) == 2


# ---------------------------------------------------------------------------
# format_practice_transcript
# ---------------------------------------------------------------------------


def test_format_practice_transcript_numbers_user_turns_in_order():
    text = format_practice_transcript(_five_turn_transcript())
    for i in range(1, 6):
        assert f"[USER {i}]" in text
    # Persona turns must NOT be numbered as USER N.
    assert "[USER 6]" not in text


def test_format_practice_transcript_labels_persona_with_metadata():
    text = format_practice_transcript(_five_turn_transcript())
    # The persona label includes the name plus resistance and escalation.
    assert "[Jamie (resistance=deflect, escalation=0.60)]" in text
    assert "[Jamie (resistance=concede, escalation=0.40)]" in text


def test_format_practice_transcript_labels_other_speaker_without_index():
    transcript = [
        {"speaker": "user", "text": "hello"},
        {"speaker": "other", "text": "world"},
        {"speaker": "user", "text": "again"},
    ]
    text = format_practice_transcript(transcript)
    assert "[USER 1] hello" in text
    assert "[OTHER] world" in text
    assert "[USER 2] again" in text


def test_format_practice_transcript_returns_sentinel_for_empty():
    assert "(empty transcript)" in format_practice_transcript([])
    assert "(empty transcript)" in format_practice_transcript(None)


def test_format_practice_transcript_skips_entries_with_no_text():
    transcript = [
        {"speaker": "user", "text": ""},
        {"speaker": "user", "text": "real one"},
        {"speaker": "persona", "reply": "   "},
        "not a dict",
    ]
    text = format_practice_transcript(transcript)
    assert "[USER 1] real one" in text
    assert "[USER 2]" not in text


# ---------------------------------------------------------------------------
# build_debrief_messages
# ---------------------------------------------------------------------------


def test_build_debrief_messages_single_user_turn():
    msgs = build_debrief_messages("INSTRUCTION")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "INSTRUCTION"


# ---------------------------------------------------------------------------
# _coerce_turn
# ---------------------------------------------------------------------------


def test_coerce_turn_clamps_into_user_turn_range():
    assert _coerce_turn(3, user_turn_count=5) == 3
    assert _coerce_turn(0, user_turn_count=5) == 1
    assert _coerce_turn(-1, user_turn_count=5) == 1
    assert _coerce_turn(99, user_turn_count=5) == 5


def test_coerce_turn_coerces_numeric_strings_and_floats():
    assert _coerce_turn("2", user_turn_count=5) == 2
    assert _coerce_turn(2.0, user_turn_count=5) == 2
    assert _coerce_turn("2.7", user_turn_count=5) == 2


def test_coerce_turn_returns_none_for_unrecoverable_values():
    assert _coerce_turn("not a number", user_turn_count=5) is None
    assert _coerce_turn(None, user_turn_count=5) is None
    assert _coerce_turn(2, user_turn_count=0) is None


# ---------------------------------------------------------------------------
# Per-array coercers
# ---------------------------------------------------------------------------


def test_coerce_ground_lost_drops_entries_with_empty_strings():
    raw = [
        {"turn": 1, "quote": "I'm sorry", "reason": "preemptive apology"},
        {"turn": 2, "quote": "  ", "reason": "still empty"},
        {"turn": 3, "quote": "valid", "reason": "  "},
    ]
    out = _coerce_ground_lost(raw, user_turn_count=5)
    assert len(out) == 1
    assert out[0]["turn"] == 1
    assert out[0]["quote"] == "I'm sorry"


def test_coerce_ground_lost_drops_entries_with_bad_turn():
    raw = [{"turn": "garbage", "quote": "q", "reason": "r"}]
    assert _coerce_ground_lost(raw, user_turn_count=5) == []


def test_coerce_ground_lost_caps_at_max_list_items():
    """One entry per distinct turn (dedup); the cap kicks in across
    turns, not within a turn. With 20 distinct turns supplied, the
    cap clips to 8."""
    raw = [
        {"turn": (i % 5) + 1, "quote": f"quote {i}", "reason": f"reason {i}"}
        for i in range(20)
    ]
    # 5 distinct turns supplied repeatedly -> 5 after dedup.
    assert len(_coerce_ground_lost(raw, user_turn_count=5)) == 5
    # 12 distinct turns supplied -> capped at 8.
    raw_wide = [
        {"turn": i + 1, "quote": f"q{i}", "reason": f"r{i}"}
        for i in range(12)
    ]
    assert len(_coerce_ground_lost(raw_wide, user_turn_count=20)) == 8


def test_coerce_ground_lost_dedupes_same_turn_keeping_first():
    """Run 1 of Step 9 produced two ground_lost entries for the same
    USER 3 turn — a full-turn quote and a fragment-of-the-same-turn
    quote. Same loss attributed twice. Runtime keeps the first entry
    per turn and drops the rest."""
    raw = [
        {"turn": 3, "quote": "first take on turn 3", "reason": "the stronger reason"},
        {"turn": 3, "quote": "fragment of turn 3", "reason": "a redundant second pass"},
        {"turn": 4, "quote": "different turn", "reason": "kept"},
    ]
    out = _coerce_ground_lost(raw, user_turn_count=5)
    assert [e["turn"] for e in out] == [3, 4]
    assert out[0]["quote"] == "first take on turn 3"


def test_coerce_over_apologies_keeps_entries_with_apology_cues():
    """Quotes containing canonical apology cues survive the cue check."""
    raw = [
        {"turn": 1, "quote": "Sorry, I should've caught that email earlier."},
        {"turn": 2, "quote": "My bad on the missed deadline."},
        {"turn": 3, "quote": "I shouldn't have brought up old commitments like that."},
        {"turn": 4, "quote": "That was on me, honestly."},
        {"turn": 5, "quote": "I messed up the timing here."},
    ]
    out = _coerce_over_apologies(raw, user_turn_count=5)
    assert [e["turn"] for e in out] == [1, 2, 3, 4, 5]


def test_coerce_over_apologies_drops_gratitude_and_agreement():
    """Step 9 Run 1 mislabelled gratitude (`"Thank you, that means a lot."`)
    as an over-apology on both ``enable_thinking`` branches despite the
    prompt naming the cues explicitly. The runtime now enforces the
    cue list in code so gratitude / agreement / de-escalation moves
    cannot leak through."""
    raw = [
        {"turn": 5, "quote": "Thank you, that means a lot."},
        {"turn": 4, "quote": "Fair point, you're right about that."},
        {"turn": 3, "quote": "I'm not blaming you — I'm trying to understand."},
        {"turn": 2, "quote": "Sorry, my bad on that one."},  # this one is a real apology
    ]
    out = _coerce_over_apologies(raw, user_turn_count=5)
    assert [e["turn"] for e in out] == [2]
    assert out[0]["quote"].startswith("Sorry")


def test_coerce_over_apologies_keeps_turn_and_quote_only():
    raw = [
        {"turn": 2, "quote": "Sorry, that was on me."},
        {"turn": 1},  # missing quote -> drop
    ]
    out = _coerce_over_apologies(raw, user_turn_count=5)
    assert len(out) == 1
    assert set(out[0].keys()) == {"turn", "quote"}


def test_apology_cue_regex_recognises_each_documented_cue():
    """The prompt names a specific list of apology cues
    (``sorry`` / ``apolog…`` / ``my bad`` / ``my fault`` /
    ``I shouldn't have`` / ``I messed up`` / ``that was on me``).
    The runtime regex must match every cue the prompt promises to
    accept, otherwise model outputs that follow the prompt would be
    dropped by the runtime — silent contract divergence."""
    documented_cues = (
        "I'm sorry about that.",
        "I apologise for the delay.",
        "My bad, I dropped the ball.",
        "That was my fault.",
        "I shouldn't have escalated like that.",
        "I shouldnt have brought it up.",  # missing apostrophe
        "I messed up the handoff.",
        "I screwed up the timing.",
        "That was on me.",
        "That is on me.",
        "That's on me.",
    )
    for quote in documented_cues:
        assert _APOLOGY_CUE_RE.search(quote), (
            f"apology cue regex should match documented cue: {quote!r}"
        )


def test_apology_cue_regex_rejects_non_apologies():
    """Gratitude, agreement, and de-escalation moves are NOT apologies
    and must not match the cue regex."""
    non_apologies = (
        "Thank you, that means a lot.",
        "Fair point, you're right.",
        "I appreciate your patience.",
        "I'm not blaming you.",
        "Let's solve this together.",
    )
    for quote in non_apologies:
        assert not _APOLOGY_CUE_RE.search(quote), (
            f"apology cue regex should NOT match: {quote!r}"
        )


def test_coerce_missed_openings_requires_description_and_better_line():
    raw = [
        {
            "turn": 4,
            "description": "Did not name the deflection.",
            "better_line": "Jamie, that's the same deflection we keep hitting.",
        },
        {"turn": 5, "description": "missing better_line"},
        {"turn": 5, "description": "", "better_line": "non-empty but description empty"},
    ]
    out = _coerce_missed_openings(raw, user_turn_count=5)
    assert len(out) == 1
    assert out[0]["turn"] == 4
    assert "deflection" in out[0]["description"]
    assert out[0]["better_line"].startswith("Jamie")


def test_coerce_wins_drops_entries_without_description():
    raw = [
        {"turn": 4, "description": "Proposed a concrete next step."},
        {"turn": 4, "description": "   "},
    ]
    out = _coerce_wins(raw, user_turn_count=5)
    assert len(out) == 1
    assert out[0]["description"].startswith("Proposed")


def test_coerce_array_non_list_returns_empty():
    assert _coerce_ground_lost({"oops": "dict"}, user_turn_count=5) == []
    assert _coerce_over_apologies(None, user_turn_count=5) == []
    assert _coerce_missed_openings("string", user_turn_count=5) == []
    assert _coerce_wins(42, user_turn_count=5) == []


# ---------------------------------------------------------------------------
# _coerce_debrief_payload
# ---------------------------------------------------------------------------


def test_coerce_payload_happy_path():
    result = _coerce_debrief_payload(_full_debrief_payload(), user_turn_count=5)
    assert result["ground_lost"][0]["turn"] == 3
    assert result["over_apologies"] == []
    assert result["missed_openings"][0]["turn"] == 2
    assert result["wins"][0]["turn"] == 4
    assert result["one_fix_next_time"].startswith("Re-anchor")
    assert result["debrief_id"].startswith("debrief_")
    assert "T" in result["generated_at"]  # ISO 8601 marker


def test_coerce_payload_clamps_out_of_range_turns():
    payload = _full_debrief_payload(
        ground_lost=[
            {"turn": 99, "quote": "q", "reason": "r"},
            {"turn": 0, "quote": "q2", "reason": "r2"},
        ]
    )
    result = _coerce_debrief_payload(payload, user_turn_count=5)
    turns = [e["turn"] for e in result["ground_lost"]]
    assert turns == [5, 1]


def test_coerce_payload_rejects_empty_one_fix_next_time():
    payload = _full_debrief_payload(one_fix_next_time="")
    with pytest.raises(DebriefError, match="one_fix_next_time"):
        _coerce_debrief_payload(payload, user_turn_count=5)


def test_coerce_payload_rejects_missing_one_fix_next_time():
    payload = _full_debrief_payload()
    del payload["one_fix_next_time"]
    with pytest.raises(DebriefError, match="one_fix_next_time"):
        _coerce_debrief_payload(payload, user_turn_count=5)


def test_coerce_payload_rejects_non_dict_top_level():
    with pytest.raises(DebriefError, match="not an object"):
        _coerce_debrief_payload(["not", "a", "dict"], user_turn_count=5)


def test_coerce_payload_returns_empty_arrays_when_input_arrays_missing():
    """Same-shape contract: a real debrief can legitimately have 0 wins
    or 0 over_apologies. The runtime should not invent entries — empty
    arrays are valid."""
    minimal = {"one_fix_next_time": "Slow down before responding."}
    result = _coerce_debrief_payload(minimal, user_turn_count=5)
    assert result["ground_lost"] == []
    assert result["over_apologies"] == []
    assert result["missed_openings"] == []
    assert result["wins"] == []
    assert result["one_fix_next_time"] == "Slow down before responding."


# ---------------------------------------------------------------------------
# Schema + prompt contract
# ---------------------------------------------------------------------------


def test_debrief_schema_required_top_level_fields_match_runtime_output():
    """The runtime emits the five required keys plus runtime-attached
    debrief_id / generated_at. Any drift in the schema's ``required``
    list means a generated debrief could fail downstream validators."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "Debrief"
    required = set(schema["required"])
    runtime_required = {
        "ground_lost",
        "over_apologies",
        "missed_openings",
        "wins",
        "one_fix_next_time",
    }
    assert required == runtime_required


def test_debrief_prompt_has_all_placeholders():
    """The default prompt file must exist and declare exactly the four
    placeholders the runtime renders. Drift breaks ``Prompt.render``."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for placeholder in (
        "$user_goal",
        "$person_profile_block",
        "$talk_dna_block",
        "$transcript",
    ):
        assert placeholder in text, f"prompt missing {placeholder}"


def test_debrief_config_defaults_match_documented_runtime():
    """Defaults are part of the public contract — guard against
    accidental flips to ``enable_thinking=True`` or a non-default
    prompt name."""
    cfg = DebriefConfig()
    assert cfg.prompt_name == DEFAULT_DEBRIEF_PROMPT_NAME
    assert cfg.enable_thinking is False
    assert cfg.transcript == []
    assert cfg.max_new_tokens >= 512  # debrief JSON body is non-trivial


# ---------------------------------------------------------------------------
# Retry-sampling configuration
# ---------------------------------------------------------------------------


def test_retry_sampling_is_low_variance():
    """Mirror the rationale from the other intelligence runtimes — just
    enough randomness to escape a single bad greedy trajectory."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1
