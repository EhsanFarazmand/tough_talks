"""Unit tests for the emotion-radar runtime.

Covers everything that can be validated without loading transformers
or a real audio file:

  - the chat-template message shape produced by
    :func:`build_emotion_messages` (audio comes before text;
    same structure as transcription so the multimodal path is
    consistent across phase 2 components)
  - ``_coerce_emotion_payload``'s contract: validates the primary-
    emotion enum, clamps numeric fields, propagates whisper /
    escalation strings, attaches runtime metadata (turn_id, speaker,
    timestamp), and tolerates missing optional fields
  - the runtime's ``ALLOWED_EMOTIONS`` constant is kept in lockstep
    with the schema enum (single source of truth check)

The actual ``analyze_emotion`` call requires a loaded multimodal model
and is exercised in
``notebooks/phase2/step04_emotion_radar.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime.emotion import (
    ALLOWED_EMOTIONS,
    ALLOWED_SPEAKERS,
    DEFAULT_EMOTION_PROMPT_NAME,
    DEFAULT_PRIOR_CONTEXT_HISTORY,
    EmotionAnalysisError,
    EmotionConfig,
    RETRY_SAMPLING,
    _coerce_emotion_payload,
    build_emotion_messages,
    format_emotion_context,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "emotion_radar.schema.json"
PROMPT_PATH = REPO_ROOT / "data" / "prompts" / f"{DEFAULT_EMOTION_PROMPT_NAME}.md"


def _minimal_emotions(**overrides):
    base = {
        "primary": "neutral",
        "intensity": 0.5,
        "tension_level": 0.5,
        "defensive": False,
        "concession_made": False,
        "escalation_risk": 0.2,
    }
    base.update(overrides)
    return base


def _minimal_payload(**overrides):
    base = {
        "transcript_snippet": "I hear you.",
        "emotions": _minimal_emotions(),
        "whisper_prompt": None,
        "escalation_alert": None,
    }
    base.update(overrides)
    return base


# ---- message shape ------------------------------------------------------


def test_build_messages_audio_before_text():
    """Gemma 4 docs require audio content before text in the prompt."""
    msgs = build_emotion_messages("/tmp/clip.wav", "analyze emotion")
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content[0]["type"] == "audio"
    assert content[0]["audio"] == "/tmp/clip.wav"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "analyze emotion"


def test_build_messages_role_is_user():
    msgs = build_emotion_messages("clip.wav", "x")
    assert msgs[0]["role"] == "user"


def test_build_messages_coerces_path_to_str():
    msgs = build_emotion_messages(Path("/tmp/clip.wav"), "x")
    assert isinstance(msgs[0]["content"][0]["audio"], str)


# ---- coercion: happy path -----------------------------------------------


def test_coerce_returns_expected_top_level_keys():
    result = _coerce_emotion_payload(_minimal_payload(), cfg=EmotionConfig())
    for key in ("turn_id", "speaker", "emotions", "timestamp"):
        assert key in result, f"missing required key: {key}"


def test_coerce_attaches_speaker_from_cfg():
    result = _coerce_emotion_payload(
        _minimal_payload(), cfg=EmotionConfig(speaker="other")
    )
    assert result["speaker"] == "other"


def test_coerce_attaches_explicit_turn_id():
    result = _coerce_emotion_payload(
        _minimal_payload(), cfg=EmotionConfig(turn_id="turn_42")
    )
    assert result["turn_id"] == "turn_42"


def test_coerce_generates_turn_id_when_missing():
    result = _coerce_emotion_payload(_minimal_payload(), cfg=EmotionConfig())
    assert isinstance(result["turn_id"], str)
    assert result["turn_id"].startswith("turn_")


def test_coerce_preserves_all_emotion_subfields():
    payload = _minimal_payload(
        emotions=_minimal_emotions(
            primary="anger",
            intensity=0.8,
            tension_level=0.75,
            defensive=True,
            concession_made=False,
            escalation_risk=0.65,
        )
    )
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig())
    e = result["emotions"]
    assert e["primary"] == "anger"
    assert e["intensity"] == 0.8
    assert e["tension_level"] == 0.75
    assert e["defensive"] is True
    assert e["concession_made"] is False
    assert e["escalation_risk"] == 0.65


def test_coerce_clamps_out_of_range_numerics():
    payload = _minimal_payload(
        emotions=_minimal_emotions(
            intensity=1.5, tension_level=-0.3, escalation_risk=99.0
        )
    )
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig())
    assert result["emotions"]["intensity"] == 1.0
    assert result["emotions"]["tension_level"] == 0.0
    assert result["emotions"]["escalation_risk"] == 1.0


def test_coerce_propagates_whisper_and_alert_strings():
    payload = _minimal_payload(
        whisper_prompt="Slow your breathing. Anchor on the data.",
        escalation_alert="Take a 30-second pause before responding.",
    )
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig())
    assert result["whisper_prompt"].startswith("Slow")
    assert result["escalation_alert"].startswith("Take")


def test_coerce_drops_whisper_when_speaker_is_other():
    """Tough Talks policy: never coach the counterparty. Step 04 Colab
    runs showed the model emits whisper_prompt for `other` despite the
    prompt forbidding it, so the runtime enforces this in code."""
    payload = _minimal_payload(
        whisper_prompt="Acknowledge their feeling before defending.",
        escalation_alert="Validate before re-engaging.",
    )
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig(speaker="other"))
    assert result["whisper_prompt"] is None
    # escalation_alert is still relevant for `other` turns — it's an
    # advisory to the user about the conversation state, not coaching for
    # the counterparty — so it MUST survive the override.
    assert result["escalation_alert"].startswith("Validate")


def test_coerce_keeps_whisper_for_user_speaker():
    payload = _minimal_payload(whisper_prompt="Hold your ground on the number.")
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig(speaker="user"))
    assert result["whisper_prompt"] == "Hold your ground on the number."


def test_coerce_empty_strings_become_null():
    """Coaching strings that come back as empty/whitespace mean 'nothing
    to say' — we normalise those to null to match the schema's intent."""
    payload = _minimal_payload(whisper_prompt="   ", escalation_alert="")
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig())
    assert result["whisper_prompt"] is None
    assert result["escalation_alert"] is None


def test_coerce_uses_cfg_transcript_snippet_when_provided():
    """Caller-supplied snippet (e.g. from Step 03's transcript) wins
    over whatever the model echoed back."""
    payload = _minimal_payload(transcript_snippet="from model")
    cfg = EmotionConfig(transcript_snippet="from caller")
    assert (
        _coerce_emotion_payload(payload, cfg=cfg)["transcript_snippet"]
        == "from caller"
    )


def test_coerce_falls_back_to_payload_transcript_snippet():
    payload = _minimal_payload(transcript_snippet="from model")
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig())
    assert result["transcript_snippet"] == "from model"


def test_coerce_omits_transcript_when_neither_present():
    payload = _minimal_payload()
    del payload["transcript_snippet"]
    result = _coerce_emotion_payload(payload, cfg=EmotionConfig())
    assert "transcript_snippet" not in result


# ---- coercion: error paths ----------------------------------------------


def test_coerce_rejects_invalid_primary():
    payload = _minimal_payload(emotions=_minimal_emotions(primary="rage"))
    with pytest.raises(EmotionAnalysisError):
        _coerce_emotion_payload(payload, cfg=EmotionConfig())


def test_coerce_rejects_invalid_speaker():
    with pytest.raises(EmotionAnalysisError):
        _coerce_emotion_payload(
            _minimal_payload(), cfg=EmotionConfig(speaker="bystander")
        )


def test_coerce_rejects_non_dict_payload():
    with pytest.raises(EmotionAnalysisError):
        _coerce_emotion_payload("not a dict", cfg=EmotionConfig())


def test_coerce_rejects_missing_emotions_object():
    payload = _minimal_payload()
    del payload["emotions"]
    with pytest.raises(EmotionAnalysisError):
        _coerce_emotion_payload(payload, cfg=EmotionConfig())


def test_coerce_rejects_non_numeric_intensity():
    payload = _minimal_payload(
        emotions=_minimal_emotions(intensity="very high")
    )
    with pytest.raises(EmotionAnalysisError):
        _coerce_emotion_payload(payload, cfg=EmotionConfig())


# ---- schema + prompt contract ------------------------------------------


def test_allowed_emotions_match_schema_enum():
    """The runtime constant must equal the on-disk enum. Drift here
    means the model would be allowed to emit a label the frontend can't
    render — failing this test loudly is the whole point."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(schema["properties"]["emotions"]["properties"]["primary"]["enum"])
    assert set(ALLOWED_EMOTIONS) == enum


def test_allowed_speakers_match_schema_enum():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(schema["properties"]["speaker"]["enum"])
    assert set(ALLOWED_SPEAKERS) == enum


def test_emotion_radar_schema_required_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "EmotionRadarResult"
    for field in ("turn_id", "speaker", "emotions", "timestamp"):
        assert field in schema["required"], f"{field!r} must be required"


def test_emotion_radar_prompt_has_speaker_and_context_placeholders():
    """The default prompt file must exist and accept exactly the two
    placeholders the runtime renders: ``$speaker`` (whose voice this is)
    and ``$prior_context`` (rolling summary of earlier turns). Drift
    here breaks ``Prompt.render`` because it strictly checks both
    missing and extra placeholders."""
    assert PROMPT_PATH.is_file(), f"missing prompt: {PROMPT_PATH}"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "$speaker" in text
    assert "$prior_context" in text


# ---- retry-sampling configuration ---------------------------------------


def test_retry_sampling_is_low_variance():
    """The retry path uses light sampling — just enough to escape a single
    bad greedy trajectory. If these get bumped too high we lose the
    determinism advantage entirely; pin sane defaults."""
    assert RETRY_SAMPLING["temperature"] <= 0.5
    assert 0.5 < RETRY_SAMPLING["top_p"] <= 1.0
    assert RETRY_SAMPLING["top_k"] >= 1


# ---- format_emotion_context (rolling-context renderer) -------------------


def _sample_result(
    turn_id="turn_01",
    speaker="user",
    primary="frustration",
    intensity=0.7,
    tension=0.6,
    escalation=0.5,
    defensive=False,
    concession=False,
    snippet="I asked for the report on Monday and it's already Thursday.",
):
    return {
        "turn_id": turn_id,
        "speaker": speaker,
        "emotions": {
            "primary": primary,
            "intensity": intensity,
            "tension_level": tension,
            "escalation_risk": escalation,
            "defensive": defensive,
            "concession_made": concession,
        },
        "transcript_snippet": snippet,
    }


def test_format_emotion_context_empty_returns_sentinel():
    """First turn / chunk has no history — caller never has to special-
    case this; we render a deterministic placeholder string instead."""
    text = format_emotion_context([])
    assert "first" in text.lower()
    assert "$" not in text  # never accidentally render a Template token


def test_format_emotion_context_includes_primary_and_speaker():
    text = format_emotion_context([_sample_result(primary="anger", speaker="user")])
    assert "anger" in text
    assert "user" in text


def test_format_emotion_context_includes_numerics():
    text = format_emotion_context(
        [_sample_result(intensity=0.85, tension=0.9, escalation=0.75)]
    )
    assert "0.85" in text
    assert "0.90" in text
    assert "0.75" in text


def test_format_emotion_context_marks_defensive_and_concession_flags():
    text = format_emotion_context(
        [_sample_result(defensive=True, concession=True)]
    )
    assert "defensive" in text
    assert "concession_made" in text


def test_format_emotion_context_omits_flags_when_false():
    text = format_emotion_context(
        [_sample_result(defensive=False, concession=False)]
    )
    assert "defensive" not in text
    assert "concession_made" not in text


def test_format_emotion_context_includes_snippet_when_present():
    text = format_emotion_context([_sample_result(snippet="hello world")])
    assert "hello world" in text


def test_format_emotion_context_omits_snippet_line_when_empty():
    text = format_emotion_context([_sample_result(snippet="")])
    assert "said:" not in text


def test_format_emotion_context_truncates_long_snippets():
    """Long transcripts don't add signal proportional to their tokens —
    the renderer caps each snippet so a multi-turn context doesn't blow
    out the prompt budget."""
    long = "a" * 1000
    text = format_emotion_context([_sample_result(snippet=long)])
    # The renderer keeps a margin of ~200 chars per snippet plus a
    # trailing ellipsis; verify it truncated and didn't paste the whole
    # 1000-char string.
    assert "aaa" in text
    assert long not in text
    assert "…" in text


def test_format_emotion_context_respects_max_history():
    """Only the most-recent ``max_history`` results should appear, in
    oldest-first order — older turns drop off when the buffer is full."""
    results = [
        _sample_result(turn_id=f"turn_{i:02d}", primary=p)
        for i, p in enumerate(
            ["anger", "fear", "sadness", "joy", "frustration", "neutral"], start=1
        )
    ]
    text = format_emotion_context(results, max_history=3)
    # Last three: joy / frustration / neutral — should all be present.
    assert "joy" in text
    assert "frustration" in text
    assert "neutral" in text
    # Earliest two should NOT be in the rendered window.
    assert "anger" not in text
    assert "fear" not in text


def test_format_emotion_context_zero_history_returns_sentinel():
    """max_history=0 means "don't include any context" — same observable
    behaviour as an empty list."""
    text = format_emotion_context(
        [_sample_result()], max_history=0
    )
    assert "first" in text.lower()


def test_default_prior_context_history_is_reasonable():
    """If this changes we want a deliberate decision, not silent drift."""
    assert 1 <= DEFAULT_PRIOR_CONTEXT_HISTORY <= 10


def test_format_emotion_context_handles_missing_fields_gracefully():
    """Defensive: external callers might pass dicts that don't quite
    match the full schema (e.g. an early prototype). The renderer
    should fall back to sensible defaults rather than crash."""
    sparse = {"speaker": "user", "emotions": {"primary": "neutral"}}
    text = format_emotion_context([sparse])
    assert "neutral" in text
    assert "user" in text


# ---- prior_context wiring through _coerce_emotion_payload ----------------


def test_coerce_payload_does_not_leak_prior_context_into_result():
    """``prior_context`` is an input to the prompt, never an output field
    on the EmotionRadarResult. Make sure adding it to the cfg doesn't
    accidentally appear in the returned dict."""
    cfg = EmotionConfig(speaker="user", prior_context="some prior summary")
    result = _coerce_emotion_payload(_minimal_payload(), cfg=cfg)
    assert "prior_context" not in result
