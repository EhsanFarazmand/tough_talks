"""Unit tests for the Tough Talks FastAPI surface.

Phase 5 / Step 12. Covers everything that can be exercised without
loading a real model:

  * lifespan-free :class:`TestClient` boot.
  * /health surfaces model-loaded flags from the injected
    :class:`backend.api.deps.ModelRegistry`.
  * Every component route validates its request body (Pydantic) and
    routes to the correct runtime entry point with the runtime config
    populated from the request fields. The runtime functions themselves
    are monkeypatched to canned payloads / canned exceptions so the
    test exercises the HTTP / Pydantic / config-translation layer only.
  * Runtime errors (``XxxError`` subclasses of ``ValueError``) map to
    HTTP 422 with the diagnostic ``attempts`` list surfaced in the
    response body.
  * Missing-registry / wrong-variant calls map to HTTP 503.
  * Audio routes (transcription, emotion_radar) accept
    ``multipart/form-data`` with an ``UploadFile`` and pass the saved
    temp-file path to the runtime.

The real generate_*/analyze_* calls require a loaded model and are
exercised in ``notebooks/phase5/step12_fastapi.ipynb``.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import ModelRegistry, get_registry
from backend.api.main import app
from backend.core._runtime import (
    AftermathError,
    DebriefError,
    EmotionAnalysisError,
    PersonaReplyError,
    PersonVaultAnalysisError,
    PremortemError,
    PulseError,
    TalkDNAAnalysisError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_only_registry() -> ModelRegistry:
    """Registry with text-only sentinel objects — no multimodal pair."""
    return ModelRegistry(
        text_processor=object(),
        text_model=object(),
    )


@pytest.fixture
def full_registry() -> ModelRegistry:
    """Registry with both text-only and multimodal sentinel objects."""
    return ModelRegistry(
        text_processor=object(),
        text_model=object(),
        multimodal_processor=object(),
        multimodal_model=object(),
    )


@pytest.fixture
def client(full_registry: ModelRegistry) -> Iterator[TestClient]:
    """TestClient with both model variants injected via dependency override.

    Constructed without the ``with`` block so the lifespan event does
    NOT run — the registry comes from the override, not from a real
    model load. Cleanup wipes the override so the next test starts
    fresh.
    """
    app.dependency_overrides[get_registry] = lambda: full_registry
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_registry, None)


@pytest.fixture
def text_only_client(text_only_registry: ModelRegistry) -> Iterator[TestClient]:
    """TestClient with only the text variant — audio routes will 503."""
    app.dependency_overrides[get_registry] = lambda: text_only_registry
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_registry, None)


@pytest.fixture
def no_registry_client() -> Iterator[TestClient]:
    """TestClient with no registry at all — every inference route 503s."""

    def _missing():
        from backend.api.deps import RegistryNotReady

        raise RegistryNotReady("test: no registry available")

    app.dependency_overrides[get_registry] = _missing
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_registry, None)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_both_models_loaded(client: TestClient):
    # /health doesn't go through get_registry — it reads app.state directly,
    # so we set the state on the actual app for this test.
    app.state.registry = ModelRegistry(
        text_processor=object(),
        text_model=object(),
        multimodal_processor=object(),
        multimodal_model=object(),
    )
    try:
        response = client.get("/health")
    finally:
        app.state.registry = None
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["text_model_loaded"] is True
    assert body["multimodal_model_loaded"] is True


def test_health_reports_missing_models_when_registry_absent(client: TestClient):
    # Default app.state.registry is None unless we set it explicitly.
    app.state.registry = None
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["text_model_loaded"] is False
    assert body["multimodal_model_loaded"] is False


# ---------------------------------------------------------------------------
# /talk-dna/analyze
# ---------------------------------------------------------------------------


_TALK_DNA_OK = {
    "user_id": "local",
    "version": 1,
    "conversation_count": 1,
    "patterns": {
        "filler_phrases": [],
        "apology_rate": 0.0,
        "silence_under_pressure": False,
        "sarcasm_frequency": "low",
        "escalation_triggers": [],
        "avg_turn_length_words": 3.0,
    },
    "strengths": ["clear_problem_statement"],
    "weaknesses": [],
    "updated_at": "2026-05-15T00:00:00+00:00",
}


def test_talk_dna_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, turns, *, cfg):
        captured["turns"] = turns
        captured["cfg"] = cfg
        return _TALK_DNA_OK

    monkeypatch.setattr(
        "backend.api.routes.talk_dna.analyze_talk_dna", _stub
    )
    body = {
        "turns": [{"speaker": "user", "text": "I just feel like this is hard."}],
        "user_id": "solmaz",
        "max_new_tokens": 512,
        "prior_profile": None,
    }
    response = client.post("/talk-dna/analyze", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == _TALK_DNA_OK
    assert captured["cfg"].user_id == "solmaz"
    assert captured["cfg"].max_new_tokens == 512
    assert captured["turns"] == body["turns"]


def test_talk_dna_runtime_error_maps_to_422_with_attempts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    def _raise(*args, **kwargs):
        err = TalkDNAAnalysisError("bad model output")
        err.attempts = [{"sampling": "greedy", "error": "bad", "raw_head": "..."}]
        raise err

    monkeypatch.setattr(
        "backend.api.routes.talk_dna.analyze_talk_dna", _raise
    )
    response = client.post(
        "/talk-dna/analyze",
        json={"turns": [{"speaker": "user", "text": "ok"}]},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["component"] == "talk_dna"
    assert "bad model output" in detail["error"]
    assert detail["attempts"][0]["sampling"] == "greedy"


def test_talk_dna_empty_turns_rejected_by_pydantic(client: TestClient):
    response = client.post("/talk-dna/analyze", json={"turns": []})
    assert response.status_code == 422
    # Pydantic surfaces the field-level error under `detail`; we just
    # need the validation to fire before any runtime call.


# ---------------------------------------------------------------------------
# /vault/build
# ---------------------------------------------------------------------------


_VAULT_OK = {
    "person_id": "person_abc",
    "name": "Jamie",
    "relationship_type": "colleague",
    "version": 1,
    "conversation_count": 1,
    "profile": {
        "communication_style": "defensive",
        "emotional_triggers": [],
        "de_escalation_keys": [],
        "common_deflections": [],
    },
    "updated_at": "2026-05-15T00:00:00+00:00",
}


def test_person_vault_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, turns, *, cfg):
        captured["cfg"] = cfg
        return _VAULT_OK

    monkeypatch.setattr(
        "backend.api.routes.person_vault.analyze_person_vault", _stub
    )
    body = {
        "turns": [{"speaker": "other", "text": "I told you the table wasn't ready."}],
        "name": "Jamie",
        "relationship_type": "colleague",
    }
    response = client.post("/vault/build", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == _VAULT_OK
    assert captured["cfg"].name == "Jamie"
    assert captured["cfg"].relationship_type == "colleague"


def test_person_vault_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.api.routes.person_vault.analyze_person_vault",
        lambda *a, **kw: (_ for _ in ()).throw(
            PersonVaultAnalysisError("style not in enum")
        ),
    )
    response = client.post(
        "/vault/build",
        json={"turns": [{"speaker": "other", "text": "x"}]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "person_vault"


# ---------------------------------------------------------------------------
# /persona/reply + /persona/run
# ---------------------------------------------------------------------------


_PERSONA_REPLY_OK = {
    "persona_name": "Jamie",
    "reply": "I told you the staging tables weren't done.",
    "resistance_type": "deflect",
    "escalation_level": 0.6,
    "turn_id": "persona_turn_abc",
    "timestamp": "2026-05-15T00:00:00+00:00",
}


def test_persona_reply_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, user_message, *, history, cfg):
        captured["user_message"] = user_message
        captured["history"] = history
        captured["cfg"] = cfg
        return _PERSONA_REPLY_OK

    monkeypatch.setattr(
        "backend.api.routes.persona_sim.generate_persona_reply", _stub
    )
    body = {
        "user_message": "I noticed Tuesday's deadline slipped — what's going on?",
        "persona_profile": {"name": "Jamie", "profile": {"communication_style": "defensive"}},
        "history": [],
        "user_goal": "Reset expectations and get Wednesday EOD on the calendar.",
        "enable_thinking": False,
    }
    response = client.post("/persona/reply", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == _PERSONA_REPLY_OK
    assert captured["user_message"] == body["user_message"]
    assert captured["cfg"].user_goal == body["user_goal"]
    assert captured["cfg"].persona_profile["name"] == "Jamie"


def test_persona_reply_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.api.routes.persona_sim.generate_persona_reply",
        lambda *a, **kw: (_ for _ in ()).throw(
            PersonaReplyError("reply is empty")
        ),
    )
    response = client.post(
        "/persona/reply",
        json={"user_message": "hi"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "persona_sim"


def test_persona_run_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, user_messages, *, cfg):
        captured["user_messages"] = user_messages
        captured["cfg"] = cfg
        return [
            {"speaker": "user", "text": user_messages[0]},
            {"speaker": "persona", **_PERSONA_REPLY_OK},
        ]

    monkeypatch.setattr(
        "backend.api.routes.persona_sim.run_practice_conversation", _stub
    )
    response = client.post(
        "/persona/run",
        json={
            "user_messages": ["Let's talk about Tuesday."],
            "persona_profile": {"name": "Jamie"},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "history" in body
    assert body["history"][0]["speaker"] == "user"
    assert body["history"][1]["resistance_type"] == "deflect"


# ---------------------------------------------------------------------------
# /premortem
# ---------------------------------------------------------------------------


_PREMORTEM_OK = {
    "goal": "Get Jamie to commit to Wednesday EOD.",
    "failure_scenarios": [
        {
            "scenario_id": i,
            "title": f"Scenario {i}",
            "description": "...",
            "likely_trigger": "...",
            "destabilization_risk": 0.5,
            "simulation_parameters": {
                "resistance_type": "deflect",
                "escalation_ceiling": 0.5,
                "opening_move": "...",
            },
        }
        for i in (1, 2, 3)
    ],
    "premortem_id": "premortem_abc",
    "generated_at": "2026-05-15T00:00:00+00:00",
}


def test_premortem_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, *, cfg):
        captured["cfg"] = cfg
        return _PREMORTEM_OK

    monkeypatch.setattr(
        "backend.api.routes.premortem.generate_premortem", _stub
    )
    response = client.post(
        "/premortem",
        json={
            "conversation_description": "Push back on the missed deadline.",
            "user_goal": "Wednesday EOD.",
            "enable_thinking": True,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == _PREMORTEM_OK
    assert captured["cfg"].enable_thinking is True
    assert captured["cfg"].conversation_description.startswith("Push back")


def test_premortem_empty_description_rejected(client: TestClient):
    response = client.post(
        "/premortem",
        json={"conversation_description": ""},
    )
    assert response.status_code == 422


def test_premortem_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.api.routes.premortem.generate_premortem",
        lambda *a, **kw: (_ for _ in ()).throw(
            PremortemError("scenarios count != 3")
        ),
    )
    response = client.post(
        "/premortem",
        json={"conversation_description": "x"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "premortem"


# ---------------------------------------------------------------------------
# /debrief
# ---------------------------------------------------------------------------


_DEBRIEF_OK = {
    "ground_lost": [],
    "over_apologies": [],
    "missed_openings": [],
    "wins": [],
    "one_fix_next_time": "Name the deflection out loud.",
    "debrief_id": "debrief_abc",
    "generated_at": "2026-05-15T00:00:00+00:00",
}


def test_debrief_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, *, cfg):
        captured["cfg"] = cfg
        return _DEBRIEF_OK

    monkeypatch.setattr(
        "backend.api.routes.debrief.generate_debrief", _stub
    )
    body = {
        "transcript": [
            {"speaker": "user", "text": "Let's talk about Tuesday."},
            {
                "speaker": "persona",
                "persona_name": "Jamie",
                "reply": "I told you...",
                "resistance_type": "deflect",
                "escalation_level": 0.5,
            },
        ],
        "user_goal": "Wednesday EOD.",
    }
    response = client.post("/debrief", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == _DEBRIEF_OK
    assert len(captured["cfg"].transcript) == 2


def test_debrief_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.api.routes.debrief.generate_debrief",
        lambda *a, **kw: (_ for _ in ()).throw(
            DebriefError("one_fix_next_time missing")
        ),
    )
    response = client.post(
        "/debrief",
        json={"transcript": [{"speaker": "user", "text": "x"}]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "debrief"


# ---------------------------------------------------------------------------
# /aftermath
# ---------------------------------------------------------------------------


_AFTERMATH_OK = {
    "goal_outcome": {"status": "partial", "summary": "Got some of it."},
    "scenario_outcomes": [],
    "unforeseen_moments": [],
    "prediction_accuracy": 0.6,
    "next_round_focus": "Soften less.",
    "aftermath_id": "aftermath_abc",
    "generated_at": "2026-05-15T00:00:00+00:00",
}


def test_aftermath_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, *, cfg):
        captured["cfg"] = cfg
        return _AFTERMATH_OK

    monkeypatch.setattr(
        "backend.api.routes.aftermath.generate_aftermath", _stub
    )
    response = client.post(
        "/aftermath",
        json={
            "premortem": {"failure_scenarios": [{}, {}, {}]},
            "transcript": [{"speaker": "user", "text": "Hi"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == _AFTERMATH_OK
    assert captured["cfg"].premortem == {"failure_scenarios": [{}, {}, {}]}


def test_aftermath_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.api.routes.aftermath.generate_aftermath",
        lambda *a, **kw: (_ for _ in ()).throw(
            AftermathError("scenarios != 3")
        ),
    )
    response = client.post(
        "/aftermath",
        json={
            "premortem": {},
            "transcript": [{"speaker": "user", "text": "x"}],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "aftermath"


# ---------------------------------------------------------------------------
# /pulse
# ---------------------------------------------------------------------------


_PULSE_OK = {
    "person_id": "person_abc",
    "person_name": "Jamie",
    "relationship_type": "colleague",
    "version": 1,
    "round_count": 2,
    "health_score": 0.6,
    "health_trend": "improving",
    "trajectory_summary": "...",
    "round_summaries": [],
    "recurring_patterns": [],
    "emerging_concerns": [],
    "relationship_wins": [],
    "next_step_recommendation": "...",
    "pulse_id": "pulse_abc",
    "updated_at": "2026-05-15T00:00:00+00:00",
}


def test_pulse_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, *, cfg):
        captured["cfg"] = cfg
        return _PULSE_OK

    monkeypatch.setattr(
        "backend.api.routes.pulse.generate_pulse", _stub
    )
    body = {
        "rounds": [
            {"round_id": "r1", "started_at": "2026-05-01T00:00:00+00:00"},
            {"round_id": "r2", "started_at": "2026-05-08T00:00:00+00:00"},
        ],
        "person_profile": {"person_id": "person_abc", "name": "Jamie"},
    }
    response = client.post("/pulse", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == _PULSE_OK
    assert captured["cfg"].enable_thinking is True  # the route default
    assert captured["cfg"].max_new_tokens == 4096


def test_pulse_min_rounds_rejected_by_pydantic(client: TestClient):
    response = client.post(
        "/pulse",
        json={
            "rounds": [{"round_id": "r1", "started_at": "x"}],
            "person_profile": {"person_id": "person_abc", "name": "Jamie"},
        },
    )
    assert response.status_code == 422


def test_pulse_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.api.routes.pulse.generate_pulse",
        lambda *a, **kw: (_ for _ in ()).throw(PulseError("bad trend")),
    )
    response = client.post(
        "/pulse",
        json={
            "rounds": [
                {"round_id": "r1", "started_at": "x"},
                {"round_id": "r2", "started_at": "y"},
            ],
            "person_profile": {"person_id": "person_abc", "name": "Jamie"},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "pulse"


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------


_TRANSCRIBE_OK = {
    "audio_source": "tmp.wav",
    "transcript": "Hello world.",
    "language": None,
    "duration_seconds": 1.0,
    "model_id": "google/gemma-4-E2B-it",
    "instruction": "...",
    "created_at": "2026-05-15T00:00:00+00:00",
}


def test_transcribe_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, audio_source, *, cfg, model_id=None, **kwargs):
        # The route saves the upload to a NamedTemporaryFile and passes
        # the path; we verify the file existed by reading it back here.
        captured["audio_source"] = audio_source
        captured["bytes"] = open(audio_source, "rb").read()
        captured["cfg"] = cfg
        captured["model_id"] = model_id
        return _TRANSCRIBE_OK

    monkeypatch.setattr(
        "backend.api.routes.transcription.transcribe_long", _stub
    )
    fake_audio = b"FAKE_WAV_BYTES"
    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", fake_audio, "audio/wav")},
        data={"max_new_tokens": "128", "language": "en"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == _TRANSCRIBE_OK
    assert captured["bytes"] == fake_audio
    assert captured["cfg"].max_new_tokens == 128
    assert captured["cfg"].language == "en"


def test_transcribe_503_on_missing_multimodal(
    text_only_client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    response = text_only_client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"FAKE", "audio/wav")},
    )
    assert response.status_code == 503
    assert "multimodal" in response.json()["detail"]["error"].lower()


# ---------------------------------------------------------------------------
# /emotion/analyze
# ---------------------------------------------------------------------------


_EMOTION_OK = {
    "turn_id": "turn_abc",
    "speaker": "user",
    "emotions": {
        "primary": "neutral",
        "intensity": 0.5,
        "tension_level": 0.5,
        "defensive": False,
        "concession_made": False,
        "escalation_risk": 0.2,
    },
    "whisper_prompt": None,
    "escalation_alert": None,
    "timestamp": "2026-05-15T00:00:00+00:00",
}


def test_emotion_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def _stub(processor, model, audio_source, *, cfg, **kwargs):
        captured["audio_source"] = audio_source
        captured["bytes"] = open(audio_source, "rb").read()
        captured["cfg"] = cfg
        return [_EMOTION_OK]

    monkeypatch.setattr(
        "backend.api.routes.emotion_radar.analyze_emotion_long", _stub
    )
    fake_audio = b"FAKE_WAV_BYTES_E"
    response = client.post(
        "/emotion/analyze",
        files={"audio": ("clip.wav", fake_audio, "audio/wav")},
        data={"speaker": "user", "turn_id": "t1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "results" in body
    assert body["results"] == [_EMOTION_OK]
    assert captured["bytes"] == fake_audio
    assert captured["cfg"].speaker == "user"
    assert captured["cfg"].turn_id == "t1"


def test_emotion_rejects_invalid_speaker(client: TestClient):
    response = client.post(
        "/emotion/analyze",
        files={"audio": ("clip.wav", b"x", "audio/wav")},
        data={"speaker": "narrator"},
    )
    assert response.status_code == 400


def test_emotion_runtime_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    def _raise(*args, **kwargs):
        raise EmotionAnalysisError("primary not in enum")

    monkeypatch.setattr(
        "backend.api.routes.emotion_radar.analyze_emotion_long", _raise
    )
    response = client.post(
        "/emotion/analyze",
        files={"audio": ("clip.wav", b"x", "audio/wav")},
        data={"speaker": "user"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["component"] == "emotion_radar"


# ---------------------------------------------------------------------------
# Registry failure modes
# ---------------------------------------------------------------------------


def test_text_route_503_when_no_registry(no_registry_client: TestClient):
    response = no_registry_client.post(
        "/talk-dna/analyze",
        json={"turns": [{"speaker": "user", "text": "hi"}]},
    )
    assert response.status_code == 503
    assert "no registry" in response.json()["detail"]["error"].lower()


def test_text_route_503_when_no_models_loaded():
    """Empty registry (no text, no multimodal) — every inference route
    503s. Same shape as a deployment that ran the lifespan with both
    include_text=False AND include_multimodal=False."""
    empty_registry = ModelRegistry()
    app.dependency_overrides[get_registry] = lambda: empty_registry
    try:
        client = TestClient(app)
        response = client.post(
            "/talk-dna/analyze",
            json={"turns": [{"speaker": "user", "text": "hi"}]},
        )
        assert response.status_code == 503
        assert "no model loaded" in response.json()["detail"]["error"].lower()
    finally:
        app.dependency_overrides.pop(get_registry, None)


def test_text_route_falls_back_to_multimodal_when_text_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Multimodal-only registry — text routes fall back to the multimodal
    pair. Same shape as the T4 / on-device deployment that loads only
    the multimodal variant to stay inside VRAM."""
    mm_processor = object()
    mm_model = object()
    multimodal_only = ModelRegistry(
        multimodal_processor=mm_processor,
        multimodal_model=mm_model,
    )
    app.dependency_overrides[get_registry] = lambda: multimodal_only

    captured: dict[str, Any] = {}

    def _stub(processor, model, turns, *, cfg):
        captured["processor"] = processor
        captured["model"] = model
        return _TALK_DNA_OK

    monkeypatch.setattr(
        "backend.api.routes.talk_dna.analyze_talk_dna", _stub
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/talk-dna/analyze",
            json={"turns": [{"speaker": "user", "text": "hi"}]},
        )
        assert response.status_code == 200, response.text
        # The fallback must hand the multimodal pair to the runtime.
        assert captured["processor"] is mm_processor
        assert captured["model"] is mm_model
    finally:
        app.dependency_overrides.pop(get_registry, None)
