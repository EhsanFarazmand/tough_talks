"""Unit tests for the audio transcription runtime.

Covers everything that can be validated without loading transformers
or a real audio file:

  - the chat-template message shape produced by
    :func:`build_transcription_messages` (audio comes before text;
    structure matches what Gemma 4's ``apply_chat_template`` expects)
  - the transcription schema is well-formed JSON with the keys the
    runtime promises to emit
  - ``_extract_text`` tolerates both the dict and bare-string return
    shapes of ``processor.parse_response``

The actual transcribe() call requires a loaded multimodal model and is
exercised in ``notebooks/phase2/step03_gemma4_audio_transcription.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.core._runtime.audio import (
    DEFAULT_INSTRUCTION,
    MAX_AUDIO_SECONDS,
    _extract_text,
    build_transcription_messages,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "transcription.schema.json"


def test_build_messages_audio_before_text():
    """Gemma 4 docs require audio content before text in the prompt."""
    msgs = build_transcription_messages("/tmp/clip.wav", DEFAULT_INSTRUCTION)
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content[0]["type"] == "audio"
    assert content[0]["audio"] == "/tmp/clip.wav"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == DEFAULT_INSTRUCTION


def test_build_messages_role_is_user():
    msgs = build_transcription_messages("clip.wav", "x")
    assert msgs[0]["role"] == "user"


def test_build_messages_coerces_path_to_str():
    msgs = build_transcription_messages(Path("/tmp/clip.wav"), "x")
    assert isinstance(msgs[0]["content"][0]["audio"], str)


def test_max_audio_seconds_is_30():
    """Hard cap from Gemma 4 audio context — surface it as a constant."""
    assert MAX_AUDIO_SECONDS == 30


def test_extract_text_handles_dict():
    assert _extract_text({"role": "assistant", "content": "hello"}) == "hello"


def test_extract_text_handles_bare_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_missing_content():
    assert _extract_text({"role": "assistant"}) == ""


def test_transcription_schema_required_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == "Transcription"
    assert schema["type"] == "object"
    for field in ("audio_source", "transcript", "model_id", "created_at"):
        assert field in schema["required"], f"{field!r} must be required"
        assert field in schema["properties"]


def test_transcription_schema_no_extra_fields():
    """additionalProperties: false keeps the contract with the frontend tight."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema.get("additionalProperties") is False
