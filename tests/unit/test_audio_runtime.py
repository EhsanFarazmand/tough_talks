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
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_INSTRUCTION,
    MAX_AUDIO_SECONDS,
    _extract_text,
    build_transcription_messages,
    compute_chunk_windows,
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


def test_transcription_schema_has_segments_field():
    """Long-audio mode adds an optional segments array."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert "segments" in schema["properties"]
    assert "segments" not in schema["required"]  # optional
    seg = schema["properties"]["segments"]
    assert seg["type"] == "array"
    item = seg["items"]
    assert set(item["required"]) == {"start_seconds", "end_seconds", "transcript"}


def test_chunk_window_safety_margin():
    """Default chunk size must stay under the hard 30 s Gemma 4 cap."""
    assert DEFAULT_CHUNK_SECONDS < MAX_AUDIO_SECONDS


def test_compute_chunks_short_clip_returns_one_window():
    """Clips that already fit shouldn't be split."""
    windows = compute_chunk_windows(10.0, max_chunk_seconds=28.0)
    assert windows == [(0.0, 10.0)]


def test_compute_chunks_exact_cap_returns_one_window():
    """A clip equal to max_chunk_seconds is still one window."""
    windows = compute_chunk_windows(28.0, max_chunk_seconds=28.0)
    assert windows == [(0.0, 28.0)]


def test_compute_chunks_covers_full_clip():
    """The last window must end exactly at total_seconds — no audio lost."""
    total = 42.0
    windows = compute_chunk_windows(total, max_chunk_seconds=28.0, overlap_seconds=0.5)
    assert len(windows) >= 2
    assert windows[0][0] == 0.0
    assert windows[-1][1] == total
    for start, end in windows:
        assert end - start <= 28.0 + 1e-9


def test_compute_chunks_have_overlap():
    """Consecutive windows should overlap by overlap_seconds."""
    windows = compute_chunk_windows(60.0, max_chunk_seconds=28.0, overlap_seconds=0.5)
    for prev, nxt in zip(windows, windows[1:]):
        # next.start should be earlier than prev.end (overlap), but only when
        # we're not on the final shortened window that just trims to total.
        if nxt[1] < 60.0:
            assert nxt[0] < prev[1]


def test_compute_chunks_zero_duration():
    """Defensive: zero-length audio returns no windows."""
    assert compute_chunk_windows(0.0) == []


def test_compute_chunks_rejects_oversized_overlap():
    """Overlap >= chunk size would loop forever — must error early."""
    import pytest

    with pytest.raises(ValueError):
        compute_chunk_windows(60.0, max_chunk_seconds=28.0, overlap_seconds=30.0)


def test_default_chunk_overlap_is_smaller_than_chunk_size():
    """Sanity check on the public default constants."""
    assert 0.0 <= DEFAULT_CHUNK_OVERLAP_SECONDS < DEFAULT_CHUNK_SECONDS
