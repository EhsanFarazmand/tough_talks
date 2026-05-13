"""Emotion radar via Gemma 4's native multimodal path.

Phase 2 / Step 4 of Tough Talks. Like ``audio.transcribe`` (Step 3) this
feeds raw audio + a text instruction directly to Gemma 4 E2B/E4B — no
separate emotion classifier — so the Live-Mode stack stays single-model
and prosody plus content are scored from the same activations.

The model is asked to emit a single JSON object describing the speaker's
emotion on this turn (primary label from a fixed enum, intensity,
tension, defensive flag, concession flag, escalation risk) plus optional
``whisper_prompt`` and ``escalation_alert`` coaching strings. The runtime
then attaches ``turn_id``, ``speaker``, and a UTC ``timestamp`` so the
result conforms to ``data/schemas/emotion_radar.schema.json``.

Same 30 s audio cap as transcription. Use :func:`analyze_emotion_long`
for clips longer than ``MAX_AUDIO_SECONDS`` — it shares the same chunking
helpers (28 s windows + 0.5 s overlap) as ``transcribe_long`` and emits
one EmotionRadarResult per chunk.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .audio import (
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    MAX_AUDIO_SECONDS,  # re-exported for callers that import from here
    _extract_text,
    chunk_audio,
)
from .parsing import parse_json
from .prompts import load_prompt

__all__ = [
    "ALLOWED_EMOTIONS",
    "ALLOWED_SPEAKERS",
    "DEFAULT_EMOTION_PROMPT_NAME",
    "EmotionAnalysisError",
    "EmotionConfig",
    "MAX_AUDIO_SECONDS",
    "analyze_emotion",
    "analyze_emotion_long",
    "build_emotion_messages",
]

LOG = logging.getLogger(__name__)

DEFAULT_EMOTION_PROMPT_NAME = "emotion_radar"

ALLOWED_SPEAKERS: tuple[str, ...] = ("user", "other")

# Mirrors the enum in data/schemas/emotion_radar.schema.json. Surfaced as
# a runtime constant so the prompt, the unit tests, and downstream
# consumers all reference the same list — one source of truth would be
# ideal but JSON Schema doesn't export to Python at import time without
# extra tooling, and a unit test pins the two together.
ALLOWED_EMOTIONS: tuple[str, ...] = (
    "anger",
    "fear",
    "sadness",
    "joy",
    "surprise",
    "disgust",
    "neutral",
    "frustration",
    "openness",
    "defensiveness",
)


class EmotionAnalysisError(ValueError):
    """The model emitted a malformed emotion-radar payload (missing
    fields, primary label not in the schema enum, etc.)."""


@dataclass
class EmotionConfig:
    """Run-time inputs for :func:`analyze_emotion`.

    Leave ``instruction`` as ``None`` to let the runtime load
    ``data/prompts/<prompt_name>.md`` and render it with the configured
    ``speaker``. Pass an explicit ``instruction`` to skip the prompt
    file (useful for ablation experiments).
    """

    max_new_tokens: int = 512
    instruction: Optional[str] = None
    speaker: str = "user"
    turn_id: Optional[str] = None
    transcript_snippet: Optional[str] = None
    prompt_name: str = DEFAULT_EMOTION_PROMPT_NAME


def build_emotion_messages(audio_source: str, instruction: str) -> list[dict]:
    """Construct the chat-template messages for an emotion-analysis call.

    Same shape as :func:`backend.core._runtime.audio.build_transcription_messages`:
    audio part first (Gemma 4's documented best practice), text
    instruction second, single user turn. Pure function — kept separate
    from :func:`analyze_emotion` so the message shape is unit-testable
    without loading transformers.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_source)},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def _resolve_instruction(cfg: EmotionConfig) -> str:
    if cfg.instruction is not None:
        return cfg.instruction
    return load_prompt(cfg.prompt_name).render(speaker=cfg.speaker)


def _clamp_unit(name: str, value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise EmotionAnalysisError(f"{name} not numeric: {value!r}") from exc
    return max(0.0, min(1.0, val))


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _coerce_emotion_payload(payload: Any, *, cfg: EmotionConfig) -> dict:
    """Turn the model's JSON output into a schema-conforming
    EmotionRadarResult dict.

    Validates the primary-emotion enum, clamps numeric fields to
    ``[0, 1]``, and attaches runtime-supplied metadata (turn_id,
    speaker, timestamp). Raises :class:`EmotionAnalysisError` for any
    missing/invalid field that can't be repaired without guessing —
    those should surface loudly so the notebook can show the bad output.
    """
    if cfg.speaker not in ALLOWED_SPEAKERS:
        raise EmotionAnalysisError(
            f"speaker must be one of {ALLOWED_SPEAKERS}, got {cfg.speaker!r}"
        )
    if not isinstance(payload, dict):
        raise EmotionAnalysisError(
            f"emotion payload not an object: {type(payload).__name__}"
        )

    emotions = payload.get("emotions")
    if not isinstance(emotions, dict):
        raise EmotionAnalysisError("payload missing 'emotions' object")

    primary = emotions.get("primary")
    if primary not in ALLOWED_EMOTIONS:
        raise EmotionAnalysisError(
            f"emotions.primary not in enum: {primary!r}; "
            f"allowed: {ALLOWED_EMOTIONS}"
        )

    intensity = _clamp_unit("emotions.intensity", emotions.get("intensity", 0.0))
    tension = _clamp_unit("emotions.tension_level", emotions.get("tension_level", 0.0))
    escalation = _clamp_unit(
        "emotions.escalation_risk", emotions.get("escalation_risk", 0.0)
    )

    defensive = bool(emotions.get("defensive", False))
    concession = bool(emotions.get("concession_made", False))

    snippet = cfg.transcript_snippet
    if snippet is None:
        snippet = _opt_str(payload.get("transcript_snippet"))

    result: dict[str, Any] = {
        "turn_id": cfg.turn_id or f"turn_{uuid.uuid4().hex[:12]}",
        "speaker": cfg.speaker,
        "emotions": {
            "primary": primary,
            "intensity": intensity,
            "tension_level": tension,
            "defensive": defensive,
            "concession_made": concession,
            "escalation_risk": escalation,
        },
        "whisper_prompt": _opt_str(payload.get("whisper_prompt")),
        "escalation_alert": _opt_str(payload.get("escalation_alert")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if snippet is not None:
        result["transcript_snippet"] = snippet
    return result


def analyze_emotion(
    processor: Any,
    model: Any,
    audio_source: str,
    *,
    cfg: Optional[EmotionConfig] = None,
) -> dict:
    """Score a single short audio clip (<= 30 s) and return an
    EmotionRadarResult dict matching
    ``data/schemas/emotion_radar.schema.json``.

    Mirrors the call-shape of
    :func:`backend.core._runtime.audio.transcribe` so the two phase-2
    components share an entry pattern.

    Raises :class:`EmotionAnalysisError` if the model emits a payload
    that doesn't match the schema's primary-emotion enum or numeric
    ranges. Lets :class:`backend.core._runtime.parsing.JsonParseError`
    propagate when the model output isn't JSON at all — that's a model
    failure worth surfacing rather than swallowing.
    """
    import torch

    cfg = cfg or EmotionConfig()
    instruction = _resolve_instruction(cfg)

    messages = build_emotion_messages(audio_source, instruction)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)

    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    text = _extract_text(processor.parse_response(raw)).strip()

    payload = parse_json(text)
    return _coerce_emotion_payload(payload, cfg=cfg)


def analyze_emotion_long(
    processor: Any,
    model: Any,
    audio_source: str,
    *,
    cfg: Optional[EmotionConfig] = None,
    max_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> list[dict]:
    """Score a clip of arbitrary length by chunking past Gemma 4's 30 s
    audio cap.

    Returns a **list** of EmotionRadarResult dicts — one per chunk
    window. A clip that already fits in a single window still returns
    a one-element list so callers don't branch on length.

    Chunk slicing is shared with the transcription path (``chunk_audio``
    / ``compute_chunk_windows``). When ``cfg.turn_id`` is set, per-chunk
    turn ids are derived as ``"<turn_id>_chunk_NNN"``; otherwise each
    chunk gets a fresh uuid.
    """
    import tempfile
    from pathlib import Path

    import soundfile as sf

    cfg = cfg or EmotionConfig()
    chunks = chunk_audio(
        audio_source,
        max_chunk_seconds=max_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    if not chunks:
        return []

    base_turn_id = cfg.turn_id
    results: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i, (_start_s, _end_s, wave) in enumerate(chunks):
            chunk_path = Path(tmp) / f"chunk_{i:03d}.wav"
            sf.write(str(chunk_path), wave, 16000)
            chunk_cfg = EmotionConfig(
                max_new_tokens=cfg.max_new_tokens,
                instruction=cfg.instruction,
                speaker=cfg.speaker,
                turn_id=(
                    f"{base_turn_id}_chunk_{i:03d}" if base_turn_id else None
                ),
                transcript_snippet=cfg.transcript_snippet,
                prompt_name=cfg.prompt_name,
            )
            results.append(
                analyze_emotion(processor, model, chunk_path, cfg=chunk_cfg)
            )
    return results
