"""Audio transcription via Gemma 4's native multimodal path.

Gemma 4 E2B/E4B accept audio directly in the chat-template ``content``
list — no separate ASR model required. We reuse the same processor and
model loaded by :func:`backend.core._runtime.model.load_model` with
``LoadConfig(multimodal=True)``.

Per the Gemma 4 audio docs: audio content goes **before** the text
instruction in the message ``content``, clips must be ≤ 30 seconds, and
``librosa`` (already in requirements.txt) is the audio loader behind the
processor's feature extractor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

LOG = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and "
    "not one point seven, and write 3 instead of three."
)

MAX_AUDIO_SECONDS = 30  # Gemma 4 audio context cap


@dataclass
class TranscribeConfig:
    max_new_tokens: int = 256
    instruction: str = DEFAULT_INSTRUCTION
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


def build_transcription_messages(audio_source: str, instruction: str) -> list[dict]:
    """Construct the chat-template messages for a transcription request.

    Pure function, no ML deps — kept separate from :func:`transcribe` so
    the message shape can be unit-tested without loading transformers.

    The audio part comes first per Gemma 4's documented best practice
    ("place image and/or audio content before the text").
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


def _extract_text(parsed: Any) -> str:
    """``processor.parse_response`` returns ``{"role": ..., "content": ...}``
    for plain assistant text. Tolerate raw strings too in case a future
    transformers release changes the return shape."""
    if isinstance(parsed, dict):
        content = parsed.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(parsed)


def transcribe(
    processor: Any,
    model: Any,
    audio_source: str,
    *,
    cfg: Optional[TranscribeConfig] = None,
    model_id: Optional[str] = None,
) -> dict:
    """Transcribe a single audio clip and return a dict matching
    ``data/schemas/transcription.schema.json``.

    ``audio_source`` is a local path or an http(s) URL — both are
    accepted by the processor's audio feature extractor (which uses
    ``librosa`` under the hood).

    ``model_id`` defaults to whatever the loaded model reports as its
    name; pass an explicit id when wrapping a quantized or finetuned
    checkpoint so the recorded provenance stays accurate.
    """
    import torch

    cfg = cfg or TranscribeConfig()

    messages = build_transcription_messages(audio_source, cfg.instruction)

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
    transcript = _extract_text(processor.parse_response(raw)).strip()

    resolved_model_id = (
        model_id
        or getattr(getattr(model, "config", None), "_name_or_path", None)
        or "unknown"
    )

    return {
        "audio_source": str(audio_source),
        "transcript": transcript,
        "language": cfg.language,
        "duration_seconds": cfg.duration_seconds,
        "model_id": resolved_model_id,
        "instruction": cfg.instruction,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
