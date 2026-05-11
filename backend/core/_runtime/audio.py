"""Audio transcription via Gemma 4's native multimodal path.

Gemma 4 E2B/E4B accept audio directly in the chat-template ``content``
list — no separate ASR model required. We reuse the same processor and
model loaded by :func:`backend.core._runtime.model.load_model` with
``LoadConfig(multimodal=True)``.

Per the Gemma 4 audio docs: audio content goes **before** the text
instruction in the message ``content``, clips must be ≤ 30 seconds, and
``librosa`` (already in requirements.txt) is the audio loader behind the
processor's feature extractor.

For clips longer than Gemma 4's 30 s audio window — and Live Mode will
see those constantly — use :func:`transcribe_long`, which splits the
clip into overlapping windows, transcribes each, and stitches the result
back together with a ``segments`` field on the output dict.
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
DEFAULT_CHUNK_SECONDS = 28.0  # safety margin under MAX_AUDIO_SECONDS
DEFAULT_CHUNK_OVERLAP_SECONDS = 0.5


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


def compute_chunk_windows(
    total_seconds: float,
    *,
    max_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> list[tuple[float, float]]:
    """Compute (start_seconds, end_seconds) windows that cover [0, total_seconds].

    Pure function — no audio I/O, no numpy, so it's unit-testable without
    librosa or transformers. ``transcribe_long`` uses this to decide where
    to slice the waveform.

    Guarantees:
      * every returned window is at most ``max_chunk_seconds`` long
      * consecutive windows overlap by ``overlap_seconds``
      * the last window ends exactly at ``total_seconds``
      * a clip that already fits in one window yields a single window
    """
    if total_seconds <= 0:
        return []
    if total_seconds <= max_chunk_seconds:
        return [(0.0, total_seconds)]
    if overlap_seconds >= max_chunk_seconds:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be < max_chunk_seconds "
            f"({max_chunk_seconds}); otherwise the loop never advances"
        )

    windows: list[tuple[float, float]] = []
    start = 0.0
    step = max_chunk_seconds - overlap_seconds
    while start < total_seconds:
        end = min(start + max_chunk_seconds, total_seconds)
        windows.append((start, end))
        if end >= total_seconds:
            break
        start += step
    return windows


def chunk_audio(
    audio_path: str,
    *,
    max_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    target_sr: int = 16000,
):
    """Load an audio file and split it into overlapping windows.

    Returns a list of ``(start_seconds, end_seconds, np.ndarray)`` tuples.
    The arrays are mono float32 at ``target_sr`` (Gemma 4's expected rate).
    Lazy-imports ``librosa`` so importing this module doesn't pull in the
    heavy audio stack on CI.
    """
    import librosa
    import numpy as np

    wave, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    total_seconds = len(wave) / float(sr)
    windows = compute_chunk_windows(
        total_seconds,
        max_chunk_seconds=max_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    out = []
    for start_s, end_s in windows:
        start_idx = int(start_s * sr)
        end_idx = int(end_s * sr)
        out.append((start_s, end_s, wave[start_idx:end_idx].astype(np.float32)))
    return out


def transcribe_long(
    processor: Any,
    model: Any,
    audio_source: str,
    *,
    cfg: Optional[TranscribeConfig] = None,
    model_id: Optional[str] = None,
    max_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> dict:
    """Transcribe audio of any length by chunking past Gemma 4's 30 s cap.

    For clips already within ``max_chunk_seconds`` this delegates to
    :func:`transcribe` and returns a single-window result with no
    ``segments`` field.

    For longer clips:
      1. ``chunk_audio`` slices the waveform into overlapping windows.
      2. Each window is written to a temp WAV.
      3. :func:`transcribe` runs on each (sharing the loaded model).
      4. Per-window transcripts concatenate into ``transcript`` and the
         per-window starts/ends/texts appear under ``segments``.

    The returned dict matches ``data/schemas/transcription.schema.json``
    (the ``segments`` field is the schema's optional addition).
    """
    import tempfile
    from pathlib import Path

    import soundfile as sf

    cfg = cfg or TranscribeConfig()

    chunks = chunk_audio(
        audio_source,
        max_chunk_seconds=max_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    if len(chunks) <= 1:
        # Short-circuit: short clip — return the existing single-call shape.
        return transcribe(processor, model, audio_source, cfg=cfg, model_id=model_id)

    LOG.info("Chunking long audio into %d windows", len(chunks))

    info = sf.info(str(audio_source))
    total_duration = info.frames / info.samplerate

    segments: list[dict] = []
    transcripts: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i, (start_s, end_s, wave) in enumerate(chunks):
            chunk_path = Path(tmp) / f"chunk_{i:03d}.wav"
            sf.write(str(chunk_path), wave, 16000)
            sub = transcribe(
                processor, model, chunk_path, cfg=cfg, model_id=model_id
            )
            text = sub["transcript"].strip()
            transcripts.append(text)
            segments.append(
                {
                    "start_seconds": round(float(start_s), 2),
                    "end_seconds": round(float(end_s), 2),
                    "transcript": text,
                }
            )

    resolved_model_id = (
        model_id
        or getattr(getattr(model, "config", None), "_name_or_path", None)
        or "unknown"
    )

    return {
        "audio_source": str(audio_source),
        "transcript": " ".join(t for t in transcripts if t).strip(),
        "language": cfg.language,
        "duration_seconds": round(total_duration, 2),
        "model_id": resolved_model_id,
        "instruction": cfg.instruction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
    }
