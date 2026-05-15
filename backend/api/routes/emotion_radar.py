"""Emotion radar route — wraps :func:`backend.core._runtime.analyze_emotion_long`.

Like the transcription route, this always uses the chunked variant so a
clip of any length works. The runtime returns a list of EmotionRadar
results (one per chunk window). For clips that fit in a single 28-second
window the list has exactly one entry; longer clips return one entry
per window with rolling-context whisper coaching.

Audio is uploaded as ``multipart/form-data`` via ``UploadFile``. The
``speaker`` form field (``user`` / ``other``) is required because the
runtime never coaches the counterparty (``whisper_prompt`` is forced to
``None`` for ``other``).
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.api.deps import (
    ModelRegistry,
    RegistryNotReady,
    get_registry,
)
from backend.api.routes._common import (
    registry_error_to_http,
    runtime_error_to_http,
)
from backend.core._runtime import (
    ALLOWED_SPEAKERS,
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_EMOTION_PROMPT_NAME,
    DEFAULT_PRIOR_CONTEXT_HISTORY,
    EmotionAnalysisError,
    EmotionConfig,
    analyze_emotion_long,
)

router = APIRouter()


@router.post(
    "/analyze",
    summary="Score an uploaded audio clip for emotion + prosody",
    response_description=(
        "{'results': [...]} — list of EmotionRadarResult dicts, one per "
        "audio chunk, each matching data/schemas/emotion_radar.schema.json"
    ),
)
def analyze(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/flac)"),
    speaker: str = Form(
        ...,
        description=f"One of {list(ALLOWED_SPEAKERS)}",
    ),
    turn_id: Optional[str] = Form(default=None),
    transcript_snippet: Optional[str] = Form(default=None),
    prior_context: Optional[str] = Form(default=None),
    max_new_tokens: int = Form(default=512),
    max_chunk_seconds: float = Form(default=DEFAULT_CHUNK_SECONDS),
    overlap_seconds: float = Form(default=DEFAULT_CHUNK_OVERLAP_SECONDS),
    max_prior_history: int = Form(default=DEFAULT_PRIOR_CONTEXT_HISTORY),
    prompt_name: str = Form(default=DEFAULT_EMOTION_PROMPT_NAME),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Score an audio clip and return one EmotionRadarResult per chunk."""
    if speaker not in ALLOWED_SPEAKERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"speaker must be one of {list(ALLOWED_SPEAKERS)}, got {speaker!r}",
            },
        )

    try:
        processor, model = registry.multimodal()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "audio upload is missing a filename"},
        )

    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        contents = audio.file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        cfg = EmotionConfig(
            max_new_tokens=max_new_tokens,
            speaker=speaker,
            turn_id=turn_id,
            transcript_snippet=transcript_snippet,
            prior_context=prior_context,
            prompt_name=prompt_name,
        )
        try:
            results = analyze_emotion_long(
                processor,
                model,
                tmp.name,
                cfg=cfg,
                max_chunk_seconds=max_chunk_seconds,
                overlap_seconds=overlap_seconds,
                max_prior_history=max_prior_history,
            )
        except EmotionAnalysisError as exc:
            raise runtime_error_to_http(exc, component="emotion_radar") from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return {"results": results}
