"""Audio transcription route — wraps :func:`backend.core._runtime.transcribe_long`.

Always routes through the chunked variant so clips of any length work
end-to-end: short clips short-circuit to the single ``transcribe`` call
(no ``segments`` field on the output); long clips return a stitched
transcript plus per-window ``segments``.

Audio is uploaded as ``multipart/form-data`` via ``UploadFile`` — the
shape browser ``MediaRecorder`` + ``FormData`` emits natively, which is
what the Phase 6 frontend will use. The route saves the upload to a
NamedTemporaryFile so ``librosa.load`` can read it, then deletes the
file after the runtime returns.
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
from backend.api.routes._common import registry_error_to_http
from backend.core._runtime import (
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_TRANSCRIPTION_INSTRUCTION,
    TranscribeConfig,
    transcribe_long,
)

router = APIRouter()


@router.post(
    "",
    summary="Transcribe an uploaded audio clip (any length)",
    response_description=(
        "Transcription dict matching data/schemas/transcription.schema.json; "
        "long clips also carry a 'segments' array."
    ),
)
def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/flac)"),
    language: Optional[str] = Form(default=None),
    duration_seconds: Optional[float] = Form(default=None),
    max_new_tokens: int = Form(default=256),
    instruction: str = Form(default=DEFAULT_TRANSCRIPTION_INSTRUCTION),
    max_chunk_seconds: float = Form(default=DEFAULT_CHUNK_SECONDS),
    overlap_seconds: float = Form(default=DEFAULT_CHUNK_OVERLAP_SECONDS),
    model_id: Optional[str] = Form(default=None),
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Transcribe an uploaded audio clip via Gemma 4's native audio path."""
    try:
        processor, model = registry.multimodal()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "audio upload is missing a filename"},
        )

    # librosa.load needs a real file path. Stream the upload to a
    # NamedTemporaryFile, preserve the original suffix so the soundfile
    # backend can dispatch on extension, then delete after the runtime
    # returns. ``delete=False`` is required on Windows — the runtime
    # opens the file by path and Windows won't let two handles coexist
    # on a default-delete temp.
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        contents = audio.file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        cfg = TranscribeConfig(
            max_new_tokens=max_new_tokens,
            instruction=instruction,
            language=language,
            duration_seconds=duration_seconds,
        )
        return transcribe_long(
            processor,
            model,
            tmp.name,
            cfg=cfg,
            model_id=model_id,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            # Best-effort cleanup — the OS will reap the tmpdir later
            # if the file is locked.
            pass
