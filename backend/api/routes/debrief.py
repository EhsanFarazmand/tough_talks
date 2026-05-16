"""Debrief route — wraps :func:`backend.core._runtime.generate_debrief`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.deps import (
    ModelRegistry,
    RegistryNotReady,
    get_registry,
)
from backend.api.routes._common import registry_error_to_http
from backend.api.routes._stream import streaming_runtime_call
from backend.api.schemas import DebriefRequest
from backend.core._runtime import (
    DebriefConfig,
    DebriefError,
    generate_debrief,
)

router = APIRouter()


@router.post(
    "",
    summary="Produce a post-round coaching debrief from a finished transcript",
    response_description=(
        "Debrief dict matching data/schemas/debrief.schema.json"
    ),
)
def generate(
    request: DebriefRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> StreamingResponse:
    """Score a finished practice round and return coaching feedback.

    Inputs: the practice transcript plus optional PersonVault and
    TalkDNA so the debrief can ground ``missed_openings`` /
    ``ground_lost`` against the counterparty's known patterns.

    Per ``knowledge/phases/rules.md § Decoding knobs by task type``,
    ``enable_thinking=true`` improves cross-field consistency on the
    bucket classifications (a USER move in the counterparty's
    ``de_escalation_keys`` belongs in ``wins``, not ``ground_lost``).

    Response is streamed via :func:`streaming_runtime_call` so the
    cloudflared edge ~100 s TTFB timeout never fires on T4 — see
    :mod:`backend.api.routes._stream` for the envelope contract.
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = DebriefConfig(
        transcript=request.transcript,
        user_goal=request.user_goal,
        person_profile=request.person_profile,
        talk_dna_profile=request.talk_dna_profile,
        max_new_tokens=request.max_new_tokens,
        prompt_name=request.prompt_name,
        enable_thinking=request.enable_thinking,
    )
    return streaming_runtime_call(
        run=lambda: generate_debrief(processor, model, cfg=cfg),
        component="debrief",
        runtime_error_cls=DebriefError,
    )
