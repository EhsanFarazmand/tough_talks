"""Debrief route — wraps :func:`backend.core._runtime.generate_debrief`."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import (
    ModelRegistry,
    RegistryNotReady,
    get_registry,
)
from backend.api.routes._common import (
    registry_error_to_http,
    runtime_error_to_http,
)
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
) -> dict:
    """Score a finished practice round and return coaching feedback.

    Inputs: the practice transcript plus optional PersonVault and
    TalkDNA so the debrief can ground ``missed_openings`` /
    ``ground_lost`` against the counterparty's known patterns.

    Per ``knowledge/phases/rules.md § Decoding knobs by task type``,
    ``enable_thinking=true`` improves cross-field consistency on the
    bucket classifications (a USER move in the counterparty's
    ``de_escalation_keys`` belongs in ``wins``, not ``ground_lost``).
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
    try:
        return generate_debrief(processor, model, cfg=cfg)
    except DebriefError as exc:
        raise runtime_error_to_http(exc, component="debrief") from exc
