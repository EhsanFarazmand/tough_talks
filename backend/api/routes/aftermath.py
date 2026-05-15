"""Aftermath route — wraps :func:`backend.core._runtime.generate_aftermath`."""

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
from backend.api.schemas import AftermathRequest
from backend.core._runtime import (
    AftermathConfig,
    AftermathError,
    generate_aftermath,
)

router = APIRouter()


@router.post(
    "",
    summary="Compare pre-mortem (plan) against the actual transcript (reality)",
    response_description=(
        "Aftermath dict matching data/schemas/aftermath.schema.json"
    ),
)
def generate(
    request: AftermathRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Produce a plan-vs-reality comparison for a finished practice round.

    Required inputs: the pre-mortem (Step 8 output) and the practice
    transcript. Optional: PersonVault and debrief — the aftermath
    reconciles its read with the post-round coaching when present.

    Per ``knowledge/phases/rules.md § Decoding knobs by task type``, set
    ``enable_thinking=true`` and ``max_new_tokens=4096`` for the
    cross-field consistency this payload requires (three scenarios
    cross-referenced against transcript, derived ``materialized`` from
    ``match_quality``, evidence-vs-quality consistency).
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = AftermathConfig(
        premortem=request.premortem,
        transcript=request.transcript,
        user_goal=request.user_goal,
        person_profile=request.person_profile,
        debrief=request.debrief,
        max_new_tokens=request.max_new_tokens,
        prompt_name=request.prompt_name,
        enable_thinking=request.enable_thinking,
    )
    try:
        return generate_aftermath(processor, model, cfg=cfg)
    except AftermathError as exc:
        raise runtime_error_to_http(exc, component="aftermath") from exc
