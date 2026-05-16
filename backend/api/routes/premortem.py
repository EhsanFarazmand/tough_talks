"""Pre-mortem route — wraps :func:`backend.core._runtime.generate_premortem`."""

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
from backend.api.schemas import PremortemRequest
from backend.core._runtime import (
    PremortemConfig,
    PremortemError,
    generate_premortem,
)

router = APIRouter()


@router.post(
    "",
    summary="Generate a 3-scenario pre-mortem for an upcoming conversation",
    response_description=(
        "PreMortem dict matching data/schemas/premortem.schema.json"
    ),
)
def generate(
    request: PremortemRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> StreamingResponse:
    """Produce three parameterised failure scenarios.

    Each scenario's ``simulation_parameters`` is shaped to feed into a
    Persona Sim practice round (the ``resistance_type`` enum is shared
    end-to-end). Pass ``person_profile`` and ``talk_dna_profile`` to
    ground scenarios in observed behaviour rather than generic plays.

    Per ``knowledge/phases/rules.md § Decoding knobs by task type``, set
    ``enable_thinking=true`` for cross-field consistency on analytical
    payloads (the rule is N=4 strong as of Step 11).

    Response is streamed via :func:`streaming_runtime_call` so the
    cloudflared edge ~100 s TTFB timeout never fires on T4 — see
    :mod:`backend.api.routes._stream` for the envelope contract.
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = PremortemConfig(
        conversation_description=request.conversation_description,
        user_goal=request.user_goal,
        person_profile=request.person_profile,
        talk_dna_profile=request.talk_dna_profile,
        max_new_tokens=request.max_new_tokens,
        prompt_name=request.prompt_name,
        enable_thinking=request.enable_thinking,
    )
    return streaming_runtime_call(
        run=lambda: generate_premortem(processor, model, cfg=cfg),
        component="premortem",
        runtime_error_cls=PremortemError,
    )
