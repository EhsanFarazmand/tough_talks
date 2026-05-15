"""Talk DNA route — wraps :func:`backend.core._runtime.analyze_talk_dna`."""

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
from backend.api.schemas import TalkDNARequest
from backend.core._runtime import (
    TalkDNAAnalysisError,
    TalkDNAConfig,
    analyze_talk_dna,
)

router = APIRouter()


@router.post(
    "/analyze",
    summary="Compute or incrementally update a TalkDNA profile",
    response_description=(
        "TalkDNA profile dict matching data/schemas/talk_dna.schema.json"
    ),
)
def analyze(
    request: TalkDNARequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Run the Talk DNA analyser on a single conversation transcript.

    Pass ``prior_profile`` to evolve an existing TalkDNA across
    conversations — the runtime bumps ``version``, increments
    ``conversation_count``, and weighted-averages the deterministic
    numerics against the prior values.
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = TalkDNAConfig(
        user_id=request.user_id,
        max_new_tokens=request.max_new_tokens,
        prior_profile=request.prior_profile,
        prompt_name=request.prompt_name,
    )
    try:
        return analyze_talk_dna(processor, model, request.turns, cfg=cfg)
    except TalkDNAAnalysisError as exc:
        raise runtime_error_to_http(exc, component="talk_dna") from exc
