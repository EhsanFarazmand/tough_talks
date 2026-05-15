"""Person Vault route — wraps :func:`backend.core._runtime.analyze_person_vault`."""

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
from backend.api.schemas import PersonVaultRequest
from backend.core._runtime import (
    PersonVaultAnalysisError,
    PersonVaultConfig,
    analyze_person_vault,
)

router = APIRouter()


@router.post(
    "/build",
    summary="Compute or incrementally update a PersonVault profile",
    response_description=(
        "PersonVault profile dict matching data/schemas/person_vault.schema.json"
    ),
)
def build(
    request: PersonVaultRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Build a counterparty profile from a conversation transcript.

    Pass ``prior_profile`` to evolve an existing vault — the runtime
    accumulates qualitative lists (emotional_triggers, de_escalation_keys,
    common_deflections) across calls, deduped and capped at 8 items.
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = PersonVaultConfig(
        name=request.name,
        relationship_type=request.relationship_type,
        person_id=request.person_id,
        max_new_tokens=request.max_new_tokens,
        prior_profile=request.prior_profile,
        prompt_name=request.prompt_name,
    )
    try:
        return analyze_person_vault(processor, model, request.turns, cfg=cfg)
    except PersonVaultAnalysisError as exc:
        raise runtime_error_to_http(exc, component="person_vault") from exc
