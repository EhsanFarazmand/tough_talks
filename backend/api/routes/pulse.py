"""Relationship Pulse route — wraps :func:`backend.core._runtime.generate_pulse`."""

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
from backend.api.schemas import PulseRequest
from backend.core._runtime import (
    PulseConfig,
    PulseError,
    generate_pulse,
)

router = APIRouter()


@router.post(
    "",
    summary="Roll N practice rounds into a relationship-level pulse",
    response_description=(
        "Pulse dict matching data/schemas/pulse.schema.json"
    ),
)
def generate(
    request: PulseRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Aggregate multiple rounds for one (user, counterparty) pair.

    Required: at least two rounds and a PersonVault profile. Pass
    ``prior_pulse`` for incremental v1 → v2 updates; the runtime bumps
    ``version``, derives ``round_count``, and accumulates
    ``recurring_patterns`` and ``relationship_wins`` across pulse
    versions (the same pattern as PersonVault qualitative lists).

    Per the promoted rule in ``knowledge/phases/rules.md``, this route
    defaults to ``enable_thinking=true`` with a 4096-token budget —
    pulse is the most cross-field-consistency-heavy Phase 4 payload.
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = PulseConfig(
        rounds=request.rounds,
        person_profile=request.person_profile,
        prior_pulse=request.prior_pulse,
        person_id=request.person_id,
        max_new_tokens=request.max_new_tokens,
        prompt_name=request.prompt_name,
        enable_thinking=request.enable_thinking,
    )
    try:
        return generate_pulse(processor, model, cfg=cfg)
    except PulseError as exc:
        raise runtime_error_to_http(exc, component="pulse") from exc
