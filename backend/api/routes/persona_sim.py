"""Persona simulator routes — single turn (``/reply``) and multi-turn (``/run``).

Both wrap :mod:`backend.core._runtime.persona_sim`. The ``/reply`` route
generates one in-character turn given a rolling history; the ``/run``
route processes a whole list of user messages in order and returns the
final alternating history.
"""

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
from backend.api.schemas import PersonaReplyRequest, PersonaRunRequest
from backend.core._runtime import (
    PersonaReplyError,
    PersonaSimConfig,
    generate_persona_reply,
    run_practice_conversation,
)

router = APIRouter()


def _build_cfg(
    *,
    persona_profile: dict,
    user_goal: str,
    max_new_tokens: int,
    enable_thinking: bool,
    prompt_name: str,
) -> PersonaSimConfig:
    return PersonaSimConfig(
        persona_profile=persona_profile,
        user_goal=user_goal,
        max_new_tokens=max_new_tokens,
        prompt_name=prompt_name,
        enable_thinking=enable_thinking,
    )


@router.post(
    "/reply",
    summary="Generate one in-character persona reply",
    response_description=(
        "PersonaReply dict matching data/schemas/persona_reply.schema.json"
    ),
)
def reply(
    request: PersonaReplyRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Produce one persona reply to the user's latest message.

    Pass the running ``history`` (prior user / persona turns) so the
    escalation arc compounds across turns. For the first turn of a new
    practice round, pass an empty ``history`` list.
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = _build_cfg(
        persona_profile=request.persona_profile,
        user_goal=request.user_goal,
        max_new_tokens=request.max_new_tokens,
        enable_thinking=request.enable_thinking,
        prompt_name=request.prompt_name,
    )
    try:
        return generate_persona_reply(
            processor,
            model,
            request.user_message,
            history=request.history,
            cfg=cfg,
        )
    except PersonaReplyError as exc:
        raise runtime_error_to_http(exc, component="persona_sim") from exc


@router.post(
    "/run",
    summary="Run a full multi-turn practice conversation non-interactively",
    response_description=(
        "{'history': [...]} — alternating user / persona turns"
    ),
)
def run(
    request: PersonaRunRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """Process a list of user messages, calling the persona for each.

    Convenient for offline replay and notebook demos. Returns the final
    history as a list of dicts alternating ``user`` and ``persona``
    entries (the runtime's emit shape).
    """
    try:
        processor, model = registry.text()
    except RegistryNotReady as exc:
        raise registry_error_to_http(exc) from exc

    cfg = _build_cfg(
        persona_profile=request.persona_profile,
        user_goal=request.user_goal,
        max_new_tokens=request.max_new_tokens,
        enable_thinking=request.enable_thinking,
        prompt_name=request.prompt_name,
    )
    try:
        history = run_practice_conversation(
            processor,
            model,
            request.user_messages,
            cfg=cfg,
        )
    except PersonaReplyError as exc:
        raise runtime_error_to_http(exc, component="persona_sim") from exc
    return {"history": history}
