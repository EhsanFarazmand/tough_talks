"""Local JSON storage routes — Phase 5 / Step 13.

Exposes filesystem-backed read / write for every long-lived artifact
produced by the Phase 1-4 components: TalkDNA, PersonVault, Pulse, and
bundled conversation rounds. The routes are thin wrappers over
:mod:`backend.core._runtime.storage` — no business logic; validation
and atomic writes already live there.

Endpoint shape:

* ``GET  /storage/talk-dna``                     read TalkDNA for ``user_id`` (default ``"local"``)
* ``PUT  /storage/talk-dna``                     write a TalkDNA payload (overwrite)

* ``GET  /storage/vault``                        list every PersonVault
* ``GET  /storage/vault/{person_id}``            read one PersonVault
* ``PUT  /storage/vault/{person_id}``            write a PersonVault (path id wins over body id)

* ``GET  /storage/pulse``                        list every Pulse
* ``GET  /storage/pulse/{person_id}``            read one Pulse
* ``PUT  /storage/pulse/{person_id}``            write a Pulse (path id wins over body id)

* ``GET  /storage/conversations``                list rounds (optional ?person_id= filter)
* ``GET  /storage/conversations/{round_id}``     read one round
* ``PUT  /storage/conversations/{round_id}``     write a round (bundle of transcript + artifacts)
* ``DELETE /storage/conversations/{round_id}``   remove one round

Error mapping (via :func:`storage_error_to_http`):

* :class:`StorageNotFoundError` → 404 (no record at path).
* Any other :class:`StorageError` → 422 (validation / shape / corruption).
* Pydantic body errors → 422 by FastAPI before we ever run.

Defaults from the runtime apply: missing ``user_id`` / ``person_id`` /
``round_id`` are minted on PUT when the caller didn't supply one (the
response body returns the minted id so the frontend can keep using it
for subsequent requests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Query, status
from pydantic import BaseModel, Field

from backend.api.deps import get_storage_root
from backend.api.routes._common import storage_error_to_http
from backend.core._runtime import (
    DEFAULT_USER_ID,
    StorageError,
    delete_conversation,
    list_conversations,
    list_person_vaults,
    list_pulses,
    load_conversation,
    load_person_vault,
    load_pulse,
    load_talk_dna,
    save_conversation,
    save_person_vault,
    save_pulse,
    save_talk_dna,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


class WriteResult(BaseModel):
    """Response for a PUT — the saved payload + the on-disk path.

    The payload is echoed back because the storage layer stamps
    defaults (``updated_at``, freshly-minted ids) that the caller
    needs to see. The path is surfaced for diagnostics only — the
    frontend never reads the file directly, but the notebook
    integration test prints it.
    """

    payload: dict[str, Any] = Field(..., description="Payload as written to disk.")
    path: str = Field(..., description="Absolute path of the written record.")


class ListResult(BaseModel):
    """Response for a LIST — wraps the entries plus a count."""

    count: int
    items: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Talk DNA
# ---------------------------------------------------------------------------


@router.get(
    "/talk-dna",
    summary="Read the local TalkDNA profile",
    response_description="TalkDNA dict matching data/schemas/talk_dna.schema.json",
)
def read_talk_dna(
    user_id: str = Query(default=DEFAULT_USER_ID),
    root: Path = Depends(get_storage_root),
) -> dict[str, Any]:
    try:
        return load_talk_dna(root, user_id=user_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="talk_dna_storage") from exc


@router.put(
    "/talk-dna",
    summary="Persist a TalkDNA profile (overwrite-in-place)",
    response_model=WriteResult,
)
def write_talk_dna(
    payload: dict[str, Any] = Body(..., description="TalkDNA payload to persist."),
    root: Path = Depends(get_storage_root),
) -> WriteResult:
    try:
        target = save_talk_dna(root, payload)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="talk_dna_storage") from exc
    # save_* normalises ``payload`` in place (mints ``updated_at`` etc.)
    # but ``payload`` came from the request body — re-read the file so
    # the response reflects exactly what's on disk.
    return WriteResult(payload=load_talk_dna(root, user_id=payload.get("user_id", DEFAULT_USER_ID)), path=str(target))


# ---------------------------------------------------------------------------
# Person Vault
# ---------------------------------------------------------------------------


@router.get(
    "/vault",
    summary="List every PersonVault on disk",
    response_model=ListResult,
)
def list_vaults(root: Path = Depends(get_storage_root)) -> ListResult:
    items = list_person_vaults(root)
    return ListResult(count=len(items), items=items)


@router.get(
    "/vault/{person_id}",
    summary="Read one PersonVault by person_id",
    response_description="PersonVault dict matching data/schemas/person_vault.schema.json",
)
def read_vault(
    person_id: str,
    root: Path = Depends(get_storage_root),
) -> dict[str, Any]:
    try:
        return load_person_vault(root, person_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="vault_storage") from exc


@router.put(
    "/vault/{person_id}",
    summary="Persist a PersonVault (path id wins over body id)",
    response_model=WriteResult,
)
def write_vault(
    person_id: str,
    payload: dict[str, Any] = Body(...),
    root: Path = Depends(get_storage_root),
) -> WriteResult:
    # The path id is authoritative — the route only persists ONE record
    # per id, so a mismatch between path and body is a caller bug. We
    # silently force the path id rather than 422-ing because the
    # frontend may want to use the same payload shape for both POST-
    # to-runtime and PUT-to-storage and the runtime mints the id.
    payload = {**payload, "person_id": person_id}
    try:
        target = save_person_vault(root, payload)
        written = load_person_vault(root, person_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="vault_storage") from exc
    return WriteResult(payload=written, path=str(target))


# ---------------------------------------------------------------------------
# Pulse
# ---------------------------------------------------------------------------


@router.get(
    "/pulse",
    summary="List every Pulse on disk",
    response_model=ListResult,
)
def list_pulses_route(root: Path = Depends(get_storage_root)) -> ListResult:
    items = list_pulses(root)
    return ListResult(count=len(items), items=items)


@router.get(
    "/pulse/{person_id}",
    summary="Read one Pulse by person_id",
    response_description="Pulse dict matching data/schemas/pulse.schema.json",
)
def read_pulse(
    person_id: str,
    root: Path = Depends(get_storage_root),
) -> dict[str, Any]:
    try:
        return load_pulse(root, person_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="pulse_storage") from exc


@router.put(
    "/pulse/{person_id}",
    summary="Persist a Pulse (path id wins over body id)",
    response_model=WriteResult,
)
def write_pulse(
    person_id: str,
    payload: dict[str, Any] = Body(...),
    root: Path = Depends(get_storage_root),
) -> WriteResult:
    payload = {**payload, "person_id": person_id}
    try:
        target = save_pulse(root, payload)
        written = load_pulse(root, person_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="pulse_storage") from exc
    return WriteResult(payload=written, path=str(target))


# ---------------------------------------------------------------------------
# Conversations (bundled rounds)
# ---------------------------------------------------------------------------


@router.get(
    "/conversations",
    summary="List bundled conversation rounds",
    response_model=ListResult,
)
def list_conversation_rounds(
    person_id: Optional[str] = Query(
        default=None,
        description="When set, restrict the list to rounds belonging to this person.",
    ),
    root: Path = Depends(get_storage_root),
) -> ListResult:
    try:
        items = list_conversations(root, person_id=person_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="conversation_storage") from exc
    return ListResult(count=len(items), items=items)


@router.get(
    "/conversations/{round_id}",
    summary="Read one bundled round by round_id",
    response_description="Conversation dict matching data/schemas/conversation.schema.json",
)
def read_conversation(
    round_id: str,
    root: Path = Depends(get_storage_root),
) -> dict[str, Any]:
    try:
        return load_conversation(root, round_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="conversation_storage") from exc


@router.put(
    "/conversations/{round_id}",
    summary="Persist a bundled round (path id wins over body id)",
    response_model=WriteResult,
)
def write_conversation(
    round_id: str,
    payload: dict[str, Any] = Body(...),
    root: Path = Depends(get_storage_root),
) -> WriteResult:
    payload = {**payload, "round_id": round_id}
    try:
        target = save_conversation(root, payload)
        written = load_conversation(root, round_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="conversation_storage") from exc
    return WriteResult(payload=written, path=str(target))


@router.delete(
    "/conversations/{round_id}",
    summary="Delete one bundled round",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_conversation(
    round_id: str,
    root: Path = Depends(get_storage_root),
) -> None:
    try:
        delete_conversation(root, round_id)
    except StorageError as exc:
        raise storage_error_to_http(exc, component="conversation_storage") from exc
