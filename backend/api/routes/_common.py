"""Shared route helpers: runtime-error → HTTP-exception mapping.

Every intelligence runtime raises a ``ValueError`` subclass on
unrepairable model output (``TalkDNAAnalysisError``, ``PremortemError``,
``PulseError``, etc.) and attaches an ``attempts`` list with the raw
output of each retry for debugging. The HTTP wrapper surfaces both:

* status 422 (Unprocessable Entity) on a model-output validation failure.
* the ``error`` string and the ``attempts`` list in the response body so
  the frontend / notebook can render diagnostics inline.

A separate 503 (Service Unavailable) is raised when the model registry
is missing — the lifespan event hasn't run, or the test forgot to
override the dependency. That failure is distinct from "the model
returned junk" and should be loud.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from backend.api.deps import RegistryNotReady
from backend.core._runtime import StorageError, StorageNotFoundError

__all__ = [
    "runtime_error_to_http",
    "registry_error_to_http",
    "storage_error_to_http",
]


def runtime_error_to_http(exc: Exception, *, component: str) -> HTTPException:
    """Wrap a runtime ``XxxError`` as a 422 with the diagnostic payload.

    The runtime attaches the per-attempt raw output as ``exc.attempts``
    (truncated to 500 chars per attempt). When present, surface it so
    the caller can inspect what the model emitted across the two
    retries.
    """
    attempts = getattr(exc, "attempts", None)
    detail: dict[str, Any] = {
        "component": component,
        "error": str(exc),
    }
    if isinstance(attempts, list):
        detail["attempts"] = attempts
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=detail,
    )


def storage_error_to_http(exc: StorageError, *, component: str) -> HTTPException:
    """Map a :class:`StorageError` to the right HTTP status.

    * :class:`StorageNotFoundError` → 404 (no record on disk for the
      given id).
    * Any other :class:`StorageError` → 422 (validation / shape /
      corruption — the request itself or the on-disk record was
      malformed).

    ``component`` matches the runtime component name surfaced on the
    other routes (``"talk_dna_storage"``, ``"vault_storage"``,
    ``"pulse_storage"``, ``"conversation_storage"``) so the frontend
    can route diagnostics inline.
    """
    if isinstance(exc, StorageNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"component": component, "error": str(exc)},
        )
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"component": component, "error": str(exc)},
    )


def registry_error_to_http(exc: RegistryNotReady) -> HTTPException:
    """Wrap a :class:`RegistryNotReady` as a 503.

    The model variant the route asked for isn't loaded. In production
    this means the lifespan event didn't run (or it was killed mid-load
    on a CUDA OOM). In tests it means the dependency override is wrong.
    """
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": str(exc)},
    )
