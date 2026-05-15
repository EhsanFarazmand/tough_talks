"""Model registry + FastAPI dependencies for the local Tough Talks API.

Phase 5 / Step 12 — the API is local-only and stateless; storage is the
job of Step 13. The ModelRegistry holds both Gemma 4 variants:

* ``text_processor`` / ``text_model`` — built with ``LoadConfig(multimodal=False)``.
  Used by every analytical / generative text route (talk_dna, person_vault,
  persona_sim, premortem, debrief, aftermath, pulse).
* ``multimodal_processor`` / ``multimodal_model`` — built with
  ``LoadConfig(multimodal=True)``. Used by the two audio routes
  (transcription, emotion_radar).

Loaded eagerly at app startup via the lifespan event in
:mod:`backend.api.main`. Notebooks and tests skip the lifespan and inject
a registry directly via ``app.dependency_overrides[get_registry]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Request

from pathlib import Path

from backend.core._runtime import (
    DEFAULT_MODEL_ID,
    LoadConfig,
    load_model,
    resolve_storage_root,
)

__all__ = [
    "ModelRegistry",
    "RegistryNotReady",
    "get_registry",
    "get_storage_root",
    "load_registry",
]

LOG = logging.getLogger(__name__)


class RegistryNotReady(RuntimeError):
    """Raised when a route asks for a model variant that wasn't loaded.

    Two cases:
      * The app's lifespan event has not run (test setup forgot to use
        ``with TestClient(app) as client`` AND forgot to override
        :func:`get_registry`).
      * The registry was built with ``include_multimodal=False`` but a
        route asked for ``multimodal()`` (or vice versa).
    """


@dataclass
class ModelRegistry:
    """Holds the text-only and multimodal Gemma 4 processor / model pairs.

    Built once per app via :func:`load_registry` at lifespan startup, or
    constructed manually in notebooks / tests and injected through
    :data:`fastapi.FastAPI.dependency_overrides`.

    The two pairs are kept as independent fields so the runtime cost is
    explicit at registry build time: a deployment that only serves text
    routes can omit the multimodal load, and the audio routes will
    raise :class:`RegistryNotReady` if invoked anyway.
    """

    text_processor: Optional[Any] = None
    text_model: Optional[Any] = None
    multimodal_processor: Optional[Any] = None
    multimodal_model: Optional[Any] = None

    def text(self) -> tuple[Any, Any]:
        """Return ``(processor, model)`` for text-only routes.

        Falls back to the multimodal pair when no dedicated text-only
        pair is loaded. Gemma 4's multimodal class wraps the same LM as
        the text-only class — text generation works on either, the
        multimodal variant just carries extra audio/image projection
        heads that go unused on a text-only chat.

        The fallback is what makes the API runnable on constrained
        hardware (T4 / on-device) where only one model variant fits in
        VRAM. High-VRAM deployments can still load both variants; the
        dedicated text-only pair is preferred when present because the
        runtime cost per inference is slightly lower.
        """
        if self.text_processor is not None and self.text_model is not None:
            return self.text_processor, self.text_model
        if (
            self.multimodal_processor is not None
            and self.multimodal_model is not None
        ):
            return self.multimodal_processor, self.multimodal_model
        raise RegistryNotReady(
            "no model loaded — initialise the registry with at least one "
            "of include_text or include_multimodal, or inject a registry "
            "with a loaded model pair"
        )

    def multimodal(self) -> tuple[Any, Any]:
        """Return ``(processor, model)`` for audio routes."""
        if self.multimodal_processor is None or self.multimodal_model is None:
            raise RegistryNotReady(
                "multimodal model is not loaded — initialise the registry "
                "with include_multimodal=True or inject a multimodal-enabled "
                "registry"
            )
        return self.multimodal_processor, self.multimodal_model


def load_registry(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    include_text: bool = True,
    include_multimodal: bool = True,
) -> ModelRegistry:
    """Eagerly load both Gemma 4 variants and return a :class:`ModelRegistry`.

    Used by the lifespan event at app startup. Notebooks and tests can
    build a registry directly from preloaded models instead of calling
    this — avoiding a second model load when the notebook already has
    one in memory.

    ``include_text`` and ``include_multimodal`` are independent so a
    text-only deployment can skip the multimodal cost. The audio routes
    will raise :class:`RegistryNotReady` when invoked on a text-only
    registry; the text routes will raise on a multimodal-only registry.
    """
    text_processor: Optional[Any] = None
    text_model: Optional[Any] = None
    if include_text:
        LOG.info("Loading text-only model: %s", model_id)
        text_processor, text_model = load_model(
            LoadConfig(model_id=model_id, multimodal=False)
        )

    mm_processor: Optional[Any] = None
    mm_model: Optional[Any] = None
    if include_multimodal:
        LOG.info("Loading multimodal model: %s", model_id)
        mm_processor, mm_model = load_model(
            LoadConfig(model_id=model_id, multimodal=True)
        )

    return ModelRegistry(
        text_processor=text_processor,
        text_model=text_model,
        multimodal_processor=mm_processor,
        multimodal_model=mm_model,
    )


def get_storage_root(request: Request) -> Path:
    """FastAPI dependency — returns the app-scoped local storage root.

    Phase 5 / Step 13. The storage layer is filesystem-backed (no
    network, no DB), so a "root" is all the routes need. The path is
    set on ``app.state.storage_root`` by the lifespan event (or by a
    test override) and resolved via the precedence-rule in
    :func:`backend.core._runtime.resolve_storage_root` —
    explicit-arg > ``TOUGH_TALKS_STORAGE_ROOT`` env > repo-local default
    (``<repo>/data/local``, gitignored).

    Tests and notebooks bypass the lifespan by overriding the
    dependency directly::

        app.dependency_overrides[get_storage_root] = lambda: tmp_path

    Falls back to :func:`resolve_storage_root` (env or default) when
    the state isn't set — this lets ``uvicorn backend.api.main:app``
    start from a clean repo without an explicit configuration step.
    """
    explicit = getattr(request.app.state, "storage_root", None)
    if isinstance(explicit, (str, Path)):
        return Path(explicit)
    return resolve_storage_root()


def get_registry(request: Request) -> ModelRegistry:
    """FastAPI dependency — returns the app-scoped :class:`ModelRegistry`.

    Routes inject this via ``Depends(get_registry)``. Tests and notebooks
    bypass the lifespan load by overriding the dependency:

        app.dependency_overrides[get_registry] = lambda: my_registry

    Raises :class:`RegistryNotReady` when the registry has not been
    initialised — which in production means the lifespan event did not
    run. Routes catch the exception and surface it as a 503 so the
    failure mode is obvious from outside.
    """
    registry = getattr(request.app.state, "registry", None)
    if not isinstance(registry, ModelRegistry):
        raise RegistryNotReady(
            "ModelRegistry not initialised — the app's lifespan event "
            "did not run, or the test forgot to override get_registry"
        )
    return registry
