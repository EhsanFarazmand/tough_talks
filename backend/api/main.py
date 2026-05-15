"""Tough Talks — FastAPI Backend.

Phase 5 / Step 12. Local-only API that exposes every intelligence
component from Phases 1-4 as a stateless HTTP endpoint. The frontend
(Phase 6) sends practice transcripts, audio uploads, and rolling
profiles in each request and stores results client-side — Step 13 wires
up the local JSON storage layer that will later persist them on the
backend.

Run locally:

    uvicorn backend.api.main:app --reload

The lifespan event eagerly loads both Gemma 4 variants (text-only +
multimodal) before serving the first request — see ``ModelRegistry`` in
:mod:`backend.api.deps`. Notebooks and tests skip the lifespan and
inject a ready-made registry via ``app.dependency_overrides``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.deps import load_registry
from backend.api.routes import (
    aftermath,
    debrief,
    emotion_radar,
    person_vault,
    persona_sim,
    premortem,
    pulse,
    storage,
    talk_dna,
    transcription,
)
from backend.core._runtime import resolve_storage_root

LOG = logging.getLogger(__name__)

# Env knobs for the lifespan loader. Useful when bringing up the API on
# a constrained machine that can only afford one model variant in VRAM,
# or when running smoke tests where the model load itself is the slow
# step. Notebooks and tests skip the lifespan entirely.
_ENV_INCLUDE_TEXT = "TOUGH_TALKS_INCLUDE_TEXT"
_ENV_INCLUDE_MULTIMODAL = "TOUGH_TALKS_INCLUDE_MULTIMODAL"
_ENV_SKIP_MODEL_LOAD = "TOUGH_TALKS_SKIP_MODEL_LOAD"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load Gemma 4 variants on startup; tear down on shutdown.

    Skipped entirely when ``TOUGH_TALKS_SKIP_MODEL_LOAD`` is set — that
    lets ``uvicorn backend.api.main:app --reload`` boot for static
    inspection of ``/docs`` without spending a minute on the model
    load. In every other case both variants are loaded by default and
    the registry is attached to ``app.state.registry``.
    """
    # Resolve the storage root regardless of whether models load —
    # the /storage/* routes are filesystem-only and stay available
    # even on a model-skipped boot (handy for the Step 13 notebook /
    # tests, and for ``uvicorn --reload`` smoke checks).
    storage_root = resolve_storage_root()
    app.state.storage_root = storage_root
    LOG.info("Storage root: %s", storage_root)

    if _env_bool(_ENV_SKIP_MODEL_LOAD, default=False):
        LOG.warning(
            "TOUGH_TALKS_SKIP_MODEL_LOAD set — skipping model load. "
            "Inference routes will 503 until a registry is injected."
        )
        app.state.registry = None
        yield
        return

    include_text = _env_bool(_ENV_INCLUDE_TEXT, default=True)
    include_multimodal = _env_bool(_ENV_INCLUDE_MULTIMODAL, default=True)
    LOG.info(
        "Building ModelRegistry (text=%s, multimodal=%s)",
        include_text,
        include_multimodal,
    )
    app.state.registry = load_registry(
        include_text=include_text,
        include_multimodal=include_multimodal,
    )
    LOG.info("ModelRegistry ready")
    try:
        yield
    finally:
        # transformers does not expose an explicit cleanup hook; clearing
        # the reference is enough for the GC to release the weights on
        # process exit. Kept explicit so the shutdown shape is clear.
        app.state.registry = None


app = FastAPI(
    title="Tough Talks API",
    description=(
        "Local AI conversation intelligence engine — powered by Gemma 4. "
        "Every endpoint runs on-device; no data leaves the machine."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS stays wide open — the API is local-only (the only origins that
# can reach it are running on the same machine).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Meta"])
async def health() -> dict:
    """Liveness check. Returns ``{"status": "ok", ...}``.

    Also surfaces which model variants the registry has, so a smoke
    test can confirm the lifespan event completed successfully before
    hitting any inference route.
    """
    registry = getattr(app.state, "registry", None)
    text_loaded = bool(registry and registry.text_model is not None)
    multimodal_loaded = bool(registry and registry.multimodal_model is not None)
    storage_root = getattr(app.state, "storage_root", None)
    return {
        "status": "ok",
        "version": "0.1.0",
        "text_model_loaded": text_loaded,
        "multimodal_model_loaded": multimodal_loaded,
        "storage_root": str(storage_root) if storage_root is not None else None,
    }


# --- Routers ---------------------------------------------------------------
app.include_router(talk_dna.router, prefix="/talk-dna", tags=["Talk DNA"])
app.include_router(person_vault.router, prefix="/vault", tags=["Person Vault"])
app.include_router(persona_sim.router, prefix="/persona", tags=["Persona Simulator"])
app.include_router(premortem.router, prefix="/premortem", tags=["Pre-Mortem"])
app.include_router(debrief.router, prefix="/debrief", tags=["Debrief"])
app.include_router(aftermath.router, prefix="/aftermath", tags=["Aftermath"])
app.include_router(pulse.router, prefix="/pulse", tags=["Relationship Pulse"])
app.include_router(
    emotion_radar.router, prefix="/emotion", tags=["Emotion Radar"]
)
app.include_router(
    transcription.router, prefix="/transcribe", tags=["Transcription"]
)
app.include_router(storage.router, prefix="/storage", tags=["Storage"])
