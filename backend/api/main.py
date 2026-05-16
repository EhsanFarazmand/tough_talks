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

Phase 6 / Step 14 also mounts the vanilla-JS frontend at ``/app`` from
:func:`resolve_frontend_dir` (``<repo>/frontend`` by default; override
via ``TOUGH_TALKS_FRONTEND_DIR``). Same-origin serving means the
working app calls every ``/storage/*`` and ``/persona/*`` route via
relative URLs — no CORS gymnastics — and the concept landing page stays
reachable at ``/app/tough_talks_concept.html``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
from backend.core._runtime import (
    resolve_storage_root,
    set_default_assistant_model,
)

LOG = logging.getLogger(__name__)

# Env knobs for the lifespan loader. Useful when bringing up the API on
# a constrained machine that can only afford one model variant in VRAM,
# or when running smoke tests where the model load itself is the slow
# step. Notebooks and tests skip the lifespan entirely.
_ENV_INCLUDE_TEXT = "TOUGH_TALKS_INCLUDE_TEXT"
_ENV_INCLUDE_MULTIMODAL = "TOUGH_TALKS_INCLUDE_MULTIMODAL"
_ENV_USE_ASSISTANT = "TOUGH_TALKS_USE_ASSISTANT"
_ENV_SKIP_MODEL_LOAD = "TOUGH_TALKS_SKIP_MODEL_LOAD"
ENV_FRONTEND_DIR = "TOUGH_TALKS_FRONTEND_DIR"

# Repo-default frontend dir — sibling of ``backend/``. Resolved at
# import time so the StaticFiles mount can be registered before any
# request reaches the app.
DEFAULT_FRONTEND_DIR: Path = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML_NAME = "app.html"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_frontend_dir(*, override: Optional[Path] = None) -> Path:
    """Resolve the static-frontend directory the ``/app`` mount serves.

    Precedence (highest wins):

    1. Explicit ``override`` arg (notebooks / tests).
    2. ``TOUGH_TALKS_FRONTEND_DIR`` env var (packaged deploys that ship
       the static bundle outside the repo).
    3. ``<repo>/frontend`` — the repo-default checked-in location.

    Returns the path regardless of whether it exists. The lifespan
    event logs a warning when the resolved dir is missing; the mount
    is only registered when the dir is actually present, so a missing
    frontend doesn't crash the API — every ``/app/...`` request just
    404s through the FastAPI router fallback instead.
    """
    if override is not None:
        return Path(override)
    env = os.getenv(ENV_FRONTEND_DIR)
    if env:
        return Path(env)
    return DEFAULT_FRONTEND_DIR


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

    # Resolve the frontend dir for the /app mount. The mount itself is
    # registered at import time (below) — this only updates the path
    # surfaced by /health and used by the explicit /app/ index route.
    frontend_dir = resolve_frontend_dir()
    app.state.frontend_dir = frontend_dir
    if frontend_dir.is_dir():
        LOG.info("Frontend dir: %s", frontend_dir)
    else:
        LOG.warning(
            "Frontend dir does not exist: %s — /app/* routes will 404",
            frontend_dir,
        )

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
    # Default ON: speculative decoding is lossless (the draft's tokens
    # are verified by the target — see google/gemma-4-E2B-it-assistant
    # model card) and gives ~3x decode speed on T4. Set
    # TOUGH_TALKS_USE_ASSISTANT=0 to switch back to plain generation
    # without any code change.
    include_assistant = _env_bool(_ENV_USE_ASSISTANT, default=True)
    LOG.info(
        "Building ModelRegistry (text=%s, multimodal=%s, assistant=%s)",
        include_text,
        include_multimodal,
        include_assistant,
    )
    app.state.registry = load_registry(
        include_text=include_text,
        include_multimodal=include_multimodal,
        include_assistant=include_assistant,
    )
    # Install the draft as the chat() module-level default so every text
    # route picks it up automatically — no per-runtime kwarg plumbing
    # required. When include_assistant=False, this passes None and
    # plain generation runs.
    set_default_assistant_model(app.state.registry.assistant())
    LOG.info("ModelRegistry ready")
    try:
        yield
    finally:
        # transformers does not expose an explicit cleanup hook; clearing
        # the reference is enough for the GC to release the weights on
        # process exit. Kept explicit so the shutdown shape is clear.
        set_default_assistant_model(None)
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
    assistant_loaded = bool(registry and registry.assistant_model is not None)
    storage_root = getattr(app.state, "storage_root", None)
    frontend_dir = getattr(app.state, "frontend_dir", None)
    frontend_present = bool(frontend_dir and Path(frontend_dir).is_dir())
    return {
        "status": "ok",
        "version": "0.1.0",
        "text_model_loaded": text_loaded,
        "multimodal_model_loaded": multimodal_loaded,
        "assistant_model_loaded": assistant_loaded,
        "storage_root": str(storage_root) if storage_root is not None else None,
        "frontend_dir": str(frontend_dir) if frontend_dir is not None else None,
        "frontend_present": frontend_present,
    }


# --- Frontend (Phase 6 / Step 14) -----------------------------------------
# The order matters here. Explicit routes for ``/`` and ``/app/`` are
# registered BEFORE the ``StaticFiles`` mount at ``/app`` so FastAPI's
# router resolves the index hit to ``app.html`` (not 404, which is what
# ``StaticFiles(html=False)`` would emit for a directory listing). The
# mount then handles every ``/app/<file>`` request — including
# ``tough_talks_concept.html``, ``app.css``, ``app.js``.
#
# We deliberately keep ``html=False`` on the mount. With ``html=True``
# StaticFiles would try to serve ``index.html`` from the dir, and we
# don't ship one (the working app lives in ``app.html`` so the concept
# page stays the marketing landing — same dir, different files). The
# explicit route is the cleanest way to make ``/app/`` resolve to the
# right file without renaming or shadowing.


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Send root visitors to the working app at ``/app/``.

    307 (not 308) preserves the method on the redirect — important for
    the rare case where a client POSTs to ``/``. The ``/`` path doesn't
    appear in the OpenAPI doc so the API surface stays clean.
    """
    return RedirectResponse(url="/app/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@app.get("/app/", include_in_schema=False)
async def serve_app_index(request: Request) -> FileResponse:
    """Serve ``app.html`` as the ``/app/`` index.

    Reads ``app.state.frontend_dir`` first (set by the lifespan event
    or by a test override) and falls back to :func:`resolve_frontend_dir`
    when the state isn't populated — same shape as ``get_storage_root``
    in ``backend/api/deps.py``.
    """
    frontend_dir = getattr(request.app.state, "frontend_dir", None) or resolve_frontend_dir()
    target = Path(frontend_dir) / APP_HTML_NAME
    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"{APP_HTML_NAME} not found in frontend dir {frontend_dir}",
            },
        )
    return FileResponse(str(target), media_type="text/html")


_frontend_dir_at_import = resolve_frontend_dir()
if _frontend_dir_at_import.is_dir():
    app.mount(
        "/app",
        StaticFiles(directory=str(_frontend_dir_at_import), html=False),
        name="frontend",
    )
else:
    LOG.warning(
        "Frontend dir missing at import time (%s) — /app/<file> requests "
        "will 404 until the dir exists and the app reloads. Set "
        "%s to point at the static bundle for packaged deploys.",
        _frontend_dir_at_import,
        ENV_FRONTEND_DIR,
    )


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
