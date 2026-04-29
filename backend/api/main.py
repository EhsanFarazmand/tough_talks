"""
Tough Talks — FastAPI Backend
Phase 5, Step 12

This file is the entrypoint for the local API server.
Run: uvicorn backend.api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Tough Talks API",
    description="Local AI conversation intelligence engine — powered by Gemma 4",
    version="0.1.0",
)

# Allow frontend to call the API locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Fine for local-only deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# --- Routers (added as each phase completes) ---
# from backend.api.routes import talk_dna, person_vault, emotion_radar
# from backend.api.routes import persona_sim, debrief, aftermath, pulse
#
# app.include_router(talk_dna.router, prefix="/talk-dna", tags=["Talk DNA"])
# app.include_router(person_vault.router, prefix="/vault", tags=["Person Vault"])
# app.include_router(emotion_radar.router, prefix="/emotion", tags=["Emotion Radar"])
# app.include_router(persona_sim.router, prefix="/persona", tags=["Persona Simulator"])
# app.include_router(debrief.router, prefix="/debrief", tags=["Debrief"])
# app.include_router(aftermath.router, prefix="/aftermath", tags=["Aftermath"])
# app.include_router(pulse.router, prefix="/pulse", tags=["Relationship Pulse"])
