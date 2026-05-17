# Tough Talks

> **Your private, on-device AI coach for the conversations that matter most.**
> Built on Gemma 4 · Zero cloud dependency · Single edge GPU end-to-end

[![Kaggle](https://img.shields.io/badge/Kaggle-Hackathon-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Gemma 4](https://img.shields.io/badge/Gemma%204-E2B--it-gold?style=flat-square)](https://)
[![Status](https://img.shields.io/badge/Status-Complete%20%E2%80%94%2015%2F15%20steps-brightgreen?style=flat-square)](PROGRESS.md)
[![On-device](https://img.shields.io/badge/On--device-100%25-blue?style=flat-square)](#design-principles)

---

## What is Tough Talks?

The conversations that matter most — salary negotiations, relationship confrontations, difficult feedback — are the ones we're least prepared for. Tough Talks gives you a **private, on-device AI** that trains you before and guides you during.

**Practice Mode** → Adversarial persona simulation, pre-mortem, debrief, plan-vs-reality aftermath
**Live Mode** → Native-audio transcription + emotion radar from the same model

Everything runs on one Gemma 4 multimodal model loaded in one uvicorn process. Nothing leaves your device.

---

## Quick start — run the live demo

The whole stack — model, FastAPI backend, frontend, public tunnel — boots from a single Colab notebook:

1. Open [`notebooks/phase6/step15_demo_recording.ipynb`](notebooks/phase6/step15_demo_recording.ipynb) on Colab with a **T4 (or larger) GPU runtime**.
2. **Run all cells.** The notebook installs deps, clones the repo, sets `TOUGH_TALKS_INCLUDE_TEXT=0` (constrained-VRAM rule), launches `uvicorn`, waits for `/health` to go green, opens a `cloudflared` quick-tunnel (no signup), and prints a `https://*.trycloudflare.com/app/` URL plus a 10/10 pass-fail table.
3. Open the URL in any browser and drive the full flow:
   *Settings (leave Base URL empty) → Person Vault → Talk DNA → Pre-Mortem → 3-turn Practice Round → Debrief → Aftermath → Save → second round → Refresh Pulse → optional Live Mode audio upload.*

To run it locally instead:

```bash
git clone https://github.com/EhsanFarazmand/tough_talks.git
cd tough_talks
pip install -r requirements.txt
uvicorn backend.api.main:app --reload
# open http://127.0.0.1:8000/app/
```

You need a CUDA GPU with ~12 GB VRAM to load the multimodal Gemma 4 variant.

---

## Architecture at a glance

```
tough_talks/
├── frontend/          # Same-origin vanilla-JS app (app.html, app.js, app.css)
├── backend/
│   ├── api/           # FastAPI routes — /talk-dna /vault /persona /premortem
│   │                  #                  /debrief /aftermath /pulse
│   │                  #                  /transcribe /emotion /storage/*
│   └── core/_runtime/ # Per-component runtimes (Talk DNA, Person Vault, etc.)
├── notebooks/         # One notebook per step — Colab/Kaggle compatible
│   ├── phase1/        # Step 1-2  : Model access & prompt engineering
│   ├── phase2/        # Step 3-4  : On-device audio + emotion radar
│   ├── phase3/        # Step 5-8  : Talk DNA, Person Vault, Persona Sim, Pre-Mortem
│   ├── phase4/        # Step 9-11 : Debrief, Aftermath, Pulse
│   ├── phase5/        # Step 12-13: FastAPI + local JSON storage
│   └── phase6/        # Step 14-15: Frontend integration + live demo
├── data/
│   ├── schemas/       # JSON schemas — the contract between core and frontend
│   ├── prompts/       # Per-component prompt templates
│   └── local/         # Runtime storage (gitignored)
├── docs/              # Writeup + dev protocol + demo assets
├── scripts/           # Dev utilities (schema validation, etc.)
└── tests/             # Unit tests for every runtime + API surface
```

**Single-process design.** One uvicorn worker. One `ModelRegistry` lifespan-loaded with the multimodal Gemma 4 variant (`gemma-4-e2b-it`). Text routes fall back to the multimodal pair via `ModelRegistry.text()` — the multimodal class is a superset of text-only, so all eleven text + audio routes serve correctly behind one weight set. This was the fix for the T4 OOM trap (loading both variants triggers a silent `accelerate.big_modeling` CPU/disk offload).

---

## Status — all 15 steps complete

| Phase | Steps | Focus | Status |
|-------|-------|-------|--------|
| **Phase 1** | 1–2   | Model access + function calling                 | ✅ |
| **Phase 2** | 3–4   | Native audio transcription + emotion radar      | ✅ |
| **Phase 3** | 5–8   | Talk DNA, Person Vault, Persona Sim, Pre-Mortem | ✅ |
| **Phase 4** | 9–11  | Debrief, Aftermath, Relationship Pulse          | ✅ |
| **Phase 5** | 12–13 | FastAPI service + local JSON storage            | ✅ |
| **Phase 6** | 14–15 | Frontend integration + live demo                | ✅ |

See [PROGRESS.md](PROGRESS.md) for per-step notes (Colab-run results, prompt iterations, hypothesis confirmations).

### Step-by-step

```
Phase 1 — Model Access & Function Calling
  Step  1  Gemma 4 E2B baseline + native <|tool_call> protocol
  Step  2  Prompt engineering for conversation intelligence

Phase 2 — Audio & Emotion Layer
  Step  3  On-device speech transcription (Gemma 4 native audio, 28 s windows)
  Step  4  Emotion & prosody analysis (Emotion Radar)

Phase 3 — Core Intelligence Features
  Step  5  Talk DNA pattern detector (per-user)
  Step  6  Person Vault construction (per-counterparty, accumulating)
  Step  7  Adversarial persona simulator (multi-turn driver)
  Step  8  Pre-Mortem generator (three worst-case scenarios)

Phase 4 — Debrief & Memory Layer
  Step  9  Post-round debrief engine
  Step 10  Aftermath — plan vs. reality comparator
  Step 11  Relationship Pulse tracker (longitudinal, N ≥ 2 rounds)

Phase 5 — Backend API
  Step 12  FastAPI service exposing every component
  Step 13  Local JSON storage (atomic writes, path-traversal defence)

Phase 6 — Frontend Integration
  Step 14  Same-origin vanilla-JS app wired to every route
  Step 15  End-to-end live demo via uvicorn + cloudflared quick-tunnel
```

---

## Key innovation layers

| # | Module | What it does |
|---|--------|-------------|
| 01 | **Talk DNA** | Personal communication-pattern profile, accumulates across conversations |
| 02 | **Person Vault** | Behavioural model of a specific counterparty, qualitative lists grow over rounds |
| 03 | **Pre-Mortem** | Three worst-case scenarios before the talk, ranked by destabilisation risk |
| 04 | **Persona Sim** | Adversarial counterparty driven by the Vault profile across N turns |
| 05 | **Debrief** | Post-round wins / ground-lost turns / one fix for next time |
| 06 | **Aftermath** | Plan-vs-reality comparison — did each pre-mortem scenario materialise? |
| 07 | **Relationship Pulse** | Longitudinal trajectory across all rounds with one person |

---

## Design principles

- **Edge-first.** Gemma 4 runs on-device. Your divorce conversation never touches a server.
- **One model, many tasks.** Same `gemma-4-e2b-it` multimodal variant serves text intelligence, transcription, and audio emotion — no separate ASR.
- **Structured outputs only.** Every component emits JSON validated against a schema in [`data/schemas/`](data/schemas/).
- **Defence in depth.** Schema + runtime coercers + prompt rules. When the model emits a bad enum, both code and prompt catch it.
- **Compounding intelligence.** Talk DNA, Person Vault, and Pulse get more accurate with use because they accumulate locally.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Core model | `gemma-4-e2b-it` — any-to-any multimodal, ~11 GB bf16, single model for all tasks |
| Transcription | Gemma 4 native audio (no Whisper) — 28 s windows + 0.5 s overlap for clips > 30 s |
| Backend | FastAPI + Pydantic + `uvicorn[standard]` |
| Storage | Local JSON under `data/local/` — atomic writes, schema-validated |
| Frontend | Vanilla JS / HTML / CSS, served same-origin from `/app/` |
| Notebooks | Jupyter — Colab + Kaggle compatible |
| Tunnel (demo) | cloudflared quick-tunnel (no account, ephemeral URL) |

---

## Commands

```bash
# Install
pip install -r requirements.txt

# Lint + format (CI runs these on backend/ and tests/)
ruff check backend/ tests/ --ignore E501
black --check backend/ tests/

# Tests
pytest tests/unit/ -v --tb=short
pytest tests/integration/ -v --asyncio-mode=auto

# Validate every JSON schema is well-formed
python scripts/validate_schemas.py

# Run the API + frontend
uvicorn backend.api.main:app --reload
# → frontend at  http://127.0.0.1:8000/app/
# → API docs at  http://127.0.0.1:8000/docs
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) installs the lightweight stack (no torch/GPU) and runs lint, unit tests, schema validation, and notebook JSON validity. Heavy ML deps run locally / in Colab / on Kaggle.

---

## Hackathon submission

- **Writeup**: [`docs/writeup.md`](docs/writeup.md)
- **Live-demo notebook**: [`notebooks/phase6/step15_demo_recording.ipynb`](notebooks/phase6/step15_demo_recording.ipynb)
- **Demo assets**: [`docs/demo/`](docs/demo/)
- **Output contracts**: [`data/schemas/`](data/schemas/)
- **Dev protocol**: [`docs/dev_protocol.md`](docs/dev_protocol.md)

---

*Tough Talks · Kaggle Gemma 4 Hackathon · All processing local · Zero cloud dependency*
