# 🎙️ Tough Talks

> **Your private AI coach for every hard conversation**  
> Built on Gemma 4 · On-device · Zero cloud dependency

[![Gemma 4 Hackathon](https://img.shields.io/badge/Gemma%204-Good%20Hackathon-gold?style=flat-square)](https://)
[![Phase](https://img.shields.io/badge/Current%20Phase-1%20%E2%80%94%20Model%20Access-blue?style=flat-square)](#roadmap)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What is Tough Talks?

The conversations that matter most — salary negotiations, relationship confrontations, difficult feedback — are the ones we're least prepared for. Tough Talks gives you a **private, on-device AI** that trains you before and guides you during.

**Practice Mode** → Adversarial persona simulation before the conversation  
**Live Mode** → Real-time emotion radar + whisper coaching during it

Everything runs on Gemma 4. Nothing leaves your device.

---

## Architecture at a Glance

```
tough-talks/
├── frontend/          # UI (tough_talks_concept.html → full React app)
├── backend/           # FastAPI service + all intelligence components
│   ├── api/           # REST endpoints
│   └── core/          # Intelligence modules (Talk DNA, Person Vault, etc.)
├── notebooks/         # One notebook per implementation step
│   ├── phase1/        # Model access & function calling
│   ├── phase2/        # Audio & emotion layer
│   ├── phase3/        # Core intelligence features
│   ├── phase4/        # Debrief & memory layer
│   ├── phase5/        # Backend API
│   └── phase6/        # Frontend integration
├── data/              # Schemas, blueprints, sample data
├── scripts/           # Dev utilities
├── tests/             # Unit, integration, e2e
└── docs/              # Technical write-up, demo assets
```

---

## Roadmap

| Phase | Steps | Focus | Status |
|-------|-------|-------|--------|
| **Phase 1** | 1–2 | Model access + function calling | 🔵 Active |
| **Phase 2** | 3–4 | Audio & emotion layer | ⬜ Pending |
| **Phase 3** | 5–8 | Core intelligence features | ⬜ Pending |
| **Phase 4** | 9–11 | Debrief & memory layer | ⬜ Pending |
| **Phase 5** | 12–13 | Backend API | ⬜ Pending |
| **Phase 6** | 14–15 | Frontend integration + demo | ⬜ Pending |

### Step-by-Step Breakdown

```
Phase 1 — Model Access & Function Calling
  Step 01  Gemma 4 E2B baseline + function calling
  Step 02  Prompt engineering for conversation intelligence

Phase 2 — Audio & Emotion Layer
  Step 03  On-device speech transcription (Whisper / distil-whisper)
  Step 04  Emotion & prosody analysis (Emotion Radar)

Phase 3 — Core Intelligence Features
  Step 05  Talk DNA pattern detector
  Step 06  Person Vault construction
  Step 07  Adversarial persona simulator
  Step 08  Pre-Mortem generator

Phase 4 — Debrief & Memory Layer
  Step 09  Post-round debrief engine
  Step 10  Aftermath — plan vs. reality comparator
  Step 11  Relationship Pulse tracker

Phase 5 — Backend API
  Step 12  FastAPI service exposing all components
  Step 13  Local JSON-based storage schema

Phase 6 — Frontend Integration
  Step 14  Connect frontend to backend APIs
  Step 15  End-to-end demo recording
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/EhsanFarazmand/tough-talks.git
cd tough-talks

# 2. Install backend deps
pip install -r requirements.txt

# 3. Run a notebook (Colab / Kaggle compatible)
jupyter notebook notebooks/phase1/step1_gemma4_baseline_function_calling.ipynb

# 4. Start FastAPI (Phase 5+)
uvicorn backend.api.main:app --reload
```

---

## Design Principles

- **Edge-first** — Gemma 4 runs on-device. Your divorce conversation never touches a server.
- **Structured outputs** — Every component emits clean JSON for frontend integration.
- **Composable modules** — Each intelligence component is independently testable.
- **Compounding intelligence** — Talk DNA, Person Vault, and Relationship Pulse grow more accurate with use.

---

## Key Innovation Layers

| # | Module | What it does |
|---|--------|-------------|
| 01 | **Talk DNA** | Personal communication pattern profile, built over time |
| 02 | **Person Vault** | Behavioral models of recurring people in your life |
| 03 | **Pre-Mortem** | Simulates the worst-case scenario before you go in |
| 04 | **Conversation Blueprints** | Pre-built templates for 10 common tough talks |
| 05 | **Aftermath** | Post-conversation plan vs. reality comparison |
| 06 | **Relationship Pulse** | Longitudinal emotional arc of key relationships |
| 07 | **Cultural Calibration** | Communication style adapts to cultural context |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core model | Gemma 4 E2B-it — single model for all tiers (5B params, any-to-any multimodal, ~11 GB RAM in bf16 on Kaggle) |
| Transcription | Whisper / distil-whisper (on-device) |
| Backend | FastAPI + Pydantic |
| Storage | Local JSON (no database dependency) |
| Frontend | HTML/CSS/JS → React |
| Notebooks | Jupyter (Colab + Kaggle compatible) |

---

## Hackathon Submission

- **Working demo**: [`docs/demo/`](docs/demo/)
- **Technical write-up**: [`docs/writeup.md`](docs/writeup.md)
- **Demo video**: [`docs/demo/demo_video.mp4`](docs/demo/)
- **Concept frontend**: [`frontend/tough_talks_concept.html`](frontend/tough_talks_concept.html)

---

## Contributing

This is a hackathon project. Development follows a strict step-by-step protocol — see [`docs/dev_protocol.md`](docs/dev_protocol.md) for the execution rules.

---

*Tough Talks · Gemma 4 Good Hackathon · All processing local · Zero cloud dependency*
