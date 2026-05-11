# Development Protocol

## Execution Rules

This project follows a strict step-by-step execution protocol.

### The One-Step Rule

**Only one step is active at a time.**

Do not start Step N+1 until Step N is:
1. Implemented and running
2. Tested in isolation (notebook or unit test)
3. Output reviewed and approved

### Step Lifecycle

```
PENDING → ACTIVE → REVIEW → DONE
                       ↓
                   BLOCKED (document the issue, fix before moving forward)
```

### After Each Step

Before marking a step DONE, confirm:
- [ ] Notebook runs end-to-end in Colab/Kaggle without errors
- [ ] Output matches the expected JSON schema (if applicable)
- [ ] Key observations added to `/knowledge/`
- [ ] `PROGRESS.md` updated

---

## Notebook Conventions

Every step lives in a single Jupyter notebook under `notebooks/phaseN/`.

### Cell Structure (mandatory order)

```
Cell 1 — Step header (Markdown)
Cell 2 — Install deps (pip installs, colab/kaggle env check)
Cell 3 — Imports
Cell 4 — Config (model path, hyperparams, paths — all in one place)
Cell 5+ — Implementation
Last cell — Structured output / validation (print JSON or assert shape)
```

### Naming Convention

```
notebooks/phase1/step01_gemma4_baseline.ipynb
notebooks/phase2/step03_gemma4_audio_transcription.ipynb
```

Always use the global step number (01–15), not per-phase numbering.

---

## Output Requirements

Every component that feeds the frontend must emit structured JSON.

### Example: Talk DNA output

```json
{
  "user_id": "local",
  "version": 1,
  "patterns": {
    "filler_phrases": ["I just feel like", "kind of", "sort of"],
    "apology_rate": 0.34,
    "silence_under_pressure": true,
    "sarcasm_frequency": "low"
  },
  "strengths": ["clear_problem_statement", "listens_actively"],
  "weaknesses": ["over_apologizes", "hedges_before_vulnerable_statements"],
  "updated_at": "2025-01-01T00:00:00Z"
}
```

Schemas live in `data/schemas/`. Every component must validate against its schema before the step is marked DONE.

---

## Git Workflow

### Branch naming

```
feature/step-01-gemma4-baseline
feature/step-03-audio-transcription
fix/step-05-talk-dna-pattern-edge-case
```

### Commit format

```
[step-01] Add Gemma 4 E2B baseline notebook
[step-01] Add function calling schema for emotion API
[fix] Handle empty transcript in Talk DNA detector
[docs] Update PROGRESS.md — Step 01 complete
```

### What NOT to commit

- Model weights (`*.gguf`, `*.bin`, `*.safetensors`)
- Audio recordings (`*.wav`, `*.mp3`)
- User data (`talk_dna_*.json`, `person_vault_*.json`)
- API keys or secrets

---

## Testing Standards

### Unit tests (every component)

```
tests/unit/test_talk_dna.py
tests/unit/test_person_vault.py
tests/unit/test_emotion_radar.py
```

### Integration tests (Phase 5+)

```
tests/integration/test_api_debrief_endpoint.py
```

### Running tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v --asyncio-mode=auto
```

---

## Knowledge Management

After each step, extract insights into `/knowledge/`:

```
/knowledge/
  INDEX.md              ← routing map
  phases/               ← step-by-step learnings
  architecture/         ← design decisions & tradeoffs
  ml/                   ← model behavior observations
  api/                  ← endpoint design learnings
```

See `/knowledge/INDEX.md` for the full routing guide.
