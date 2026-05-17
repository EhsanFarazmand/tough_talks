# Tough Talks

**Your private, on-device AI coach for the conversations that matter most — built end-to-end on Gemma 4.**

---

## Submission track

Primary: **Main Track.** Strong cross-fits in *Impact — Health & Sciences* (relationship and mental-health coaching), *Impact — Safety & Trust* (transparent, explainable, on-device outputs), and *Impact — Digital Equity & Inclusivity* (cultural calibration in prompts and personas).

## The problem

The conversations that decide most of our lives — a salary push, a difficult relationship talk, a tough piece of feedback — are the ones we're least rehearsed for. Cloud assistants are a non-starter for these: nobody wants their divorce conversation in a vendor's logs. Therapists and coaches are expensive, scheduled, and absent in the moment that matters. We wanted a coach that lives on your device, remembers you across months, and can be talked to in private the night before something hard.

## What we built

Tough Talks is a fully on-device coach with two modes:

- **Practice Mode** — load a real counterparty's behavioural profile, run a worst-case pre-mortem, simulate the conversation against an adversarial persona for three turns, then debrief and compare the plan against what actually happened.
- **Live Mode** — Gemma 4's native audio transcribes the room in 28-second windows and the *same* model returns an emotional read of prosody, pacing, and word choice.

Nothing leaves the device. The model, the JSON storage, the FastAPI service, and the static frontend all run inside one process on a single Colab T4 (or any equivalent edge GPU).

## How Gemma 4 is used

A single model — `gemma-4-e2b-it`, the any-to-any multimodal variant — is loaded once at boot and serves every text and audio route from one set of weights:

- Native **`<|tool_call>`** protocol for the function-calling path (Step 1).
- Native **audio** for `/transcribe` and `/emotion/analyze` — no separate Whisper or external ASR.
- **`enable_thinking=True`** for the analytical components (Pre-Mortem, Debrief, Aftermath, Pulse). We promoted this from hypothesis to a project rule after four independent task families showed it eliminates cross-field consistency violations.
- **`enable_thinking=False`** for the in-character adversarial persona — thinking-on consistently washed out voice toward a "balanced facilitator" register, the opposite of what the simulator needs.

Seven intelligence modules sit on top, each emitting JSON validated against a schema in `data/schemas/`:

| Module | Role |
|---|---|
| **Talk DNA** | User's communication-pattern profile, accumulating across conversations |
| **Person Vault** | Behavioural model of one counterparty, qualitative lists accumulate across rounds |
| **Persona Sim** | Multi-turn adversarial driver using the Vault profile |
| **Pre-Mortem** | Three worst-case scenario projections before the talk |
| **Debrief** | One-shot post-round analysis — wins, ground-lost turns, one fix next time |
| **Aftermath** | Plan-vs-reality comparison of pre-mortem to transcript |
| **Pulse** | Per-relationship rollup of N ≥ 2 rounds; longitudinal trajectory |

## Architecture

```
Single uvicorn process on edge GPU
├── lifespan-managed ModelRegistry (multimodal Gemma 4)
├── /app/        static frontend (vanilla JS, same-origin)
├── /storage/*   local JSON: vault, talk-dna, pulse, conversation bundles
├── /talk-dna /vault /persona /premortem /debrief /aftermath /pulse
├── /transcribe /emotion   multipart audio routes
└── /health
```

The frontend ships at `/app/` and consumes the same-origin API — no CORS, no tokens, no remote dependency. Storage is filesystem JSON under `data/local/` (gitignored), with atomic writes via `tempfile.mkstemp` + `os.replace`, path-traversal defence on every id, and schema validation on every write.

## Three technical choices that paid off

**1. One model, two registry handles.** A T4 won't fit both the text-only and multimodal Gemma 4 variants simultaneously — `accelerate.big_modeling` silently offloads the second to CPU/disk, and inference dies at the first request. We load only the multimodal variant and have `ModelRegistry.text()` fall back to the multimodal pair (the multimodal class is a superset — same LM, extra projection heads). All eleven text and audio routes serve correctly behind one weight set.

**2. Defence-in-depth around model outputs.** Every component runs a three-layer contract: a JSON schema, runtime coercers, and prompt rules. When the model emitted `resistance_type=silent` for a multi-sentence persona reply, the prompt rule alone was not enough — we added `_demote_silent_for_long_reply` in code (a 5-word ceiling plus an acceptance-phrase lexicon) and unit-tested the exact failure case. The same shape repeats across the stack: enum synonym maps (`aggressive → dominant`, `evasive → avoidant`), an apology-cue regex on Debrief's `over_apologies` label, transcript-render-prefix stripping on Aftermath evidence, `evidence_round_ids` filtered to known ids on Pulse. Schema is the source of truth, runtime is the watchdog, prompt is the first defence.

**3. Bundled conversation storage.** One JSON per round at `conversations/<round_id>.json` carries transcript + pre-mortem + debrief + aftermath + optional emotion results. Step 14's frontend integration became pure JS wiring because the storage routes were already a composable unit, and Pulse's rollup over N rounds is one read per `round_id` with no joins.

## Challenges overcome

- **Token-budget drift on analytical tasks.** Aftermath needed `max_new_tokens=4096` (double Pre-Mortem's 2048) because its prompt carries three context blocks instead of two. Promoted to a project rule: budget ≥ 2× the JSON body plus ~1024 per extra rendered context block.
- **Cross-field consistency failures.** On Pulse with `enable_thinking=False` we saw four distinct violations in one run — patterns silently dropped by the downstream `minItems: 2` filter, `health_trend=volatile` on a monotonic-up arc, a `trajectory_summary` reverse-engineered to justify the wrong trend, and a backward-looking `next_step_recommendation`. Thinking-on cleaned all four. That run was the fourth task-family confirmation behind the project rule.
- **Vocabulary drift on enums.** Mid-run, Person Vault's `communication_style` schema needed `assertive` / `empathetic` / `defensive` added *and* a synonym normaliser in code. Belt and braces.
- **Audio 30-second cap.** Gemma 4's native audio has a hard per-call ceiling. `transcribe_long` chunks at 28 s with 0.5 s overlap and stitches results.
- **Speaker-label asymmetry.** Person Vault consumes Step 03's `speaker=="other"` / `text` shape; everything else uses Step 07's `speaker=="persona"` / `reply` shape. We normalise once at the API boundary instead of papering over it in seven runtimes.

## Running the live demo

The demo runs end-to-end from a single Colab notebook — [`notebooks/phase6/step15_demo_recording.ipynb`](../notebooks/phase6/step15_demo_recording.ipynb) — with no account, no key, and no local install.

1. Open the Step 15 notebook on Colab with a T4 (or larger) runtime.
2. **Run all cells.** The notebook installs dependencies, clones the repo, sets `TOUGH_TALKS_INCLUDE_TEXT=0` (constrained-VRAM rule), launches `uvicorn backend.api.main:app` in the background, waits for `/health` to report `multimodal_model_loaded=true`, then opens a `cloudflared` quick-tunnel (no signup).
3. The final cell prints a `https://<random>.trycloudflare.com/app/` URL plus a 10/10 pass-fail table proving the full path — browser → Cloudflare edge → tunnel → uvicorn → FastAPI → static mount / route handler — actually works.
4. Open the URL in any browser:
   - **Settings** — leave Base URL empty (same-origin).
   - **Person Vault** → build "Jamie" → **Save**.
   - **Talk DNA** → analyse → **Save**.
   - **Pre-Mortem** → generate.
   - **Practice Round** → three user turns; the persona replies each time.
   - **Debrief** → **Aftermath** → **Save bundle**.
   - Run a second round, then **Refresh Pulse**.
   - Optional **Live Mode** — record a clip under 30 s and upload for `/transcribe` + `/emotion/analyze`.

All artefacts saved during the demo land in `data/local/` on the Colab VM and can be copied to Drive before the runtime expires.

## Why it matters

Gemma 4's real advantage in this domain is not raw intelligence — it's **persistent, private personalisation**. A cloud assistant that forgets you between sessions cannot model your apology rate, your specific counterparty's deflection patterns, or the trajectory of one relationship over months. Tough Talks does, because it never has to ask permission to remember.

---

## Attachments

- **GitHub:** https://github.com/EhsanFarazmand/tough_talks
- **Demo video:** [`docs/demo/Gemma4Good.mp4`](https://youtu.be/vLpLSa11Qsw)
- **Offline Practice-flow capture:** [one-click render](https://htmlpreview.github.io/?https://github.com/EhsanFarazmand/tough_talks/blob/main/docs/demo/Tough%20Talks%20%E2%80%94%20App_Sallary%20Negotiation.html)
- **Offline product-concept page:** [one-click render](https://htmlpreview.github.io/?https://github.com/EhsanFarazmand/tough_talks/blob/main/docs/demo/Tough%20Talks%20%E2%80%94%20Product%20Concept.html)
- **Live-demo notebook:** [`notebooks/phase6/step15_demo_recording.ipynb`](../notebooks/phase6/step15_demo_recording.ipynb)
- **Output contracts (JSON schemas):** [`data/schemas/`](../data/schemas/)
- **Development protocol:** [`docs/dev_protocol.md`](dev_protocol.md)
