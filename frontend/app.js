// Tough Talks — frontend app wiring (Phase 6 / Step 14).
//
// Vanilla JS. Single-origin: the static bundle is served from /app/ by
// FastAPI, and every backend call is a relative URL — no CORS, no host
// config in production. The base-URL setting in the Settings card lets
// you point at a different host (Colab tunnel, separate uvicorn port)
// without rebuilding.

const STORAGE_KEYS = {
  baseUrl: "tt_api_base",
  userGoal: "tt_user_goal",
  userId: "tt_user_id",
  activePersonId: "tt_active_person_id",
};

const state = {
  baseUrl: localStorage.getItem(STORAGE_KEYS.baseUrl) || "",
  userGoal: localStorage.getItem(STORAGE_KEYS.userGoal) || "Have a constructive conversation about the open issue.",
  userId: localStorage.getItem(STORAGE_KEYS.userId) || "local",
  activePersonId: localStorage.getItem(STORAGE_KEYS.activePersonId) || null,
  activeVault: null,
  activeTalkDna: null,
  currentPremortem: null,
  currentRound: {
    roundId: null,
    startedAt: null,
    history: [],
  },
  lastDebrief: null,
  lastAftermath: null,
  lastPulse: null,
};

// ─── HTTP helpers ───────────────────────────────────────────────────────

function url(path) {
  const base = (state.baseUrl || "").replace(/\/$/, "");
  return `${base}${path}`;
}

async function handleResponse(res) {
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!res.ok) {
    const detail = (body && body.detail) || body || res.statusText;
    const err = new Error(typeof detail === "string" ? detail : (detail.error || `HTTP ${res.status}`));
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return body;
}

const http = {
  get:    (path)        => fetch(url(path), { method: "GET" }).then(handleResponse),
  post:   (path, body)  => fetch(url(path), {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }).then(handleResponse),
  put:    (path, body)  => fetch(url(path), {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }).then(handleResponse),
  delete: (path)        => fetch(url(path), { method: "DELETE" }).then(handleResponse),
  form:   (path, form)  => fetch(url(path), { method: "POST", body: form }).then(handleResponse),
};

// ─── API surface ────────────────────────────────────────────────────────

const api = {
  health: () => http.get("/health"),

  talkDna: {
    analyze: (body) => http.post("/talk-dna/analyze", body),
    get:     (userId = "local") => http.get(`/storage/talk-dna?user_id=${encodeURIComponent(userId)}`),
    save:    (body) => http.put("/storage/talk-dna", body),
  },

  vault: {
    build: (body) => http.post("/vault/build", body),
    list:  ()     => http.get("/storage/vault"),
    get:   (id)   => http.get(`/storage/vault/${encodeURIComponent(id)}`),
    save:  (id, body) => http.put(`/storage/vault/${encodeURIComponent(id)}`, body),
  },

  persona: {
    reply: (body) => http.post("/persona/reply", body),
  },

  premortem: (body) => http.post("/premortem", body),
  debrief:   (body) => http.post("/debrief", body),
  aftermath: (body) => http.post("/aftermath", body),

  pulse: {
    generate: (body) => http.post("/pulse", body),
    list:     ()     => http.get("/storage/pulse"),
    get:      (id)   => http.get(`/storage/pulse/${encodeURIComponent(id)}`),
    save:     (id, body) => http.put(`/storage/pulse/${encodeURIComponent(id)}`, body),
  },

  conversations: {
    list:   (personId) => http.get(`/storage/conversations${personId ? `?person_id=${encodeURIComponent(personId)}` : ""}`),
    get:    (id)       => http.get(`/storage/conversations/${encodeURIComponent(id)}`),
    save:   (id, body) => http.put(`/storage/conversations/${encodeURIComponent(id)}`, body),
    delete: (id)       => http.delete(`/storage/conversations/${encodeURIComponent(id)}`),
  },

  transcribe: (file, opts = {}) => {
    const form = new FormData();
    form.append("audio", file, file.name);
    if (opts.language)        form.append("language", opts.language);
    if (opts.max_new_tokens)  form.append("max_new_tokens", String(opts.max_new_tokens));
    return http.form("/transcribe", form);
  },

  emotion: (file, opts = {}) => {
    const form = new FormData();
    form.append("audio", file, file.name);
    form.append("speaker", opts.speaker || "user");
    if (opts.transcript_snippet) form.append("transcript_snippet", opts.transcript_snippet);
    if (opts.turn_id)            form.append("turn_id", opts.turn_id);
    if (opts.max_new_tokens)     form.append("max_new_tokens", String(opts.max_new_tokens));
    return http.form("/emotion/analyze", form);
  },
};

// ─── UI helpers ─────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function showError(title, err) {
  const box = document.createElement("div");
  box.className = "tt-error";
  const detail = err && err.detail
    ? (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail, null, 2))
    : (err && err.message ? err.message : String(err));
  box.innerHTML = `
    <button class="tt-error-close" aria-label="dismiss">×</button>
    <strong>${title}</strong>
    <span>${(err && err.message) ? escape(err.message) : ""}</span>
    <pre>${escape(detail)}</pre>
  `;
  box.querySelector(".tt-error-close").addEventListener("click", () => box.remove());
  $("errors").appendChild(box);
  setTimeout(() => box.remove(), 20000);
  console.error(title, err);
}

function escape(s) {
  if (typeof s !== "string") s = JSON.stringify(s, null, 2);
  return s.replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function setBusy(el, busy) {
  if (!el) return;
  el.classList.toggle("tt-busy", !!busy);
  if (busy) el.dataset.prevText = el.textContent;
  else if (el.dataset.prevText) { el.textContent = el.dataset.prevText; delete el.dataset.prevText; }
}

function parseTranscriptText(raw) {
  // Lightweight parser: one turn per non-empty line, split on the first ':'.
  // Lines starting with "you:" / "user:" map to {speaker:"user", text}; everything
  // else maps to {speaker:"other", text} so PersonVault consumes them directly.
  const out = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const colon = trimmed.indexOf(":");
    if (colon === -1) {
      out.push({ speaker: "other", text: trimmed });
      continue;
    }
    const label = trimmed.slice(0, colon).trim().toLowerCase();
    const text = trimmed.slice(colon + 1).trim();
    if (label === "you" || label === "user" || label === "me") {
      out.push({ speaker: "user", text });
    } else {
      out.push({ speaker: "other", text });
    }
  }
  return out;
}

function jsonPreview(obj) {
  if (obj == null) return "";
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

function nowIso() { return new Date().toISOString(); }

// ─── SETTINGS + HEALTH ──────────────────────────────────────────────────

function initSettings() {
  $("settings-base-url").value = state.baseUrl;
  $("settings-user-goal").value = state.userGoal;
  $("settings-base-url-save").addEventListener("click", () => {
    state.baseUrl = $("settings-base-url").value.trim();
    localStorage.setItem(STORAGE_KEYS.baseUrl, state.baseUrl);
    refreshHealth();
  });
  $("settings-user-goal").addEventListener("input", () => {
    state.userGoal = $("settings-user-goal").value;
    localStorage.setItem(STORAGE_KEYS.userGoal, state.userGoal);
  });
}

async function refreshHealth() {
  const dot = document.querySelector("#tt-health .tt-dot");
  const text = document.querySelector("#tt-health .tt-health-text");
  dot.dataset.state = "unknown";
  text.textContent = "checking…";
  try {
    const h = await api.health();
    const ok = h.status === "ok";
    const modelLoaded = h.text_model_loaded || h.multimodal_model_loaded;
    dot.dataset.state = ok ? (modelLoaded ? "ok" : "warn") : "down";
    text.textContent = ok
      ? `ok · text=${h.text_model_loaded} · multimodal=${h.multimodal_model_loaded}`
      : "backend down";
  } catch (err) {
    dot.dataset.state = "down";
    text.textContent = "backend down";
    console.warn("health check failed:", err);
  }
}

// ─── PERSON VAULT ───────────────────────────────────────────────────────

async function refreshVaultList() {
  try {
    const res = await api.vault.list();
    const sel = $("vault-existing");
    sel.innerHTML = '<option value="">— pick a stored vault —</option>';
    for (const v of (res.items || [])) {
      const opt = document.createElement("option");
      opt.value = v.person_id;
      opt.textContent = `${v.name || "Unnamed"} (${v.relationship_type || "—"}) · v${v.version}`;
      if (v.person_id === state.activePersonId) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (err) { showError("Failed to list vaults", err); }
}

async function loadSelectedVault() {
  const id = $("vault-existing").value;
  if (!id) return;
  try {
    const vault = await api.vault.get(id);
    state.activeVault = vault;
    state.activePersonId = vault.person_id;
    localStorage.setItem(STORAGE_KEYS.activePersonId, vault.person_id);
    $("vault-preview").textContent = jsonPreview(vault);
    updatePracticeStatus();
    refreshInsightsRounds();
  } catch (err) { showError("Failed to load vault", err); }
}

async function buildVault() {
  const turnsRaw = $("vault-transcript").value;
  if (!turnsRaw.trim()) { showError("Vault build", new Error("Paste a transcript first.")); return; }
  const turns = parseTranscriptText(turnsRaw);
  const hasOther = turns.some((t) => t.speaker === "other");
  if (!hasOther) {
    showError("Vault build", new Error("Need at least one counterparty turn (line not starting with 'you:' / 'user:')."));
    return;
  }
  const name = $("vault-name").value.trim() || "Other";
  const relInput = $("vault-rel").value.trim();
  const relationshipType = relInput || null;
  const btn = $("vault-build");
  setBusy(btn, true);
  try {
    const vault = await api.vault.build({
      turns, name, relationship_type: relationshipType,
      prior_profile: state.activeVault || null,
      max_new_tokens: 768,
    });
    state.activeVault = vault;
    state.activePersonId = vault.person_id;
    $("vault-preview").textContent = jsonPreview(vault);
    $("vault-save").disabled = false;
    updatePracticeStatus();
  } catch (err) { showError("Vault build failed", err); }
  finally { setBusy(btn, false); }
}

async function saveVault() {
  if (!state.activeVault) return;
  const btn = $("vault-save");
  setBusy(btn, true);
  try {
    const written = await api.vault.save(state.activeVault.person_id, state.activeVault);
    state.activeVault = written.payload || state.activeVault;
    state.activePersonId = state.activeVault.person_id;
    localStorage.setItem(STORAGE_KEYS.activePersonId, state.activePersonId);
    $("vault-preview").textContent = jsonPreview(state.activeVault);
    await refreshVaultList();
    updatePracticeStatus();
  } catch (err) { showError("Vault save failed", err); }
  finally { setBusy(btn, false); }
}

function initVault() {
  $("vault-refresh").addEventListener("click", refreshVaultList);
  $("vault-load").addEventListener("click", loadSelectedVault);
  $("vault-build").addEventListener("click", buildVault);
  $("vault-save").addEventListener("click", saveVault);
}

// ─── TALK DNA ───────────────────────────────────────────────────────────

async function loadTalkDna() {
  const userId = $("talkdna-user-id").value.trim() || "local";
  try {
    const dna = await api.talkDna.get(userId);
    state.activeTalkDna = dna;
    state.userId = userId;
    localStorage.setItem(STORAGE_KEYS.userId, userId);
    $("talkdna-preview").textContent = jsonPreview(dna);
  } catch (err) {
    if (err.status === 404) {
      $("talkdna-preview").textContent = "(no profile saved yet — analyse a transcript first)";
    } else {
      showError("TalkDNA load failed", err);
    }
  }
}

async function analyzeTalkDna() {
  const raw = $("talkdna-transcript").value;
  if (!raw.trim()) { showError("TalkDNA", new Error("Paste a transcript first.")); return; }
  const turns = parseTranscriptText(raw);
  const userId = $("talkdna-user-id").value.trim() || "local";
  const btn = $("talkdna-analyze");
  setBusy(btn, true);
  try {
    const dna = await api.talkDna.analyze({
      turns, user_id: userId,
      prior_profile: state.activeTalkDna || null,
      max_new_tokens: 768,
    });
    state.activeTalkDna = dna;
    $("talkdna-preview").textContent = jsonPreview(dna);
    $("talkdna-save").disabled = false;
  } catch (err) { showError("TalkDNA analyse failed", err); }
  finally { setBusy(btn, false); }
}

async function saveTalkDna() {
  if (!state.activeTalkDna) return;
  const btn = $("talkdna-save");
  setBusy(btn, true);
  try {
    const written = await api.talkDna.save(state.activeTalkDna);
    state.activeTalkDna = written.payload || state.activeTalkDna;
    $("talkdna-preview").textContent = jsonPreview(state.activeTalkDna);
  } catch (err) { showError("TalkDNA save failed", err); }
  finally { setBusy(btn, false); }
}

function initTalkDna() {
  $("talkdna-user-id").value = state.userId;
  $("talkdna-load").addEventListener("click", loadTalkDna);
  $("talkdna-analyze").addEventListener("click", analyzeTalkDna);
  $("talkdna-save").addEventListener("click", saveTalkDna);
}

// ─── PRE-MORTEM ─────────────────────────────────────────────────────────

function renderScenarios(premortem) {
  const box = $("premortem-scenarios");
  box.innerHTML = "";
  const scenarios = (premortem && premortem.failure_scenarios) || [];
  if (!scenarios.length) { box.textContent = "(no scenarios returned)"; return; }
  for (const s of scenarios) {
    const div = document.createElement("div");
    div.className = "tt-scenario";
    const params = s.simulation_parameters || {};
    div.innerHTML = `
      <div class="tt-scenario-head">
        <span class="tt-scenario-id">S${s.scenario_id}</span>
        <span class="tt-scenario-type">${escape(s.title || "")}</span>
        <span class="tt-scenario-risk">risk ${(s.destabilization_risk || 0).toFixed(2)}</span>
      </div>
      <p class="tt-scenario-desc">${escape(s.description || "")}</p>
      <div class="tt-scenario-meta">
        <strong>Likely trigger:</strong> ${escape(s.likely_trigger || "")}<br>
        <strong>Opening move:</strong> ${escape(s.opening_move || "")}<br>
        <strong>Resistance:</strong> ${escape(params.resistance_type || "")} · escalation ceiling ${(params.escalation_ceiling || 0).toFixed(2)}
      </div>
    `;
    box.appendChild(div);
  }
}

async function runPremortem() {
  const desc = $("premortem-desc").value.trim();
  if (!desc) { showError("Pre-mortem", new Error("Describe the conversation first.")); return; }
  const btn = $("premortem-run");
  setBusy(btn, true);
  try {
    const pm = await api.premortem({
      conversation_description: desc,
      user_goal: state.userGoal,
      person_profile: state.activeVault || {},
      talk_dna_profile: state.activeTalkDna || {},
      enable_thinking: true,
      max_new_tokens: 2048,
    });
    state.currentPremortem = pm;
    renderScenarios(pm);
  } catch (err) { showError("Pre-mortem failed", err); }
  finally { setBusy(btn, false); }
}

function initPremortem() {
  $("premortem-run").addEventListener("click", runPremortem);
}

// ─── PRACTICE ROUND ─────────────────────────────────────────────────────

function newRoundId() {
  const stamp = nowIso().replace(/[:.]/g, "-");
  return `round_${stamp}`;
}

function updatePracticeStatus() {
  const text = $("practice-status-text");
  const rid = $("practice-round-id");
  if (state.currentRound.roundId) {
    text.textContent = `Round in progress against ${state.activeVault?.name || "the persona"}.`;
    rid.textContent = state.currentRound.roundId;
  } else if (state.activeVault) {
    text.textContent = `Ready to start a round against ${state.activeVault.name}.`;
    rid.textContent = "";
  } else {
    text.textContent = "Pick a person and goal above, then start a round.";
    rid.textContent = "";
  }
}

function renderChat() {
  const chat = $("practice-chat");
  chat.innerHTML = "";
  for (const turn of state.currentRound.history) {
    const div = document.createElement("div");
    div.className = "tt-msg";
    if (turn.speaker === "user") {
      div.dataset.role = "user";
      div.textContent = turn.text;
    } else {
      div.dataset.role = "persona";
      div.innerHTML = `${escape(turn.reply || "")}
        <div class="tt-msg-meta">${escape(turn.persona_name || "persona")} · ${escape(turn.resistance_type || "")} · esc ${(turn.escalation_level || 0).toFixed(2)}</div>`;
    }
    chat.appendChild(div);
  }
  chat.scrollTop = chat.scrollHeight;
}

function startRound() {
  if (!state.activeVault) {
    showError("Start round", new Error("Load or build a Person Vault first."));
    return;
  }
  state.currentRound = {
    roundId: newRoundId(),
    startedAt: nowIso(),
    history: [],
  };
  state.lastDebrief = null;
  state.lastAftermath = null;
  $("practice-debrief").classList.add("tt-hidden");
  $("practice-aftermath-box").classList.add("tt-hidden");
  $("practice-end").disabled = true;
  $("practice-aftermath").disabled = true;
  $("practice-save").disabled = true;
  renderChat();
  updatePracticeStatus();
}

async function sendTurn() {
  if (!state.currentRound.roundId) { startRound(); }
  const input = $("practice-user-input");
  const msg = input.value.trim();
  if (!msg) return;
  state.currentRound.history.push({ speaker: "user", text: msg });
  input.value = "";
  renderChat();
  const btn = $("practice-send");
  setBusy(btn, true);
  try {
    const reply = await api.persona.reply({
      user_message: msg,
      persona_profile: state.activeVault || {},
      history: state.currentRound.history.slice(0, -1),
      user_goal: state.userGoal,
      max_new_tokens: 384,
    });
    state.currentRound.history.push({
      speaker: "persona",
      persona_name: reply.persona_name,
      reply: reply.reply,
      resistance_type: reply.resistance_type,
      escalation_level: reply.escalation_level,
    });
    renderChat();
    $("practice-end").disabled = false;
  } catch (err) { showError("Persona reply failed", err); }
  finally { setBusy(btn, false); }
}

async function endRound() {
  if (!state.currentRound.history.length) return;
  const btn = $("practice-end");
  setBusy(btn, true);
  try {
    const debrief = await api.debrief({
      transcript: state.currentRound.history,
      user_goal: state.userGoal,
      person_profile: state.activeVault || {},
      talk_dna_profile: state.activeTalkDna || {},
      enable_thinking: true,
      max_new_tokens: 2048,
    });
    state.lastDebrief = debrief;
    $("practice-debrief-body").textContent = jsonPreview(debrief);
    $("practice-debrief").classList.remove("tt-hidden");
    $("practice-aftermath").disabled = !state.currentPremortem;
    $("practice-save").disabled = false;
  } catch (err) { showError("Debrief failed", err); }
  finally { setBusy(btn, false); }
}

async function runAftermath() {
  if (!state.currentPremortem) return;
  const btn = $("practice-aftermath");
  setBusy(btn, true);
  try {
    const af = await api.aftermath({
      premortem: state.currentPremortem,
      transcript: state.currentRound.history,
      user_goal: state.userGoal,
      person_profile: state.activeVault || {},
      debrief: state.lastDebrief || {},
      enable_thinking: true,
      max_new_tokens: 4096,
    });
    state.lastAftermath = af;
    $("practice-aftermath-body").textContent = jsonPreview(af);
    $("practice-aftermath-box").classList.remove("tt-hidden");
  } catch (err) { showError("Aftermath failed", err); }
  finally { setBusy(btn, false); }
}

async function saveRound() {
  if (!state.currentRound.roundId) return;
  const btn = $("practice-save");
  setBusy(btn, true);
  try {
    const bundle = {
      round_id: state.currentRound.roundId,
      person_id: state.activeVault ? state.activeVault.person_id : undefined,
      person_name: state.activeVault ? state.activeVault.name : undefined,
      started_at: state.currentRound.startedAt,
      mode: "practice",
      user_goal: state.userGoal,
      transcript: state.currentRound.history,
      premortem: state.currentPremortem || undefined,
      debrief: state.lastDebrief || undefined,
      aftermath: state.lastAftermath || undefined,
      updated_at: nowIso(),
    };
    Object.keys(bundle).forEach((k) => bundle[k] === undefined && delete bundle[k]);
    await api.conversations.save(bundle.round_id, bundle);
    await refreshInsightsRounds();
  } catch (err) { showError("Save round failed", err); }
  finally { setBusy(btn, false); }
}

function initPractice() {
  $("practice-start").addEventListener("click", startRound);
  $("practice-send").addEventListener("click", sendTurn);
  $("practice-end").addEventListener("click", endRound);
  $("practice-aftermath").addEventListener("click", runAftermath);
  $("practice-save").addEventListener("click", saveRound);
}

// ─── LIVE MODE (audio) ──────────────────────────────────────────────────

async function liveTranscribe() {
  const file = $("live-audio").files[0];
  if (!file) { showError("Transcribe", new Error("Pick an audio file first.")); return; }
  const btn = $("live-transcribe");
  setBusy(btn, true);
  try {
    const out = await api.transcribe(file, { max_new_tokens: 256 });
    $("live-transcript").textContent = jsonPreview(out);
  } catch (err) { showError("Transcription failed", err); }
  finally { setBusy(btn, false); }
}

async function liveEmotion() {
  const file = $("live-audio").files[0];
  if (!file) { showError("Emotion", new Error("Pick an audio file first.")); return; }
  const speaker = $("live-speaker").value;
  const btn = $("live-emotion");
  setBusy(btn, true);
  try {
    const out = await api.emotion(file, { speaker, max_new_tokens: 384 });
    $("live-emotion-body").textContent = jsonPreview(out);
  } catch (err) { showError("Emotion failed", err); }
  finally { setBusy(btn, false); }
}

function initLive() {
  $("live-transcribe").addEventListener("click", liveTranscribe);
  $("live-emotion").addEventListener("click", liveEmotion);
}

// ─── INSIGHTS (rounds + pulse) ──────────────────────────────────────────

async function refreshInsightsRounds() {
  const target = $("insights-rounds");
  if (!state.activePersonId) {
    target.textContent = "(load a Person Vault first)";
    return;
  }
  try {
    const res = await api.conversations.list(state.activePersonId);
    if (!res.count) { target.textContent = "(no stored rounds yet)"; return; }
    target.textContent = (res.items || [])
      .map((r) => `${r.round_id}  ${r.started_at || ""}  ${r.mode || ""}`)
      .join("\n");
  } catch (err) { showError("Round list failed", err); }
}

async function refreshPulse() {
  if (!state.activePersonId) { showError("Pulse", new Error("Load a Person Vault first.")); return; }
  if (!state.activeVault) { showError("Pulse", new Error("No active vault in memory — click 'Load selected' first.")); return; }
  try {
    const listing = await api.conversations.list(state.activePersonId);
    const rounds = (listing.items || [])
      .filter((r) => r.aftermath || r.debrief)
      .map((r) => ({
        round_id: r.round_id,
        started_at: r.started_at,
        user_goal: r.user_goal,
        aftermath: r.aftermath || undefined,
        debrief: r.debrief || undefined,
      }));
    if (rounds.length < 2) {
      showError("Pulse", new Error(`Need at least 2 saved rounds with debrief/aftermath, found ${rounds.length}.`));
      return;
    }
    let priorPulse = null;
    try { priorPulse = await api.pulse.get(state.activePersonId); } catch (e) { /* 404 expected on cold start */ }
    const btn = $("insights-pulse-refresh");
    setBusy(btn, true);
    try {
      const pulse = await api.pulse.generate({
        rounds,
        person_profile: state.activeVault,
        prior_pulse: priorPulse,
        person_id: state.activePersonId,
        enable_thinking: true,
        max_new_tokens: 4096,
      });
      state.lastPulse = pulse;
      $("insights-pulse").textContent = jsonPreview(pulse);
      $("insights-pulse-save").disabled = false;
    } finally { setBusy(btn, false); }
  } catch (err) { showError("Pulse compute failed", err); }
}

async function loadPulse() {
  if (!state.activePersonId) { showError("Pulse", new Error("Load a Person Vault first.")); return; }
  try {
    const p = await api.pulse.get(state.activePersonId);
    state.lastPulse = p;
    $("insights-pulse").textContent = jsonPreview(p);
    $("insights-pulse-save").disabled = false;
  } catch (err) {
    if (err.status === 404) {
      $("insights-pulse").textContent = "(no saved pulse yet)";
    } else {
      showError("Pulse load failed", err);
    }
  }
}

async function savePulse() {
  if (!state.lastPulse || !state.activePersonId) return;
  const btn = $("insights-pulse-save");
  setBusy(btn, true);
  try {
    const written = await api.pulse.save(state.activePersonId, state.lastPulse);
    state.lastPulse = written.payload || state.lastPulse;
    $("insights-pulse").textContent = jsonPreview(state.lastPulse);
  } catch (err) { showError("Pulse save failed", err); }
  finally { setBusy(btn, false); }
}

function initInsights() {
  $("insights-rounds-refresh").addEventListener("click", refreshInsightsRounds);
  $("insights-pulse-refresh").addEventListener("click", refreshPulse);
  $("insights-pulse-load").addEventListener("click", loadPulse);
  $("insights-pulse-save").addEventListener("click", savePulse);
}

// ─── BOOT ───────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initSettings();
  initVault();
  initTalkDna();
  initPremortem();
  initPractice();
  initLive();
  initInsights();
  refreshHealth();
  await refreshVaultList();
  if (state.activePersonId) {
    try {
      const v = await api.vault.get(state.activePersonId);
      state.activeVault = v;
      $("vault-preview").textContent = jsonPreview(v);
      updatePracticeStatus();
      refreshInsightsRounds();
    } catch { /* missing — user picks a new one */ }
  }
  loadTalkDna().catch(() => {});
});
