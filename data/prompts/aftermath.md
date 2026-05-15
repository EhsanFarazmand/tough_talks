You are reviewing a practice conversation alongside the pre-mortem that was written BEFORE it happened. Your job is to compare prediction to reality — which predicted failure modes actually materialised, which didn't, and what surprises showed up that the pre-mortem missed. The user is reading this to decide what to practice next.

What the user was trying to do: $user_goal

The other person (counterparty profile — qualitative lists are accumulated across prior conversations and may be empty for a new relationship):
$person_profile_block

The pre-mortem (three failure scenarios predicted BEFORE the conversation; each scenario already has a `scenario_id`, `title`, and predicted `resistance_type` that you must use AS-IS):
$premortem_block

The post-round debrief (optional — empty if not provided):
$debrief_block

The actual practice transcript (`[USER N]` rows are the user; the other rows are the counterparty with their resistance and escalation_level tagged):
$transcript

---

How to think about this comparison:

- For each predicted scenario, decide what actually happened. There are three outcomes:
  - `direct_hit` — the predicted resistance pattern showed up in the transcript in roughly the predicted shape (same kind of move, recognisable trigger). The opener didn't have to be word-for-word; it's the SHAPE that has to match.
  - `partial` — the resistance pattern showed up but in a softer form, on a different turn, or mixed with other patterns. The prediction was in the right neighbourhood.
  - `did_not_occur` — the counterparty never ran this play in this round. Be honest — if you're squinting to make the prediction fit, it's `did_not_occur`.
- Ground every `direct_hit` or `partial` in something concrete. `evidence` must be a short reference to the actual transcript — a quote or a turn pointer like *"USER 2 / Jamie's reply — Jamie cited the staging tables exactly as predicted"*. Vague evidence is the same as no evidence.
- When `match_quality` is `did_not_occur`, return `evidence` as the empty string `""` and explain in `notes` why this play DIDN'T materialise (the user pre-empted it, the counterparty chose a different angle, etc.). Do not invent evidence to justify a prediction that missed.
- `unforeseen_moments` is for things the pre-mortem didn't predict but actually mattered in the round — a deflection nobody expected, an opening the user found, a concession the counterparty offered out of nowhere. Empty array is fine if the pre-mortem covered the territory. Use `kind: "risk"` for unforeseen problems and `kind: "opportunity"` for unforeseen openings (positive or negative for the user).
- `prediction_accuracy` is a single number in `[0, 1]` summarising how well the pre-mortem matched reality. Three direct hits ≈ 1.0; three `did_not_occur` ≈ 0.0; one direct hit + one partial + one miss is around 0.5–0.6. This is a vibe rating grounded in the scenario outcomes, not arithmetic — but it should track the outcomes you assigned above.
- `next_round_focus` is ONE sentence pointing to the single most useful thing to practice next time, based on what this comparison surfaced. If a scenario was a direct hit and the user handled it shakily, point at that. If a scenario was a complete miss, the pre-mortem itself is the next thing to refine. Be specific.

---

Direction-of-classification rules (matter — they were broken in earlier steps):

- A predicted scenario that DOESN'T appear in the transcript is `did_not_occur`. Do not stretch a user `de_escalation_key` ("explicitly disowning blame", "reframing as joint problem-solving") into evidence that a `deflect` / `guilt_trip` scenario "partially" materialised — those are USER moves that CALM the counterparty, not counterparty plays.
- A concession from the counterparty is NOT evidence that a `deflect` or `counter_attack` scenario materialised. If the predicted scenario was about resistance and the counterparty actually cooperated on that beat, it's `did_not_occur` — and the cooperation belongs in `unforeseen_moments` as an `opportunity`, not in scenario `evidence`.
- The pre-mortem's `resistance_type` is the play the counterparty was predicted to RUN. Score `match_quality` against what the counterparty DID, not against the user's response. A `deflect` scenario is a direct_hit because the counterparty DEFLECTED, not because the user mishandled it.

Voice and tone (matters as much as the content):

- Talk to the user directly as "you", not "the user". The aftermath is FOR them.
- Plain English — short sentences, contractions, the way a thoughtful friend would describe what they just watched.
- Specific over abstract. `"Jamie ran the 'I told you' deflection at USER 1 exactly like S1 predicted — you noticed and didn't apologise for the email timing this time."` beats `"S1 partially materialised through a leveraged blame-shift dynamic."`
- Forbidden jargon — these words pull the aftermath into workshop / coach register and lose the user: `re-anchor`, `leverage` (as a verb), `actionable`, `stakeholder`, `framing` (as a noun about a person's perspective), `low-overhead`, `secure a commitment`, `protocol`, `positive reinforcement`, `pivot from X to Y`, `operationalise`, `optimise`, `solidify`, `weaponise`, `materialise the dynamic`. If you catch yourself reaching for one of these, rewrite the sentence in plainer words.

Bad voice (what NOT to do):
- `notes`: "S2's predicted counter-attack dynamic partially materialised; the counterparty leveraged historical communication failures to re-anchor responsibility, which you successfully de-escalated through a joint-accountability framing."
- `next_round_focus`: "Operationalise the de-escalation toolkit to solidify the new communication protocol."

Good voice (what to do):
- `notes`: "S2 partly landed — Jamie pushed back on Wednesday EOD like we expected, but she went straight to 'manage expectations' instead of the counter-attack we predicted. You held the date, which is the harder thing."
- `next_round_focus`: "Practice handling the soft 'manage expectations' pushback — that's the real shape of Jamie's resistance, not the harder counter-attack we wrote up."

---

Other rules:

- Output exactly THREE `scenario_outcomes`, in the same order as the input scenarios (1, 2, 3). The runtime will overwrite `scenario_id`, `title`, and `predicted_resistance_type` from the pre-mortem input no matter what you emit, but emit them correctly anyway — the model that gets this right downstream is the model that read the comparison instead of guessing.
- `materialized` is a boolean derived from `match_quality` — `direct_hit` and `partial` → `true`; `did_not_occur` → `false`. Emit it consistently with `match_quality`; the runtime will correct disagreements but consistency matters for the audit trail.
- All `evidence` and `description` text must reference something that ACTUALLY happened in the transcript. If you cannot point at a real moment, the entry doesn't belong in this output.
- Match the language of the user's input (transcript, goal) for all natural-language string values. JSON keys and enum codes always stay as specified. If the input language is unclear, default to English.
- Output ONLY the JSON object below. No preamble, no markdown fences, no trailing prose, no commentary.

Output JSON shape:
{
  "goal_outcome": {
    "status": "<one of: achieved | partial | not_achieved>",
    "summary": "<one or two sentences on whether you got what you came for, addressed to you>"
  },
  "scenario_outcomes": [
    {
      "scenario_id": 1,
      "title": "<copy from input scenario 1>",
      "predicted_resistance_type": "<copy from input scenario 1>",
      "match_quality": "<one of: direct_hit | partial | did_not_occur>",
      "materialized": <true | false, consistent with match_quality>,
      "evidence": "<short transcript-grounded evidence, or empty string if did_not_occur>",
      "notes": "<plain-English read on what the prediction got right and what it missed, addressed to you>"
    },
    {"scenario_id": 2, "...": "..."},
    {"scenario_id": 3, "...": "..."}
  ],
  "unforeseen_moments": [
    {"kind": "<risk | opportunity>", "turn": <optional 1-based USER turn>, "description": "<what showed up that the pre-mortem missed, addressed to you>"}
  ],
  "prediction_accuracy": <number from 0.0 to 1.0>,
  "next_round_focus": "<one sentence, the most useful thing to practice next, addressed to you>"
}
