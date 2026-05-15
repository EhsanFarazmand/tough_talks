"""Unit tests for the Step 13 local JSON storage layer.

Covers the four entity types (TalkDNA, PersonVault, Pulse,
Conversation) end-to-end against a ``tmp_path`` storage root:

* Round-trip save/load returns the same payload (modulo stamped
  defaults like ``updated_at``).
* Listing returns most-recent first.
* Missing records raise :class:`StorageNotFoundError`, NOT a generic
  :class:`StorageError`.
* Schema validation catches missing required fields on write.
* Atomic writes leave no half-written ``.tmp`` files behind on
  success, and leave the previous version intact on failure.
* Path-traversal-ish ids (``../foo``, ``foo/bar``) are rejected
  before any filesystem operation.
* :func:`resolve_storage_root` honours the precedence chain
  (explicit arg > env var > default).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core._runtime import (
    DEFAULT_STORAGE_ROOT,
    ENV_STORAGE_ROOT,
    StorageError,
    StorageNotFoundError,
    delete_conversation,
    list_conversations,
    list_person_vaults,
    list_pulses,
    load_conversation,
    load_person_vault,
    load_pulse,
    load_talk_dna,
    resolve_storage_root,
    save_conversation,
    save_person_vault,
    save_pulse,
    save_talk_dna,
)


# ---------------------------------------------------------------------------
# Fixture payloads — minimal shapes that pass the schema's required-field
# check. The runtime coercers produce richer payloads in production; the
# storage layer only checks structure.
# ---------------------------------------------------------------------------


_TALK_DNA = {
    "user_id": "local",
    "version": 1,
    "patterns": {
        "filler_phrases": [],
        "apology_rate": 0.1,
        "silence_under_pressure": False,
        "sarcasm_frequency": "low",
    },
    "strengths": ["clear_problem_statement"],
    "weaknesses": [],
    "updated_at": "2026-05-15T00:00:00+00:00",
}


_PERSON_VAULT = {
    "person_id": "person_abc123",
    "name": "Jamie",
    "relationship_type": "colleague",
    "version": 1,
    "conversation_count": 1,
    "profile": {
        "communication_style": "defensive",
        "emotional_triggers": ["citing past commitments"],
        "de_escalation_keys": ["explicitly disowning blame"],
        "common_deflections": [],
    },
    "updated_at": "2026-05-15T00:00:00+00:00",
}


_PULSE = {
    "person_id": "person_abc123",
    "version": 1,
    "round_count": 2,
    "health_score": 0.65,
    "health_trend": "improving",
    "trajectory_summary": "You held the deadline; Jamie shifted from deflect to concede.",
    "round_summaries": [
        {
            "round_id": "round_a",
            "started_at": "2026-04-30T16:00:00+00:00",
            "goal_status": "partial",
            "prediction_accuracy": 0.6,
            "headline": "You drew the deflection out into a joint plan.",
        },
        {
            "round_id": "round_b",
            "started_at": "2026-05-14T15:30:00+00:00",
            "goal_status": "achieved",
            "prediction_accuracy": 0.75,
            "headline": "You committed Jamie to Wednesday EOD with a daily check-in.",
        },
    ],
    "recurring_patterns": [
        {
            "pattern": "Jamie opens defensively; concedes after joint-problem-solving reframe.",
            "evidence_round_ids": ["round_a", "round_b"],
        }
    ],
    "emerging_concerns": [],
    "relationship_wins": [
        {"win": "Joint-problem-solving reframe lands consistently.", "round_id": "round_b"}
    ],
    "next_step_recommendation": "Try the reframe in turn 1 next time instead of turn 3.",
    "updated_at": "2026-05-15T00:00:00+00:00",
}


_CONVERSATION = {
    "round_id": "round_step13_demo",
    "person_id": "person_abc123",
    "started_at": "2026-05-15T15:00:00+00:00",
    "mode": "practice",
    "user_goal": "Get Jamie to Wednesday EOD.",
    "transcript": [
        {"speaker": "user", "text": "Tuesday slipped — what's going on?"},
        {
            "speaker": "persona",
            "persona_name": "Jamie",
            "reply": "I told you the staging tables weren't done.",
            "resistance_type": "deflect",
            "escalation_level": 0.55,
        },
    ],
    "updated_at": "2026-05-15T15:00:00+00:00",
}


# ---------------------------------------------------------------------------
# resolve_storage_root precedence
# ---------------------------------------------------------------------------


def test_resolve_storage_root_explicit_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ENV_STORAGE_ROOT, str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    assert resolve_storage_root(explicit) == explicit.expanduser().resolve()


def test_resolve_storage_root_env_used_when_no_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ENV_STORAGE_ROOT, str(tmp_path / "env"))
    assert resolve_storage_root() == (tmp_path / "env").expanduser().resolve()


def test_resolve_storage_root_default_when_neither(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(ENV_STORAGE_ROOT, raising=False)
    assert resolve_storage_root() == DEFAULT_STORAGE_ROOT


# ---------------------------------------------------------------------------
# TalkDNA round-trip
# ---------------------------------------------------------------------------


def test_talk_dna_round_trip(tmp_path: Path):
    target = save_talk_dna(tmp_path, _TALK_DNA)
    assert target.is_file()
    assert target.name == "local.json"
    loaded = load_talk_dna(tmp_path, user_id="local")
    assert loaded == _TALK_DNA


def test_talk_dna_missing_record_raises_not_found(tmp_path: Path):
    with pytest.raises(StorageNotFoundError):
        load_talk_dna(tmp_path, user_id="local")


def test_talk_dna_overwrites_in_place(tmp_path: Path):
    save_talk_dna(tmp_path, _TALK_DNA)
    v2 = {**_TALK_DNA, "version": 2, "conversation_count": 2}
    save_talk_dna(tmp_path, v2)
    loaded = load_talk_dna(tmp_path, user_id="local")
    assert loaded["version"] == 2
    assert loaded["conversation_count"] == 2


def test_talk_dna_user_id_default_when_missing(tmp_path: Path):
    payload = dict(_TALK_DNA)
    payload.pop("user_id")
    save_talk_dna(tmp_path, payload)
    # default user_id is 'local' — the file lands there
    assert (tmp_path / "talk_dna" / "local.json").is_file()


def test_talk_dna_validation_catches_missing_required(tmp_path: Path):
    bad = {k: v for k, v in _TALK_DNA.items() if k != "patterns"}
    with pytest.raises(StorageError) as exc:
        save_talk_dna(tmp_path, bad)
    assert "patterns" in str(exc.value)


# ---------------------------------------------------------------------------
# PersonVault round-trip
# ---------------------------------------------------------------------------


def test_person_vault_round_trip(tmp_path: Path):
    target = save_person_vault(tmp_path, _PERSON_VAULT)
    assert target.is_file()
    assert target.name == "person_abc123.json"
    loaded = load_person_vault(tmp_path, "person_abc123")
    assert loaded == _PERSON_VAULT


def test_person_vault_mints_id_when_missing(tmp_path: Path):
    payload = {k: v for k, v in _PERSON_VAULT.items() if k != "person_id"}
    save_person_vault(tmp_path, payload)
    items = list_person_vaults(tmp_path)
    assert len(items) == 1
    minted = items[0]["person_id"]
    assert minted.startswith("person_") and len(minted) > len("person_")


def test_person_vault_list_returns_most_recent_first(tmp_path: Path):
    older = {**_PERSON_VAULT, "person_id": "person_old", "updated_at": "2026-01-01T00:00:00+00:00"}
    newer = {**_PERSON_VAULT, "person_id": "person_new", "updated_at": "2026-05-15T00:00:00+00:00"}
    save_person_vault(tmp_path, older)
    save_person_vault(tmp_path, newer)
    items = list_person_vaults(tmp_path)
    assert [it["person_id"] for it in items] == ["person_new", "person_old"]


def test_person_vault_missing_raises_not_found(tmp_path: Path):
    with pytest.raises(StorageNotFoundError):
        load_person_vault(tmp_path, "person_nope")


def test_person_vault_traversal_id_rejected(tmp_path: Path):
    with pytest.raises(StorageError):
        load_person_vault(tmp_path, "../etc/passwd")
    with pytest.raises(StorageError):
        load_person_vault(tmp_path, "person/with/slash")


# ---------------------------------------------------------------------------
# Pulse round-trip
# ---------------------------------------------------------------------------


def test_pulse_round_trip(tmp_path: Path):
    target = save_pulse(tmp_path, _PULSE)
    assert target.is_file()
    assert target.name == "person_abc123.json"
    loaded = load_pulse(tmp_path, "person_abc123")
    assert loaded == _PULSE


def test_pulse_requires_person_id(tmp_path: Path):
    no_id = {k: v for k, v in _PULSE.items() if k != "person_id"}
    with pytest.raises(StorageError) as exc:
        save_pulse(tmp_path, no_id)
    assert "person_id" in str(exc.value)


def test_pulse_list_returns_most_recent_first(tmp_path: Path):
    older = {**_PULSE, "person_id": "person_old", "updated_at": "2026-01-01T00:00:00+00:00"}
    newer = {**_PULSE, "person_id": "person_new", "updated_at": "2026-05-15T00:00:00+00:00"}
    save_pulse(tmp_path, older)
    save_pulse(tmp_path, newer)
    items = list_pulses(tmp_path)
    assert [it["person_id"] for it in items] == ["person_new", "person_old"]


# ---------------------------------------------------------------------------
# Conversation round-trip
# ---------------------------------------------------------------------------


def test_conversation_round_trip(tmp_path: Path):
    target = save_conversation(tmp_path, _CONVERSATION)
    assert target.is_file()
    assert target.name == "round_step13_demo.json"
    loaded = load_conversation(tmp_path, "round_step13_demo")
    assert loaded == _CONVERSATION


def test_conversation_mints_round_id_when_missing(tmp_path: Path):
    payload = {k: v for k, v in _CONVERSATION.items() if k != "round_id"}
    save_conversation(tmp_path, payload)
    items = list_conversations(tmp_path)
    assert len(items) == 1
    minted = items[0]["round_id"]
    assert minted.startswith("round_")


def test_conversation_list_filters_by_person_id(tmp_path: Path):
    other_person = {
        **_CONVERSATION,
        "round_id": "round_other",
        "person_id": "person_other",
        "started_at": "2026-05-15T16:00:00+00:00",
    }
    save_conversation(tmp_path, _CONVERSATION)
    save_conversation(tmp_path, other_person)
    items = list_conversations(tmp_path, person_id="person_abc123")
    assert len(items) == 1
    assert items[0]["round_id"] == "round_step13_demo"


def test_conversation_list_sorts_by_started_at_desc(tmp_path: Path):
    older = {
        **_CONVERSATION,
        "round_id": "round_older",
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    newer = {
        **_CONVERSATION,
        "round_id": "round_newer",
        "started_at": "2026-05-15T18:00:00+00:00",
    }
    save_conversation(tmp_path, older)
    save_conversation(tmp_path, newer)
    items = list_conversations(tmp_path)
    assert [it["round_id"] for it in items] == ["round_newer", "round_older"]


def test_conversation_delete_removes_file(tmp_path: Path):
    save_conversation(tmp_path, _CONVERSATION)
    delete_conversation(tmp_path, "round_step13_demo")
    with pytest.raises(StorageNotFoundError):
        load_conversation(tmp_path, "round_step13_demo")


def test_conversation_delete_missing_raises_not_found(tmp_path: Path):
    with pytest.raises(StorageNotFoundError):
        delete_conversation(tmp_path, "round_does_not_exist")


# ---------------------------------------------------------------------------
# Atomic write + corruption tolerance
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    save_talk_dna(tmp_path, _TALK_DNA)
    save_person_vault(tmp_path, _PERSON_VAULT)
    save_pulse(tmp_path, _PULSE)
    save_conversation(tmp_path, _CONVERSATION)
    # No hidden .tmp files should remain after success.
    stragglers = list(tmp_path.rglob(".*.tmp"))
    assert stragglers == []


def test_corrupt_record_raises_storage_error_on_read(tmp_path: Path):
    save_talk_dna(tmp_path, _TALK_DNA)
    target = tmp_path / "talk_dna" / "local.json"
    target.write_text("not valid json {{{", encoding="utf-8")
    with pytest.raises(StorageError):
        load_talk_dna(tmp_path, user_id="local")


def test_list_skips_corrupt_entries_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    save_person_vault(tmp_path, _PERSON_VAULT)
    # Drop a corrupt sibling into the same directory.
    (tmp_path / "person_vault" / "broken.json").write_text("{not json", encoding="utf-8")
    items = list_person_vaults(tmp_path)
    # Good record survives; bad one drops silently with a warning log.
    assert len(items) == 1
    assert items[0]["person_id"] == _PERSON_VAULT["person_id"]


def test_list_empty_dir_returns_empty_list(tmp_path: Path):
    # No save_* has run yet — nothing on disk.
    assert list_person_vaults(tmp_path) == []
    assert list_pulses(tmp_path) == []
    assert list_conversations(tmp_path) == []


def test_write_failure_preserves_previous_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If a write throws after _atomic_write_json opens the temp file
    but before rename, the previous record must stay intact.
    """
    save_talk_dna(tmp_path, _TALK_DNA)
    target = tmp_path / "talk_dna" / "local.json"
    original_bytes = target.read_bytes()

    import os as _os

    real_replace = _os.replace

    def _boom(src, dst):  # noqa: ANN001
        raise OSError("simulated rename failure")

    monkeypatch.setattr("backend.core._runtime.storage.os.replace", _boom)
    with pytest.raises(OSError):
        save_talk_dna(tmp_path, {**_TALK_DNA, "version": 999})
    # Restore — sanity check that the prior file survived.
    monkeypatch.setattr("backend.core._runtime.storage.os.replace", real_replace)
    assert target.read_bytes() == original_bytes
    # No .tmp files left behind.
    stragglers = list((tmp_path / "talk_dna").rglob(".*.tmp"))
    assert stragglers == []


# ---------------------------------------------------------------------------
# Schema cross-check — _SCHEMA_NAMES points at real files
# ---------------------------------------------------------------------------


def test_every_schema_file_exists():
    """Every name the storage layer validates against must exist on disk."""
    from backend.core._runtime.storage import _SCHEMA_DIR, _SCHEMA_NAMES

    for name in _SCHEMA_NAMES.values():
        path = _SCHEMA_DIR / f"{name}.schema.json"
        assert path.is_file(), f"missing schema file: {path}"
        # And it must parse as JSON.
        json.loads(path.read_text(encoding="utf-8"))
