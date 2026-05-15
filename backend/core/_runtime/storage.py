"""Local JSON-based storage layer for Tough Talks user data.

Phase 5 / Step 13. Persists every long-lived user artifact to the
filesystem as plain JSON. Storage is **local-only** — the API never
ships these files off the device, and the on-disk layout is the
contract that the frontend reads / writes via the ``/storage/*`` HTTP
routes (and that the next round of intelligence components consume as
``prior_profile`` / ``prior_pulse`` inputs).

Layout, rooted at :data:`DEFAULT_STORAGE_ROOT` (``<repo>/data/local`` —
gitignored; override via ``TOUGH_TALKS_STORAGE_ROOT``)::

    data/local/
    ├── talk_dna/<user_id>.json              # one per user (default: "local")
    ├── person_vault/<person_id>.json        # one per recurring counterparty
    ├── pulse/<person_id>.json               # one rolled-up pulse per relationship
    └── conversations/<round_id>.json        # one bundled round per file:
                                             # transcript + premortem + debrief
                                             # + aftermath + optional emotion_results

The bundled-conversation shape was chosen at step-13 planning time
over a split-per-artifact layout: the frontend renders a round-detail
view in one read, the file is the natural unit of versioning, and the
existing per-component schemas slot in unchanged as nested objects.
The wrapper schema lives at ``data/schemas/conversation.schema.json``.

Architectural shape mirrors prior runtime modules:

* Single-file module with one public ``StorageConfig`` dataclass + a
  small flat function surface (``save_*`` / ``load_*`` / ``list_*``).
  No class hierarchies — keeps the FastAPI dependency layer trivial
  (a route just calls the function with the configured root).
* **Atomic writes** — every ``save_*`` writes through a per-directory
  ``.tmp`` file, fsyncs, then renames into place. A crash mid-write
  leaves the previous version intact instead of a truncated file.
* **Defence-in-depth structural validation** — every write goes
  through :func:`_validate_against_schema` which checks ``required``
  fields + top-level type against the matching
  ``data/schemas/*.schema.json``. Same shape as the per-component
  runtime coercers: prompt teaches the model, code enforces the
  contract. The validator deliberately stays lightweight (no
  ``jsonschema`` runtime dep) — the per-component coercers already
  do the deep validation; this layer is the last "did you actually
  hand me the right shape?" check before the bytes hit disk. Read
  validation is best-effort + log-only so a schema evolution doesn't
  lock users out of older payloads.
* **Stable error class** — :class:`StorageError` is a ``ValueError``
  subclass to match the rest of the runtime (``TalkDNAAnalysisError``,
  ``PulseError``, etc.). The HTTP layer maps it to 422 / 404 as
  appropriate.

Speaker-label asymmetry from Step 12 still applies — the storage layer
does NOT normalise transcript shapes. A conversation record's
``transcript`` is stored verbatim; whoever wrote it picked the
``other``/``text`` or ``persona``/``reply`` shape and downstream
consumers (PersonVault vs debrief/aftermath/pulse) read what they
need.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "DEFAULT_STORAGE_ROOT",
    "DEFAULT_USER_ID",
    "ENV_STORAGE_ROOT",
    "StorageConfig",
    "StorageError",
    "StorageNotFoundError",
    "default_storage_root",
    "delete_conversation",
    "list_conversations",
    "list_person_vaults",
    "list_pulses",
    "load_conversation",
    "load_person_vault",
    "load_pulse",
    "load_talk_dna",
    "resolve_storage_root",
    "save_conversation",
    "save_person_vault",
    "save_pulse",
    "save_talk_dna",
]

LOG = logging.getLogger(__name__)

ENV_STORAGE_ROOT = "TOUGH_TALKS_STORAGE_ROOT"

# Repo-relative default. Resolved lazily in :func:`default_storage_root` so
# the constant survives test runs that monkeypatch ``__file__``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_ROOT: Path = _REPO_ROOT / "data" / "local"

# Mirrors :data:`backend.core._runtime.talk_dna.DEFAULT_USER_ID`. Kept
# duplicated to avoid an import cycle (talk_dna.py imports from
# generation.py / prompts.py / parsing.py only; pulling storage into
# the chain would re-import them indirectly).
DEFAULT_USER_ID = "local"

# Layout. Keep these in lock-step with :func:`_paths_for` below and the
# ``data/schemas/<name>.schema.json`` filenames.
_TALK_DNA_DIR = "talk_dna"
_PERSON_VAULT_DIR = "person_vault"
_PULSE_DIR = "pulse"
_CONVERSATION_DIR = "conversations"

# Schemas the storage layer enforces on write. The values map to the
# filenames under ``data/schemas/`` minus ``.schema.json``. The schemas
# are loaded lazily on first save (the read path is best-effort).
_SCHEMA_DIR = _REPO_ROOT / "data" / "schemas"
_SCHEMA_NAMES: dict[str, str] = {
    "talk_dna": "talk_dna",
    "person_vault": "person_vault",
    "pulse": "pulse",
    "conversation": "conversation",
}
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}

# Filename safety. Identifiers go into directory entries on disk, so
# anything outside a conservative charset is rejected up front — both
# to prevent path traversal and to keep the layout grep-friendly.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StorageError(ValueError):
    """Storage layer rejected the operation (validation / IO / shape).

    ``ValueError`` subclass to match the rest of the runtime
    (``TalkDNAAnalysisError``, ``PulseError``, etc.) — the HTTP layer
    already maps these to 422 on the analytical routes; the storage
    routes use the same translator.
    """


class StorageNotFoundError(StorageError):
    """The requested record does not exist on disk.

    Distinct from :class:`StorageError` so the HTTP layer can map it
    to 404 instead of 422.
    """


# ---------------------------------------------------------------------------
# Config + root resolution
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    """Where to read / write user data.

    ``root`` is the directory that holds the ``talk_dna/`` /
    ``person_vault/`` / ``pulse/`` / ``conversations/`` subdirs.
    Subdirs are created on demand by every ``save_*`` call so an empty
    root is a valid starting state.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)


def default_storage_root() -> Path:
    """Return the default repo-local storage root.

    Resolved lazily so test code can monkeypatch
    :data:`DEFAULT_STORAGE_ROOT` if it ever needs to.
    """
    return DEFAULT_STORAGE_ROOT


def resolve_storage_root(
    explicit: Optional[Path | str] = None,
) -> Path:
    """Pick the storage root with the precedence: arg > env > default.

    Tests pass ``explicit`` directly; production reads the
    ``TOUGH_TALKS_STORAGE_ROOT`` env var (so on-device deployments can
    point at a user-scoped directory like ``~/.tough_talks`` without
    code changes). Falls back to :func:`default_storage_root` when
    neither is set.

    The path is returned without being created — callers that need
    the directory should call ``Path.mkdir(parents=True, exist_ok=True)``.
    Every ``save_*`` does this for its own subdirectory.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_val = os.environ.get(ENV_STORAGE_ROOT)
    if env_val:
        return Path(env_val).expanduser().resolve()
    return default_storage_root()


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """ISO-8601 timestamp with timezone — used as ``updated_at`` default."""
    return datetime.now(timezone.utc).isoformat()


def _validate_id(value: str, *, field_name: str) -> str:
    """Reject identifiers that can't safely become filenames.

    Path-traversal defence + grep-friendliness. The cap of 128 chars
    keeps the on-disk layout fixed even when callers pass UUIDv4 with
    a prefix.
    """
    if not isinstance(value, str) or not _SAFE_ID_RE.match(value):
        raise StorageError(
            f"invalid {field_name}: must match {_SAFE_ID_RE.pattern}, got "
            f"{value!r}"
        )
    return value


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically.

    Uses ``tempfile.NamedTemporaryFile`` in the destination directory
    (the rename has to be cross-FS-safe), fsyncs the file descriptor,
    and renames into place. A crash mid-write leaves the previous
    version of the file intact.
    """
    parent = path.parent
    _ensure_dir(parent)
    # ``delete=False`` so we can rename after closing. We clean up the
    # temp file ourselves on any failure between open and rename.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # fsync isn't supported on every FS (some Colab tmp
                # mounts, some Windows network shares). The rename is
                # still atomic on the same FS; treat fsync as
                # best-effort.
                LOG.debug("fsync not supported for %s — relying on rename", tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup on failure. Don't mask the original error.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            LOG.warning("failed to clean up storage temp file %s", tmp_path)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StorageNotFoundError(f"no record at {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageError(f"corrupt JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageError(
            f"expected JSON object at {path}, got {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Lightweight schema validation
# ---------------------------------------------------------------------------


def _load_schema(name: str) -> dict[str, Any]:
    """Load and cache a schema from :data:`_SCHEMA_DIR`.

    Cached for the process lifetime — the schema files don't change
    between writes. Missing schema raises :class:`StorageError` (an
    install problem, not a user-data problem).
    """
    cached = _SCHEMA_CACHE.get(name)
    if cached is not None:
        return cached
    schema_path = _SCHEMA_DIR / f"{name}.schema.json"
    try:
        cached = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageError(
            f"schema file missing: {schema_path}. The storage layer "
            "validates writes against the per-component schemas; check the "
            "install."
        ) from exc
    except json.JSONDecodeError as exc:
        raise StorageError(f"schema file corrupt: {schema_path}: {exc}") from exc
    _SCHEMA_CACHE[name] = cached
    return cached


def _validate_against_schema(
    payload: dict[str, Any],
    *,
    schema_name: str,
) -> None:
    """Structural check on the top level only.

    Enforces:

    * payload is a JSON object,
    * every name in ``schema["required"]`` is present (and not ``None``).

    Does NOT do deep validation — that's the per-component runtime
    coercer's job (``_coerce_talk_dna_payload``, ``_coerce_pulse_payload``,
    etc.). The storage layer is the last "did you actually hand me the
    right shape?" check before the bytes hit disk; a violation here
    means the caller bypassed the runtime entirely, which is the
    only case worth shouting about.

    Raises :class:`StorageError` on violation.
    """
    if not isinstance(payload, dict):
        raise StorageError(
            f"{schema_name}: expected JSON object, got "
            f"{type(payload).__name__}"
        )
    schema = _load_schema(schema_name)
    required = schema.get("required") or []
    missing = [
        key for key in required
        if key not in payload or payload[key] is None
    ]
    if missing:
        raise StorageError(
            f"{schema_name}: required fields missing: {sorted(missing)}"
        )


def _validate_on_read(
    payload: dict[str, Any],
    *,
    schema_name: str,
    path: Path,
) -> None:
    """Best-effort check on read. Logs warnings, never raises."""
    try:
        _validate_against_schema(payload, schema_name=schema_name)
    except StorageError as exc:
        LOG.warning(
            "storage: %s does not match %s.schema.json: %s",
            path,
            schema_name,
            exc,
        )


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Paths:
    """Concrete directory paths under one storage root."""

    talk_dna: Path
    person_vault: Path
    pulse: Path
    conversations: Path


def _paths_for(root: Path) -> _Paths:
    return _Paths(
        talk_dna=root / _TALK_DNA_DIR,
        person_vault=root / _PERSON_VAULT_DIR,
        pulse=root / _PULSE_DIR,
        conversations=root / _CONVERSATION_DIR,
    )


# ---------------------------------------------------------------------------
# Talk DNA
# ---------------------------------------------------------------------------


def save_talk_dna(root: Path | str, payload: dict[str, Any]) -> Path:
    """Persist a TalkDNA dict to ``<root>/talk_dna/<user_id>.json``.

    ``user_id`` defaults to :data:`DEFAULT_USER_ID` (``"local"``) when
    absent — matching the on-device single-user shape. Stamps
    ``updated_at`` when the caller omitted it (defence-in-depth: the
    schema requires the field but a partial write should still
    succeed).

    Returns the absolute path the record was written to so the caller
    can log / surface it.
    """
    if not isinstance(payload, dict):
        raise StorageError(
            f"TalkDNA payload must be a dict, got {type(payload).__name__}"
        )
    # Default user_id before validation so the schema's required-field
    # check sees a populated value.
    payload = dict(payload)
    payload.setdefault("user_id", DEFAULT_USER_ID)
    payload.setdefault("updated_at", _now_iso())
    user_id = _validate_id(payload["user_id"], field_name="user_id")

    _validate_against_schema(payload, schema_name="talk_dna")
    paths = _paths_for(Path(root))
    target = paths.talk_dna / f"{user_id}.json"
    _atomic_write_json(target, payload)
    LOG.info("storage: wrote TalkDNA v%s for user_id=%s -> %s",
             payload.get("version"), user_id, target)
    return target


def load_talk_dna(
    root: Path | str,
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Read the TalkDNA dict for ``user_id`` from ``<root>``.

    Raises :class:`StorageNotFoundError` when there's no record on disk
    (so a 404 path is distinct from a 422 corruption path at the HTTP
    layer). Reads do best-effort schema validation — a warning is
    logged on mismatch but the dict is still returned, so a schema
    evolution doesn't lock users out of older payloads.
    """
    user_id = _validate_id(user_id, field_name="user_id")
    paths = _paths_for(Path(root))
    target = paths.talk_dna / f"{user_id}.json"
    payload = _read_json(target)
    _validate_on_read(payload, schema_name="talk_dna", path=target)
    return payload


# ---------------------------------------------------------------------------
# Person Vault
# ---------------------------------------------------------------------------


def _ensure_person_id(payload: dict[str, Any]) -> str:
    """Mint a stable ``person_id`` if the caller didn't supply one.

    The runtime's ``analyze_person_vault`` already mints one when
    building a fresh vault; this guard is for the case where a caller
    builds a payload by hand (notebook fixture, frontend draft). Same
    UUID-prefix shape PersonVault has used since Step 06
    (``person_<12hex>``) — keeps the layout single-format.
    """
    existing = payload.get("person_id")
    if existing:
        return _validate_id(existing, field_name="person_id")
    minted = f"person_{uuid.uuid4().hex[:12]}"
    payload["person_id"] = minted
    return minted


def save_person_vault(
    root: Path | str, payload: dict[str, Any]
) -> Path:
    """Persist a PersonVault dict to ``<root>/person_vault/<person_id>.json``.

    Mints ``person_id`` when missing. Stamps ``updated_at`` and
    defaults ``version`` to ``1`` so a notebook-built draft is
    writable without re-doing every coercer step.
    """
    if not isinstance(payload, dict):
        raise StorageError(
            f"PersonVault payload must be a dict, got {type(payload).__name__}"
        )
    payload = dict(payload)
    person_id = _ensure_person_id(payload)
    payload.setdefault("version", 1)
    payload.setdefault("conversation_count", payload.get("conversation_count", 1))
    payload.setdefault("updated_at", _now_iso())

    _validate_against_schema(payload, schema_name="person_vault")
    paths = _paths_for(Path(root))
    target = paths.person_vault / f"{person_id}.json"
    _atomic_write_json(target, payload)
    LOG.info(
        "storage: wrote PersonVault v%s for person_id=%s (name=%s) -> %s",
        payload.get("version"),
        person_id,
        payload.get("name"),
        target,
    )
    return target


def load_person_vault(
    root: Path | str, person_id: str
) -> dict[str, Any]:
    person_id = _validate_id(person_id, field_name="person_id")
    paths = _paths_for(Path(root))
    target = paths.person_vault / f"{person_id}.json"
    payload = _read_json(target)
    _validate_on_read(payload, schema_name="person_vault", path=target)
    return payload


def list_person_vaults(root: Path | str) -> list[dict[str, Any]]:
    """Return every PersonVault on disk, sorted by ``updated_at`` desc.

    Returns the full payloads (not just metadata) — the frontend's
    person list needs the ``name`` / ``relationship_type`` /
    ``communication_style`` / ``updated_at`` fields, and serialising
    the whole dict over the localhost loop is cheaper than fanning
    out N ``GET /storage/vault/{id}`` round-trips. The cap of 128
    chars on ``person_id`` keeps the directory listing fast even on
    a many-relationship user.
    """
    return _list_dir(
        directory=_paths_for(Path(root)).person_vault,
        schema_name="person_vault",
    )


# ---------------------------------------------------------------------------
# Pulse
# ---------------------------------------------------------------------------


def save_pulse(root: Path | str, payload: dict[str, Any]) -> Path:
    """Persist a Pulse dict to ``<root>/pulse/<person_id>.json``.

    Pulse is per-relationship — one file per ``person_id``, overwritten
    on each version bump. Stamps ``updated_at`` when absent.
    """
    if not isinstance(payload, dict):
        raise StorageError(
            f"Pulse payload must be a dict, got {type(payload).__name__}"
        )
    payload = dict(payload)
    raw_person_id = payload.get("person_id")
    if not raw_person_id:
        raise StorageError(
            "Pulse payload missing required 'person_id' — pulse is "
            "per-relationship and the file name is keyed on it"
        )
    person_id = _validate_id(raw_person_id, field_name="person_id")
    payload.setdefault("updated_at", _now_iso())

    _validate_against_schema(payload, schema_name="pulse")
    paths = _paths_for(Path(root))
    target = paths.pulse / f"{person_id}.json"
    _atomic_write_json(target, payload)
    LOG.info(
        "storage: wrote Pulse v%s for person_id=%s (rounds=%s, trend=%s) -> %s",
        payload.get("version"),
        person_id,
        payload.get("round_count"),
        payload.get("health_trend"),
        target,
    )
    return target


def load_pulse(root: Path | str, person_id: str) -> dict[str, Any]:
    person_id = _validate_id(person_id, field_name="person_id")
    paths = _paths_for(Path(root))
    target = paths.pulse / f"{person_id}.json"
    payload = _read_json(target)
    _validate_on_read(payload, schema_name="pulse", path=target)
    return payload


def list_pulses(root: Path | str) -> list[dict[str, Any]]:
    return _list_dir(
        directory=_paths_for(Path(root)).pulse,
        schema_name="pulse",
    )


# ---------------------------------------------------------------------------
# Conversations (bundled rounds)
# ---------------------------------------------------------------------------


def _ensure_round_id(payload: dict[str, Any]) -> str:
    existing = payload.get("round_id")
    if existing:
        return _validate_id(existing, field_name="round_id")
    minted = f"round_{uuid.uuid4().hex[:12]}"
    payload["round_id"] = minted
    return minted


def save_conversation(
    root: Path | str, payload: dict[str, Any]
) -> Path:
    """Persist a bundled conversation round.

    Path is ``<root>/conversations/<round_id>.json``. The bundle
    follows ``data/schemas/conversation.schema.json``:

    * top-level metadata: ``round_id`` / ``person_id`` / ``started_at``
      / ``user_goal`` / ``mode``;
    * the verbatim ``transcript`` array (whichever speaker-label
      convention the caller wrote it with — storage does NOT
      normalise);
    * nested ``premortem`` / ``debrief`` / ``aftermath`` artifacts as
      they were emitted by the corresponding routes;
    * optional ``emotion_results`` from the Live-mode emotion radar.

    Stamps ``started_at`` and ``updated_at`` when the caller omits
    them so a minimal draft is writable.
    """
    if not isinstance(payload, dict):
        raise StorageError(
            f"Conversation payload must be a dict, got {type(payload).__name__}"
        )
    payload = dict(payload)
    round_id = _ensure_round_id(payload)
    payload.setdefault("started_at", _now_iso())
    payload.setdefault("updated_at", _now_iso())

    _validate_against_schema(payload, schema_name="conversation")
    paths = _paths_for(Path(root))
    target = paths.conversations / f"{round_id}.json"
    _atomic_write_json(target, payload)
    LOG.info(
        "storage: wrote Conversation round_id=%s person_id=%s -> %s",
        round_id,
        payload.get("person_id"),
        target,
    )
    return target


def load_conversation(
    root: Path | str, round_id: str
) -> dict[str, Any]:
    round_id = _validate_id(round_id, field_name="round_id")
    paths = _paths_for(Path(root))
    target = paths.conversations / f"{round_id}.json"
    payload = _read_json(target)
    _validate_on_read(payload, schema_name="conversation", path=target)
    return payload


def list_conversations(
    root: Path | str,
    *,
    person_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List bundled rounds, optionally filtered by ``person_id``.

    Sorted by ``started_at`` desc (most recent first) so the frontend
    can render the round picker in display order without re-sorting.
    Falls back to ``updated_at`` then filename when ``started_at`` is
    missing — older drafts written without the field still surface.
    """
    if person_id is not None:
        person_id = _validate_id(person_id, field_name="person_id")
    items = _list_dir(
        directory=_paths_for(Path(root)).conversations,
        schema_name="conversation",
        sort_key="started_at",
    )
    if person_id is not None:
        items = [it for it in items if it.get("person_id") == person_id]
    return items


def delete_conversation(root: Path | str, round_id: str) -> None:
    """Remove a bundled round from disk.

    Conversations are the only entity type we expose a delete for —
    talk_dna / vault / pulse all accumulate versions and overwriting
    in place is the documented update pattern; conversation rounds
    are append-only individual events that a user may want to redact
    (a draft they don't want kept, a sensitive practice run). Raises
    :class:`StorageNotFoundError` when the round doesn't exist so the
    HTTP layer can 404 cleanly.
    """
    round_id = _validate_id(round_id, field_name="round_id")
    target = _paths_for(Path(root)).conversations / f"{round_id}.json"
    try:
        target.unlink()
    except FileNotFoundError as exc:
        raise StorageNotFoundError(f"no record at {target}") from exc
    LOG.info("storage: deleted Conversation round_id=%s", round_id)


# ---------------------------------------------------------------------------
# Shared list helper
# ---------------------------------------------------------------------------


def _list_dir(
    *,
    directory: Path,
    schema_name: str,
    sort_key: str = "updated_at",
) -> list[dict[str, Any]]:
    """List every ``*.json`` in ``directory`` as parsed dicts.

    Skips the dot-prefixed temp files :func:`_atomic_write_json`
    creates mid-write. Tolerates corrupt entries — logs a warning and
    drops the broken file rather than failing the whole list.
    """
    if not directory.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith(".") or not name.endswith(".json"):
            continue
        try:
            payload = _read_json(entry)
        except StorageError as exc:
            LOG.warning("storage: skipping corrupt record %s: %s", entry, exc)
            continue
        _validate_on_read(payload, schema_name=schema_name, path=entry)
        items.append(payload)
    # Most-recent first. Missing key sorts after everything else so
    # drafts without the field don't crowd out the canonical entries.
    items.sort(
        key=lambda d: (d.get(sort_key) or d.get("updated_at") or ""),
        reverse=True,
    )
    return items
