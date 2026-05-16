"""Streaming helper for long-running model routes.

Cloudflare quick-tunnels enforce a ~100 s "time to first response byte"
edge timeout. Pre-mortem / debrief / aftermath / pulse all run
thinking-on analytical calls that routinely overshoot that on a T4 (per
``knowledge/phases/rules.md § Decoding knobs by task type``), so the
trycloudflare proxy returns 524 even though the model finishes
correctly.

We dodge it by:

* immediately emitting a single whitespace byte — resets the edge's
  TTFB clock so the 100 s rule never fires;
* running the runtime call on a background thread and emitting another
  whitespace byte every ``HEARTBEAT_INTERVAL_S`` seconds while it runs
  (keeps the connection live well under any edge / proxy idle cap);
* emitting the final JSON payload as the last chunk.

JSON tolerates leading whitespace, so the concatenated body still
parses cleanly via ``await res.json()`` on the browser side without a
custom stream reader — the frontend's existing ``handleResponse`` keeps
working, and ``TestClient.json()`` keeps working in tests.

Errors raised after streaming starts (e.g. the runtime's model-output
validation fails) can't be signalled as HTTP 422 because the 200 status
was already committed when the first whitespace byte left. We wrap the
diagnostic in a ``__tt_error__`` envelope as the final JSON payload,
and the frontend's ``handleResponse`` is taught to surface those as the
same shape as a real 422.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Type

from fastapi.responses import StreamingResponse

# 15 s leaves a 6-7x safety margin under cloudflare's ~100 s edge cap
# and cloudflared's ~90 s idle cap. Small enough that the worst-case
# extra post-completion latency (the final asyncio.wait timeout
# bucket) stays imperceptible next to a 60-180 s model call.
HEARTBEAT_INTERVAL_S = 15.0

__all__ = ["streaming_runtime_call", "HEARTBEAT_INTERVAL_S"]


def streaming_runtime_call(
    *,
    run: Callable[[], dict[str, Any]],
    component: str,
    runtime_error_cls: Type[Exception],
) -> StreamingResponse:
    """Wrap a synchronous runtime call as a heartbeat-padded JSON stream.

    Parameters
    ----------
    run
        Zero-arg callable that invokes the runtime, e.g.
        ``lambda: generate_premortem(processor, model, cfg=cfg)``.
        Executed in the default thread pool so the event loop stays
        free to emit heartbeats.
    component
        Component name surfaced in the ``__tt_error__`` envelope on
        runtime failure. Matches the existing ``runtime_error_to_http``
        shape so the frontend renders the same error UI regardless of
        whether the error came back as a 422 or as an inline envelope.
    runtime_error_cls
        The runtime's ``XxxError`` subclass (``PremortemError``,
        ``DebriefError``, etc.). Anything else propagates — Starlette
        will close the connection mid-stream and the browser will see a
        network error, which is the right shape for a genuine
        programming bug (distinct from "model returned junk").
    """

    async def _gen():
        # First byte ASAP — resets the cloudflare edge TTFB clock so
        # the 100 s rule never fires.
        yield b" "

        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, run)

        # Keepalive loop: wait up to HEARTBEAT_INTERVAL_S for the
        # future to resolve; if it doesn't, emit one whitespace byte
        # and loop. Using asyncio.wait (not wait_for) so the future's
        # own exception doesn't propagate here — we handle it
        # explicitly below via fut.result().
        while True:
            done, _pending = await asyncio.wait({fut}, timeout=HEARTBEAT_INTERVAL_S)
            if fut in done:
                break
            yield b" "

        try:
            result = fut.result()
        except runtime_error_cls as exc:
            detail: dict[str, Any] = {"component": component, "error": str(exc)}
            attempts = getattr(exc, "attempts", None)
            if isinstance(attempts, list):
                detail["attempts"] = attempts
            envelope = {"__tt_error__": {"status": 422, "detail": detail}}
            yield json.dumps(envelope).encode("utf-8")
            return

        yield json.dumps(result).encode("utf-8")

    return StreamingResponse(_gen(), media_type="application/json")
