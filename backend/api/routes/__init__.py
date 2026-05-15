"""FastAPI routers for each Tough Talks intelligence component.

One router per Phase 1-4 component. Each route is a thin wrapper that
builds the runtime ``XxxConfig`` from the request body, calls the
matching ``generate_*`` / ``analyze_*`` entry point, and returns the
runtime output dict (already conforming to ``data/schemas/*.schema.json``).

Runtime exceptions (``XxxError`` subclasses of ``ValueError``) are mapped
to HTTP 422 with the ``attempts`` diagnostic attached when present.
``RegistryNotReady`` from :mod:`backend.api.deps` is mapped to HTTP 503.
"""
