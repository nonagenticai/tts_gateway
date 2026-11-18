"""Liveness / readiness probes (auth-exempt)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.session import session_factory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# A readiness probe answers a QUESTION ("may I receive traffic?"), so a
# dependency that is down is a 503 -- never a 500. `readyz` used to run its
# `SELECT 1` completely unguarded, so a dead Postgres surfaced as an unhandled
# exception: a 500 (or an ASGI-level error with no body at all). kubelet does
# treat any non-2xx as not-ready, but the 500 is indistinguishable from a
# genuine server fault, so the pod's own logs and any alerting read a
# dependency outage as an application bug. This is also the exact path the
# gitops readinessProbe targets, so it is the one handler that must not lie.
#
# SQLAlchemyError covers OperationalError / InterfaceError / DBAPI wrappers;
# OSError covers the network layer (ConnectionRefusedError, timeouts).
_DB_PROBE_ERRORS = (SQLAlchemyError, OSError)


# Two paths, one handler → give each an explicit operation_id so the shared
# factory's canonical ``<tag>_<name>`` fallback does not collide on ``health_health``.
@router.get("/health", operation_id="health_health")
@router.get("/livez", operation_id="health_livez")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "tts-gateway"}


@router.get("/readyz", operation_id="health_readyz")
async def readyz() -> dict[str, str]:
    """Return 200 once Postgres answers, 503 while it does not."""
    try:
        async with session_factory()() as session:
            await session.execute(text("SELECT 1"))
    except _DB_PROBE_ERRORS as exc:
        logger.warning("readiness probe failed: database unreachable: %s", exc)
        raise HTTPException(status_code=503, detail="database unreachable") from exc
    return {"status": "ready"}
