"""FastAPI app: submit/poll + worker claim/heartbeat/result, on the claim-queue.

Built on the shared ``w2v_common.asgi.create_app`` factory: it wires the canonical
middleware stack (RequestId, SecurityHeaders, AuthMiddleware, error-envelope
handlers) and the license-token OpenAPI customization. Auth:
``X-Tenant-Id: Bearer <license_token>`` (``/api`` protected; health exempt). Both
the orchestrator (submit/poll) and the remote kokoro workers (claim/…)
authenticate with a license token.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from w2v_common.asgi import create_app

from src.api.body_limit import MULTIPART_FRAMING_HEADROOM_BYTES, BodySizeLimitMiddleware
from src.api.routes import batches, health, workers
from src.config import get_settings
from src.services.storage import get_storage

# Whole-REQUEST ceiling enforced by BodySizeLimitMiddleware, ahead of Starlette's
# multipart parser (which spools each file part to an unbounded temp file before the
# endpoint runs). Derived from the SAME setting as the upload route's per-FILE cap,
# `max_result_bytes`, plus multipart framing headroom — so the transport layer can
# never refuse a clip the route itself would have accepted, and raising
# MAX_RESULT_BYTES raises both bounds together.
_MAX_REQUEST_BYTES = get_settings().max_result_bytes + MULTIPART_FRAMING_HEADROOM_BYTES


async def _startup(app: FastAPI) -> None:
    # Non-fatal: the queue/claim/submit surface works without MinIO; only clip
    # upload + poll-presign need it. A storage blip (or an unprovisioned bucket
    # on a dark deploy) must not crashloop the pod — log and carry on; the bucket
    # is re-ensured on first use. (on_startup hooks are NOT swallowed by the
    # factory, so the guard has to live here.)
    try:
        await get_storage().initialize()
    except Exception:
        logging.getLogger(__name__).exception(
            "storage init failed; continuing (queue still serves)"
        )


async def _shutdown(app: FastAPI) -> None:
    await get_storage().close()


app = create_app(
    get_settings(),
    routers=[health.router, batches.router, workers.router],
    title="tts-gateway",
    version="0.1.0",
    on_startup=[_startup],
    on_shutdown=[_shutdown],
    extra_middleware=[(BodySizeLimitMiddleware, {"max_body_bytes": _MAX_REQUEST_BYTES})],
)


def main() -> None:
    # Bind all interfaces: required inside the container (k8s routes to the pod IP).
    uvicorn.run("src.asgi:app", host="0.0.0.0", port=get_settings().api_port)  # noqa: S104  # nosec B104


if __name__ == "__main__":
    main()
