"""Pure-ASGI request-body size guard.

This is the layer that actually prevents resource exhaustion on the clip-upload
endpoint. The route's ``await audio.read(max_bytes + 1)`` cap runs *after*
Starlette has already parsed the request: ``starlette/formparsers.py`` spools
every FILE part into a ``SpooledTemporaryFile`` with no size bound
(``max_part_size`` covers non-file parts only), so the whole body is
materialised — rolling to DISK past 1 MiB — before the endpoint function is
entered. The in-handler cap therefore only stopped the storage write and the
whole-body ``bytes`` allocation; it converted an API-replica OOM into node disk
pressure rather than refusing the upload.

Running as raw ASGI middleware, this guard sits above the router and above the
form parser, so an oversized request is refused before a single byte is spooled.

It is deliberately NOT a ``BaseHTTPMiddleware`` subclass: that class builds a
``Request`` and pumps the body through an anyio stream, which reintroduces the
buffering we are trying to avoid.

Two independent checks, because either one alone is bypassable:

1. ``Content-Length`` larger than the limit is refused up front, without
   invoking the downstream app at all — the cheapest possible rejection.
2. The ``receive`` callable is wrapped to count the bytes that actually arrive,
   so a request with NO ``Content-Length`` (``Transfer-Encoding: chunked``) or a
   lying one is aborted mid-stream the moment the limit is passed.

The 413 body is the fleet error envelope (``w2v_common.asgi.make_error_payload``)
so a client can branch on the same shape it gets from every routed error.
"""

from __future__ import annotations

import json
import uuid

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from w2v_common.asgi import make_error_payload

# Headroom added on top of the per-FILE cap (``Settings.max_result_bytes``) when
# deriving the whole-REQUEST limit. The route's cap bounds the decoded audio
# part; the request that carries it also contains multipart boundaries, per-part
# Content-Disposition/Content-Type headers, and the sibling ``duration_seconds``
# / ``worker_id`` form fields. Without headroom a clip of exactly the cap size
# would be refused at the transport layer while the route considers it legal.
# 1 MiB is enormous relative to the actual framing (a few hundred bytes) and
# negligible relative to the 100 MiB cap, so it makes the guard strictly
# permissive at the boundary without meaningfully widening the attack surface.
MULTIPART_FRAMING_HEADROOM_BYTES = 1024 * 1024


class _BodyTooLarge(BaseException):
    """Internal signal raised from the wrapped ``receive`` — never leaves this module.

    Deliberately a ``BaseException``, for the same reason ``asyncio.CancelledError``
    is one: this is raised from INSIDE the downstream app's body-read loop, and
    that loop is wrapped in blanket ``except Exception`` handlers we do not
    control. ``fastapi/routing.py`` turns any ``Exception`` escaping body parsing
    into ``HTTPException(400, "There was an error parsing the body")``, and
    ``starlette/formparsers.py`` has its own ``except (MultiPartException, OSError)``.
    Subclassing ``Exception`` made a real chunked oversize upload answer **400**
    instead of 413 — the signal never reached the ``except`` below, so the guard
    silently degraded into a confusing parse error. Verified by test:
    ``test_absent_content_length_is_refused_by_the_streaming_counter``.

    Nothing else may catch this, and it never escapes this module — ``__call__``
    always handles it.
    """


def _declared_length(raw: str | None) -> int | None:
    """Parse a ``Content-Length`` header, treating a malformed value as absent.

    A garbage header is not trusted as a pass — it just falls through to the
    streaming counter, which enforces the limit regardless.
    """
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class BodySizeLimitMiddleware:
    """Refuse HTTP requests whose body exceeds ``max_body_bytes`` with a 413."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan and websocket scopes carry no HTTP body; pass them straight through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(Headers(scope=scope).get("content-length"))
        if declared is not None and declared > self.max_body_bytes:
            # Cheapest path: the client told us it is too big. The downstream app
            # — router, form parser, endpoint — is never invoked.
            await self._reject(scope, send)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if received > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        response_started = False

        async def watching_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, watching_send)
        except _BodyTooLarge:
            # Raised from inside the form parser's read loop. We sit ABOVE
            # Starlette's ExceptionMiddleware, so nothing downstream has turned
            # this into a 500. If the app already began a response we cannot
            # replace it — the abort simply drops the connection.
            if not response_started:
                await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = Headers(scope=scope).get("x-request-id") or str(uuid.uuid7())
        payload = make_error_payload(
            code="HTTP_413",
            message="request body exceeds size limit",
            status=413,
            request_id=request_id,
        )
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
