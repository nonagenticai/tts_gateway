"""Transport-layer body-size guard (``src/api/body_limit.py``).

This is the gate that actually prevents resource exhaustion on the clip-upload
endpoint. The route's ``await audio.read(max + 1)`` cap runs only after Starlette's
multipart parser has spooled the whole file part into an unbounded
``SpooledTemporaryFile`` (rolling to DISK past 1 MiB), so on its own it converted an
API-replica OOM into node disk pressure. ``BodySizeLimitMiddleware`` runs as raw
ASGI above the router, so an oversized request is refused before the parser sees a
byte of it.

**Every test here fails if the middleware is removed** — that is the point:

* the two rejection tests get a ``200`` from the pass-through app instead of a
  ``413``, and observe the endpoint actually running on an oversized body;
* the mid-stream test never stops pulling from an infinite body;
* the pass-through and wiring tests fail at import / assertion time, because the
  module under test and the ``src.asgi`` registration would both be gone.

The app under test is a minimal stand-in, NOT the real upload route: the real route
has its own ``413`` for oversized clips, which would mask a missing middleware and
make these tests pass on revert.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send
from w2v_common.asgi import register_middleware

from src.api.body_limit import MULTIPART_FRAMING_HEADROOM_BYTES, BodySizeLimitMiddleware

# Tiny limit so a test payload can exceed it without allocating anything real.
LIMIT = 2048
CHUNK = 512


@pytest.fixture
def calls() -> list[int]:
    """Byte counts the endpoint actually saw — empty means it never ran."""
    return []


@pytest.fixture
def client(calls: list[int]) -> Iterator[TestClient]:
    app = FastAPI()

    @app.post("/upload")
    async def _upload(audio: UploadFile = File(...)) -> dict[str, int]:
        data = await audio.read()
        calls.append(len(data))
        return {"received": len(data)}

    register_middleware(app, BodySizeLimitMiddleware, max_body_bytes=LIMIT)
    with TestClient(app) as test_client:
        yield test_client


# ── Declared oversize: refused before the app is invoked ─────────────────────


def test_oversized_content_length_is_refused_before_the_route_runs(
    client: TestClient, calls: list[int]
) -> None:
    """A request whose ``Content-Length`` exceeds the limit gets 413, and the
    downstream app — router, multipart parser, endpoint — is never entered.

    Without the middleware the endpoint runs and answers 200, so ``calls`` is
    non-empty: the assertions below are exactly what regresses on revert.
    """
    response = client.post(
        "/upload", files={"audio": ("clip.wav", b"x" * (LIMIT * 3), "audio/wav")}
    )

    assert response.status_code == 413
    assert calls == [], "the endpoint must never run for a declared-oversize body"

    error = response.json()["error"]
    assert error["code"] == "HTTP_413"
    assert error["status"] == 413
    assert error["request_id"]


# ── Undeclared / chunked oversize: refused mid-stream ────────────────────────


def test_absent_content_length_is_refused_by_the_streaming_counter(
    client: TestClient, calls: list[int]
) -> None:
    """A chunked upload carries NO ``Content-Length``, so the header pre-check
    cannot fire — the wrapped ``receive`` counter is what must catch it.

    The body has to be a genuine ``multipart/form-data`` stream, not an opaque
    octet-stream: the counter only sees bytes that the downstream app actually
    pulls, and FastAPI answers 422 on a content-type the route cannot bind
    WITHOUT draining the request. That 422 is a rejection too, but it is not
    this middleware's, so asserting on it would prove nothing about the counter.
    """
    boundary = "----w2vBodyLimitTest"

    def _chunks() -> Iterator[bytes]:
        yield (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="clip.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        for _ in range(8):
            yield b"z" * LIMIT
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post(
        "/upload",
        content=_chunks(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.request.headers.get("content-length") is None, (
        "this test is only meaningful while the request is genuinely chunked"
    )
    assert response.status_code == 413
    assert calls == []


def test_streamed_body_is_aborted_without_being_fully_consumed() -> None:
    """Drive the middleware as raw ASGI over an INFINITE body.

    The body never ends and never declares a length. If the counter did not abort,
    this test would hang or OOM rather than fail — so the assertion that it stops
    after the first chunk past the limit is the real proof that a lying or absent
    ``Content-Length`` cannot be used to stream unbounded data into the parser.
    """
    consumed: list[int] = []
    emitted: list[Message] = []
    pulled = 0

    async def _inner(scope: Scope, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            consumed.append(len(message["body"]))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def _receive() -> Message:
        nonlocal pulled
        pulled += 1
        return {"type": "http.request", "body": b"q" * CHUNK, "more_body": True}

    async def _send(message: Message) -> None:
        emitted.append(message)

    middleware = BodySizeLimitMiddleware(_inner, max_body_bytes=LIMIT)
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload",
        "headers": [],
    }

    asyncio.run(middleware(scope, _receive, _send))

    assert emitted[0]["type"] == "http.response.start"
    assert emitted[0]["status"] == 413
    # Exactly one chunk past the limit, then stop — nothing further is pulled.
    assert pulled == LIMIT // CHUNK + 1
    assert sum(consumed) == LIMIT
    # The inner app never got to answer 204.
    assert [m for m in emitted if m.get("status") == 204] == []


# ── Pass-through ─────────────────────────────────────────────────────────────


def test_normal_upload_passes_through_untouched(client: TestClient, calls: list[int]) -> None:
    """A legal upload must reach the endpoint with its bytes intact.

    Guards the other direction: a body-size guard that breaks the happy path is
    worse than none. On revert this module cannot even import.
    """
    payload = b"riff-real-audio" * 8
    response = client.post("/upload", files={"audio": ("clip.wav", payload, "audio/wav")})

    assert response.status_code == 200
    assert response.json() == {"received": len(payload)}
    assert calls == [len(payload)]


def test_non_http_scopes_pass_through() -> None:
    """Lifespan/websocket scopes carry no HTTP body and must not be touched."""
    seen: list[str] = []
    emitted: list[Message] = []

    async def _inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(str(scope["type"]))
        await send({"type": "lifespan.startup.complete"})

    middleware = BodySizeLimitMiddleware(_inner, max_body_bytes=LIMIT)

    async def _receive() -> Message:
        return {"type": "lifespan.startup"}

    async def _send(message: Message) -> None:
        emitted.append(message)

    asyncio.run(middleware({"type": "lifespan"}, _receive, _send))

    assert seen == ["lifespan"]
    assert emitted == [{"type": "lifespan.startup.complete"}]


# ── Wiring ───────────────────────────────────────────────────────────────────


def test_app_wires_the_guard_from_max_result_bytes() -> None:
    """The real app must register the guard, sized from the SAME setting as the
    route's per-file cap plus multipart framing headroom.

    Pinning the derivation (not a literal) is what keeps the two layers from
    drifting: raising ``MAX_RESULT_BYTES`` must raise the request ceiling too, or
    the transport layer would start refusing clips the route considers legal.
    """
    from src.asgi import app
    from src.config import get_settings

    registered = [m for m in app.user_middleware if m.cls is BodySizeLimitMiddleware]

    assert registered, "BodySizeLimitMiddleware is not registered on the app"
    assert registered[0].kwargs["max_body_bytes"] == (
        get_settings().max_result_bytes + MULTIPART_FRAMING_HEADROOM_BYTES
    )
