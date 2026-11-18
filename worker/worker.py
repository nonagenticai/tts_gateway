#!/usr/bin/env python3
"""tts_gateway pull-worker — a thin bridge for a Mac kokoro box.

Claims jobs from the gateway, synthesizes via the box's ALREADY-RUNNING kokoro
(``tts.test:9202`` / local ``:9202``), and uploads the result. Deliberately
does NOT load a second kokoro instance — it reuses the live one, so a 16 GB mini
runs one model, not two. Adding a box = registering one more worker = +1
parallelism (bounded by that box's single synth lock).

Env:
  TTS_GATEWAY_URL   gateway base. MUST be the in-cluster Service URL
                    (http://tts-gateway.w2v:6320, the default):
                    the P7 service cell token is minted ONLY for the gateway's
                    in-cell endpoint, so an alias (tts-gateway.test, a
                    NodePort) gets no token and 401s once the gateway enforces.
  TTS_SYNTH_URL     local kokoro base (default http://tts.test:9202)
  TTS_GATEWAY_TOKEN OPTIONAL licence token, sent as ``Authorization: Bearer``
                    AGAINST THE GATEWAY ONLY (the legacy lane; P7 dual mode).
                    Unset = cell-only: the service cell token is the worker's
                    sole gateway credential. At least one of the two must be
                    configured, or the worker refuses to start.
  CELL_*            P7 cell signing config (CELL_SERVICE_NAME = this worker's
                    identity, CELL_DEPLOYMENT_ID, CELL_JWT_SIGNING_KEY,
                    CELL_JWT_KID). Every gateway call carries a fresh
                    ``scope=service`` X-W2V-Cell-Token; unset key = no token.
  TTS_SYNTH_TOKEN   Synth-scoped license token for TTS_SYNTH_URL. No silent
                    fallback: absent, the worker refuses to start unless
                    TTS_SYNTH_ALLOW_SHARED_TOKEN says otherwise.
  TTS_SYNTH_ALLOW_SHARED_TOKEN
                    Opt-in ("1"/"true"/"yes"/"on"). Reuse TTS_GATEWAY_TOKEN for
                    synth when no scoped token exists, accepting the exposure
                    knowingly. Warns loudly at every startup.
  TTS_WORKER_ID     worker id (default: hostname)
  SYNTH_EFFECTIVE_TENANT_ENABLED
                    Opt-in ("1"/"true"/"yes"/"on"), DEFAULT OFF. Stamp the claimed
                    job's tenant as X-Effective-Tenant-Id on the synth hop so the
                    cost event lands on the customer. Off by default because on the
                    in-cluster hop the receiver ACCEPTS the delegation and binds
                    license_id=None — an unprovisioned tenant then 4xxes and the job
                    poisons. See the block above SYNTH_EFFECTIVE_TENANT_ENABLED.
"""

from __future__ import annotations

import atexit
import base64
import email.utils
import math
import os
import signal
import socket
import sys
import threading
import time
import uuid
from types import FrameType

import httpx
from w2v_common.auth.cell_jwt import cell_audience_for_url, cell_signer_configured
from w2v_common.auth.identity_context import service_cell_token_hook_sync
from w2v_common.auth.outbound import EFFECTIVE_TENANT_HEADER

GATEWAY = os.environ.get("TTS_GATEWAY_URL", "http://tts-gateway.w2v:6320")
SYNTH = os.environ.get("TTS_SYNTH_URL", "http://tts.test:9202")
#: The gateway's cell identity (its k8s Service name) — the ``aud`` of every token.
GATEWAY_CELL_AUDIENCE = "tts-gateway"

# The gateway credential: a service cell token, the legacy licence, or both (dual).
#
# The licence is the fleet-wide DOWNSTREAM_LICENSE_TOKEN (see the synth block below),
# so it is OPTIONAL once this worker can sign: the gateway admits a worker
# ``scope=service`` token on its own (``WORKER_CELL_CALLERS``), in dual AND enforce,
# and under enforce the licence lane is refused for this delegating subject anyway.
# Blank counts as unset — an empty Bearer is not a credential.
#
# ``cell_signer_configured()`` RAISES on a half-configured signer (key without
# kid/name/deployment), which is the right boot failure for that too.
TOKEN: str | None = (os.environ.get("TTS_GATEWAY_TOKEN") or "").strip() or None
CELL_SIGNER_CONFIGURED = cell_signer_configured()

# ⛔ FAIL LOUD. With neither, every gateway call is anonymous: the claim loop would
# swallow 401s forever (it keeps itself alive through transients) and the TTS queue
# would stall with a Running, Ready pod and nothing in the logs but noise.
if TOKEN is None and not CELL_SIGNER_CONFIGURED:
    raise RuntimeError(
        "tts worker has no gateway credential: TTS_GATEWAY_TOKEN is unset and cell "
        "signing is not configured (CELL_JWT_SIGNING_KEY unset). Configure the P7 cell "
        "signer (CELL_JWT_SIGNING_KEY, CELL_JWT_KID, CELL_SERVICE_NAME, "
        "CELL_DEPLOYMENT_ID) or, for the legacy lane, TTS_GATEWAY_TOKEN. Then restart."
    )

# Cell-only has no fallback lane, and the hook mints ONLY for the gateway's in-cell
# endpoint — an alias (tts-gateway.test, a NodePort) would go out unsigned and
# 401 on every call. Refuse that at boot instead of in the claim loop.
if TOKEN is None and cell_audience_for_url(GATEWAY) != GATEWAY_CELL_AUDIENCE:
    raise RuntimeError(
        f"TTS_GATEWAY_URL ({GATEWAY}) is not the gateway's in-cell endpoint, so no cell "
        "token would be minted for it, and TTS_GATEWAY_TOKEN is unset — every gateway "
        "call would be unauthenticated. Point TTS_GATEWAY_URL at the in-cluster Service "
        "URL (http://tts-gateway.w2v:6320) or set TTS_GATEWAY_TOKEN."
    )


# ⚠️ NAMED TUPLES, not inline `except A, B:` — and not `except (A, B):` either.
#
# The bare multi-exception form is PEP 758 and PARSES ONLY ON PYTHON >= 3.14. This
# module is the container entrypoint for both worker Deployments, so one base-image
# downgrade (or anyone running it under the system python of an older box) turns it
# into a startup SyntaxError that kills the whole worker fleet at once — before a
# single line of it executes, so no test, probe or log would precede it.
#
# Parenthesising is the obvious fix and it does NOT survive. MEASURED, not assumed
# (2026-09-15), on THIS file with THIS repo's pyproject: `ruff format` rewrites
# `except (A, B):` straight back to the bare 3.14-only form. Reproduced on ruff
# 0.15.0, 0.15.20 (the version in this repo's .venv), 0.16.0 and 0.16.5 — i.e. the
# whole range the `ruff>=0.15.0` dev floor can resolve to.
#
# ⚠️ THE REWRITE IS GATED ON `target-version`, AND THAT IS WHY IT LOOKS ABSENT.
# Under `target-version = "py314"` (this repo, [tool.ruff] in pyproject.toml, and
# also what ruff infers from `requires-python = ">=3.14"`) ruff rewrites. Under
# py313, or in a scratch directory where no config is picked up at all, the same
# ruff reports "1 file left unchanged". So a check run OUTSIDE this repo's config
# measures the wrong thing and concludes, wrongly, that parentheses are safe here.
# An audit reached exactly that conclusion; re-running it inside the real config
# refuted it. If you are about to "simplify" this back to parentheses, reproduce it
# in a directory that actually loads this pyproject before you believe yourself.
#
# Binding the tuple to a NAME leaves the formatter nothing to unwrap, and says what
# the group means.
_UUID_PARSE_ERRORS = (ValueError, AttributeError, TypeError)
_NUMERIC_COERCION_ERRORS = (TypeError, ValueError)
_HTTP_DATE_PARSE_ERRORS = (TypeError, ValueError, IndexError)


def _env_flag(name: str) -> bool:
    """An env flag is OFF unless it says yes, affirmatively.

    A bare truthiness test would read "false"/"0"/"no" as consent, which is how an
    operator DISABLING something silently re-enables it.
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# The synth endpoint gets its OWN credential, and there is NO fallback to TOKEN.
#
# ⚠️ WHAT THE SHARED CREDENTIAL WAS (verified against GitOps 2026-08-03):
# `clusters/strato/services/w2v/deployments.yaml` wires TTS_GATEWAY_TOKEN from
# `tts-gateway-secrets` key **DOWNSTREAM_LICENSE_TOKEN** — the FLEET-WIDE S2S
# license token every w2v service uses for every cross-service call, not a
# gateway-only service account. SYNTH is operator-configurable and the box behind
# it is not the gateway's, so `SYNTH_TOKEN = _env or TOKEN` handed a third party
# the credential the whole fleet authenticates with — and the one that can claim,
# complete and fail TTS jobs for EVERY tenant.
#
# That fallback was the live configuration: TTS_SYNTH_TOKEN was unset on both
# workers, so every synth call shipped the fleet token. The old code logged a
# WARNING and then made the call anyway. A warning is not a control — it does not
# stop the disclosure, and nobody reads worker stdout.
#
# The defect is the SILENCE, not the sharing as such. Sharing is sometimes the only
# thing that works: verified live 2026-08-03, the mini2 worker's synth target
# (in-cluster modality-gateway) enforces BYO posture on the CALLER's tenant and
# answers a synth-scoped `svc-tts-synth` token with HTTP 402 "tenant must configure
# its own tts provider before use", while the gateway token gets a 200. The mini1
# worker (off-cluster kokoro at tts-mini1.test:9202) takes either. So an
# unconditional hard-fail would CrashLoop a worker that has a real reason to share.
#
# Hence three states, with the DEFAULT still fail-closed:
#   TTS_SYNTH_TOKEN set                      → use it, never fall back
#   absent + TTS_SYNTH_ALLOW_SHARED_TOKEN    → share the gateway token, WARN loudly
#   absent + no opt-in                       → refuse to start
# Silent reuse — the actual defect — is unreachable in all three.
#
# NOT gated on the synth HOSTNAME. Neither deployed worker points at an in-cluster
# default (they are `modality-gateway…:8890` and `tts-mini1.test:9202`), so a
# "must be the default host" guard would break both.
_SYNTH_TOKEN_ENV = os.environ.get("TTS_SYNTH_TOKEN")
SYNTH_TOKEN_SHARING_IS_OPTED_IN = _env_flag("TTS_SYNTH_ALLOW_SHARED_TOKEN")

if _SYNTH_TOKEN_ENV:
    SYNTH_TOKEN = _SYNTH_TOKEN_ENV
elif SYNTH_TOKEN_SHARING_IS_OPTED_IN:
    if TOKEN is None:
        raise RuntimeError(
            "TTS_SYNTH_ALLOW_SHARED_TOKEN is set but TTS_GATEWAY_TOKEN is unset, so there "
            "is no token to share with the synth host. Either wire TTS_SYNTH_TOKEN, or "
            "keep TTS_GATEWAY_TOKEN on this worker until the synth hop has its own "
            "credential. Then restart."
        )
    SYNTH_TOKEN = TOKEN
else:
    raise RuntimeError(
        "TTS_SYNTH_TOKEN is unset (or empty) and TTS_SYNTH_ALLOW_SHARED_TOKEN is not set. "
        f"TTS_SYNTH_TOKEN is the synth-scoped license token for calls to TTS_SYNTH_URL "
        f"({SYNTH}). This worker will NOT silently fall back to TTS_GATEWAY_TOKEN: that "
        "value is the fleet-wide DOWNSTREAM_LICENSE_TOKEN, and TTS_SYNTH_URL is an "
        "operator-configurable host the gateway does not own, so the old fallback was a "
        "fleet-wide credential disclosure nothing detected. Either wire TTS_SYNTH_TOKEN "
        "from a SEPARATE, synth-scoped license (GitOps: tts-gateway-runtime-secret), or — "
        "if the synth target rejects a scoped tenant, as in-cluster modality-gateway does "
        "with HTTP 402 — set TTS_SYNTH_ALLOW_SHARED_TOKEN=true to accept the exposure "
        "knowingly. Then restart."
    )

# Set, but byte-identical to the gateway token, is the same exposure as the old
# fallback — the configuration changed and nothing else did. It cannot be a hard
# failure (only the operator can tell whether a token is genuinely synth-scoped, and
# crashing on it would be a second way to take a worker down), but it must not read
# as remediated either. Compared by VALUE: the two arrive under different env names
# and only the value decides what reaches the synth host.
SYNTH_TOKEN_IS_EXPLICIT_DUPLICATE = (
    bool(_SYNTH_TOKEN_ENV) and TOKEN is not None and _SYNTH_TOKEN_ENV == TOKEN
)
# The union: every configuration in which the synth host receives the fleet credential.
SYNTH_TOKEN_IS_SHARED = SYNTH_TOKEN_IS_EXPLICIT_DUPLICATE or not _SYNTH_TOKEN_ENV
WORKER_ID = os.environ.get("TTS_WORKER_ID", socket.gethostname())
POLL_IDLE_SECONDS = 3.0


def _env_float(name: str, default: float) -> float:
    """A positive float from env, or ``default`` when unset/blank/unparseable/<=0."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


# The gateway's lease. The AUTHORITATIVE value arrives on every claim
# (``ClaimedJob.lease_seconds``) and ``_heartbeat_interval`` derives the cadence from
# it per job. This env value is only the FALLBACK for a claim from an older gateway
# that does not send it (QUEUE_LEASE_SECONDS default, 180). Either way the worker
# beats at least every lease/3, so two missed beats still leave the claim alive,
# however long a GPU-busy wait sleeps.
LEASE_SECONDS = _env_float("TTS_LEASE_SECONDS", 180.0)
HEARTBEAT_SECONDS = min(60.0, LEASE_SECONDS / 3.0)
SYNTH_TIMEOUT = 300.0

# GPU admission control (tts_gateway#45). Upstream GPU services answer 503 +
# ``Retry-After`` + ``{"error":"gpu_busy","lane","retry_after"}`` when their lane is
# full; a gateway in between may answer 429. Both are a BUSY signal, not a failure:
# the job is retried IN PLACE (the claim is held and heartbeated meanwhile) until a
# per-job deadline or retry count runs out, and then the claim is RELEASED — which
# refunds the attempt — never /fail'ed, which would burn one of max_attempts lives
# and, four busy spells later, poison a perfectly good chunk.
BUSY_STATUSES = frozenset({429, 503})
BUSY_RETRY_AFTER_DEFAULT = 5.0
BUSY_RETRY_AFTER_MIN = 1.0
BUSY_RETRY_AFTER_MAX = 120.0
BUSY_DEADLINE_SECONDS = _env_float("TTS_BUSY_DEADLINE_SECONDS", 900.0)
BUSY_MAX_RETRIES = int(_env_float("TTS_BUSY_MAX_RETRIES", 60.0))

RELEASE_TIMEOUT = 5.0

# Gateway-bound headers: claim / heartbeat / result / fail / release.
#
# ``Authorization: Bearer`` — the platform convention. ``X-Tenant-Id: Bearer`` is
# deprecated in w2v_common (counted as ``legacy-x-tenant-id``). Same licence; this is
# the legacy lane that still decides while the gateway runs CELL_AUTH_MODE=dual.
# EMPTY in cell-only mode: no Authorization header at all, never ``Bearer None``.
_GW_H: dict[str, str] = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def _gateway_client(
    *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    """An ``httpx.Client`` for GATEWAY calls only: every send gets a service cell token.

    ``scope=service`` (no tenant): the worker routes are a cross-tenant surface
    authorized by worker IDENTITY, and a claim poll has no tenant to bind. The hook
    is pinned to ``tts-gateway``, so it mints only when the request host:port is the
    gateway's in-cell endpoint and strips the header everywhere else. Synth calls
    use a SEPARATE client (``_synth_client``) and never see a cell token.
    """
    return httpx.Client(
        event_hooks={"request": [service_cell_token_hook_sync(audience=GATEWAY_CELL_AUDIENCE)]},
        timeout=timeout,
        transport=transport,
    )


def _synth_client(
    *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    """Plain client for the synth host — no gateway credential, no cell token."""
    return httpx.Client(timeout=timeout, transport=transport)


# Synth-bound headers: the minimum the :9202 contract needs, and nothing gateway-ish.
#
# ⚠️ MODULE-LEVEL CONSTANT, shared by every request. Never mutate it — per-request
# attribution is added by _synth_headers() onto a COPY.
_SYNTH_H = {"X-Tenant-Id": f"Bearer {SYNTH_TOKEN}", "Accept": "application/json"}

# ⛔ SHIPS DARK. Default FALSE — the stamp below is built, tested and deployed, but
# does nothing until an operator turns it on PER DEPLOYMENT, after checking what the
# receiver on THAT hop does with an unlicensed tenant.
#
# WHY IT CANNOT DEFAULT ON. There are two Deployments on this image, and the hop is
# not equally inert on both:
#
#   tts-gateway-worker-mini1 → tts-mini1.test:9202, Bearer TTS_SYNTH_TOKEN
#       (subject svc-tts-synth). NOT in that receiver's AUTH_DELEGATING_SUBJECTS,
#       so the assertion is refused and the hop behaves exactly as today.
#   tts-gateway-worker       → modality-gateway.modality-gateway.svc:8890, Bearer
#       DOWNSTREAM_LICENSE_TOKEN (via TTS_SYNTH_ALLOW_SHARED_TOKEN=true). Its subject
#       8382f0b3-c082-4f97-8d47-120ce5a18e05 IS listed in that service's
#       AUTH_DELEGATING_SUBJECTS (clusters/strato/services/modality-gateway/
#       deployment.yaml). The guard PASSES there.
#
# On the passing hop w2v_common takes the accepting branch: tenant_delegated=True and
# **license_id is bound to None** (the library says, in as many words, no `or
# license_id` there). A narration job for a customer tenant that has no
# modality-gateway provider / BYO row is served TODAY under the service-account
# tenant; stamped, it arrives as an unlicensed, never-provisioned tenant → 4xx →
# `r.raise_for_status()` below → the job fails, retries, and hits the poison cap.
# That is the narration-outage shape this fleet already had once.
#
# So: enable on ONE deployment at a time, only after verifying the receiver's
# behaviour for a tenant with no provider row. Do not flip this because "the code
# looks right" — the code IS right; the receiving side is what is unknown.
SYNTH_EFFECTIVE_TENANT_ENABLED = _env_flag("SYNTH_EFFECTIVE_TENANT_ENABLED")


def _synth_headers(tenant_id: str | None) -> dict[str, str]:
    """Synth headers for ONE job, attributing the hop to the job's owning tenant.

    THE BUG THIS CLOSES. The Bearer credential on this hop authenticates the
    WORKER, not the customer: it introspects to a service-account tenant. So
    tts_backend scoped the request — and the cost event it emits at the end of
    synthesis (``tts_service.py::_emit_cost_event``) — to the service account, and
    a customer's narration was billed to the fleet SA tenant. The gateway emits no
    cost event of its own, so nothing anywhere corrected it.

    ``X-Effective-Tenant-Id`` is w2v_common's existing side-channel for exactly
    this: ``AuthMiddleware._apply_effective_tenant`` re-scopes
    ``request.state.tenant_id`` to the asserted tenant. The header name comes from
    ``w2v_common.auth.outbound.EFFECTIVE_TENANT_HEADER`` rather than a literal, so
    a rename in the library cannot silently un-attribute this hop.

    ``build_downstream_tenant_headers()`` is deliberately NOT reused here. It mints
    the Bearer from ``DOWNSTREAM_LICENSE_TOKEN`` — the fleet-wide credential this
    worker exists to keep AWAY from an operator-configurable synth host (see the
    TTS_SYNTH_TOKEN block above) — and it takes a ``TenantSource`` this worker
    cannot honestly supply: the tenant arrives as a field of a claim payload, not
    from an introspected licence or a verified JWT on an inbound request. Calling
    it would either re-open the credential disclosure or raise
    ``UnvouchedTenantError`` on every job.

    ⛔ FAIL SAFE. An unknown, blank or non-UUID tenant sends the request EXACTLY as
    it is sent today — no header at all, not an empty one. This path had a full
    narration outage this week; a mis-billed row is recoverable, a dead narration
    queue is not. A blank header would also be worse than none: w2v_common treats
    a present-but-empty value as no assertion, but any downstream that validated it
    strictly would start refusing perfectly good jobs.
    """
    # ⛔ OFF BY DEFAULT — see SYNTH_EFFECTIVE_TENANT_ENABLED above. Read through the
    # module global (not captured at def time) so a test can flip it without a reload.
    if not SYNTH_EFFECTIVE_TENANT_ENABLED:
        return dict(_SYNTH_H)
    if not tenant_id:
        return dict(_SYNTH_H)
    candidate = str(tenant_id).strip()
    try:
        uuid.UUID(candidate)
    except _UUID_PARSE_ERRORS:
        # See _UUID_PARSE_ERRORS: under this repo's `target-version = "py314"`, the
        # name is what keeps `ruff format` from rewriting
        # this back into the 3.14-only bare form.
        #
        # A delegating subject asserting a NON-UUID effective tenant is rejected
        # outright by w2v_common (401 INVALID_TENANT) — it does not fall back. So
        # sending garbage would not mis-bill the job, it would KILL it. Drop the
        # assertion instead and let the job synthesize unattributed.
        _log(f"ignoring unparseable job tenant {candidate[:16]!r} — synthesizing unattributed")
        return dict(_SYNTH_H)
    return {**_SYNTH_H, EFFECTIVE_TENANT_HEADER: candidate}


# The job this worker currently holds a claim on, if any. Read by the shutdown
# path, which runs in a signal handler — a plain dict mutation is atomic enough
# for that; nothing here takes a lock (a shutdown must never block).
_CURRENT: dict[str, str] = {}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _release_current() -> None:
    """Hand back the in-flight claim so the gateway refunds its attempt.

    THIS IS THE FIX FOR THE POISON BUG. The gateway charges an attempt on every
    claim and fails a job at ``max_attempts``; a k8s RollingUpdate or a Mac reboot
    interrupting a HEALTHY job would otherwise burn one of its lives, and four
    rollouts would poison a chunk that never once misbehaved.

    Runs from a signal handler and from ``atexit``, so it is written to never block
    and never raise: a FRESH ``httpx.Client`` (the main client may be mid-request
    and is not signal-reentrant), a short timeout, NO retries, and every exception
    swallowed. If the release does not land, the gateway self-heals when the lease
    expires — slower, but correct. Popping ``job_id`` makes a second call a no-op.
    """
    job_id = _CURRENT.pop("job_id", None)
    if not job_id:
        return
    try:
        with _gateway_client(timeout=RELEASE_TIMEOUT) as client:
            r = client.post(
                f"{GATEWAY}/api/tts/jobs/{job_id}/release",
                headers=_GW_H,
                data={"worker_id": WORKER_ID},
            )
        _log(f"released job={job_id[:12]} (attempt refunded) → {r.status_code}")
    except Exception as exc:  # noqa: BLE001 -- shutdown must not raise; lease expiry is the backstop
        _log(f"release failed job={job_id[:12]}: {exc} — lease expiry will reclaim it")


def _on_shutdown_signal(signum: int, _frame: FrameType | None) -> None:
    """SIGTERM/SIGINT → release the claim, then exit regardless of the outcome."""
    _log(f"signal {signum} received — releasing in-flight claim, shutting down")
    _release_current()
    sys.exit(0)


def _install_shutdown_hooks() -> None:
    signal.signal(signal.SIGTERM, _on_shutdown_signal)
    signal.signal(signal.SIGINT, _on_shutdown_signal)
    # Covers the paths a signal handler does not: a normal return, an unhandled
    # exception, sys.exit from elsewhere.
    atexit.register(_release_current)


def _measured_duration(raw: object) -> float | None:
    """The backend's ``duration_sec``, or ``None`` when it did not measure one.

    Absent, null, non-numeric, non-finite and non-POSITIVE all mean the same thing:
    nobody measured this clip. A real clip is never zero seconds long — the queue
    only ever holds non-empty text — so a reported ``0.0`` is the backend's
    ``if sr else 0.0`` sentinel, not a length. Negative is nonsense outright.
    """
    # ``bool`` is excluded explicitly: it is an ``int`` subclass, so ``float(True)``
    # would otherwise turn a JSON ``true`` into a 1.0-second "measurement".
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        return None
    try:
        value = float(raw)
    except _NUMERIC_COERCION_ERRORS:
        return None
    # ``isfinite`` covers nan AND both infinities. nan matters: every comparison
    # against it is False, so it would sail straight through the ``<= 0.0`` test and
    # then 422 at the gateway's ``allow_inf_nan=False`` bound — turning a missing
    # measurement back into a failed job, which is the outcome this whole change exists
    # to avoid.
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _synthesize(
    client: httpx.Client,
    text: str,
    voice: str | None,
    speed: float,
    provider: str | None = None,
    tenant_id: str | None = None,
) -> tuple[bytes, float | None]:
    """Call the local kokoro (base64 WAV + duration_sec — the :9202 contract).

    Returns ``None`` for the duration when the backend did NOT measure one. That is
    a live path, not a defensive branch: ``tts_backend`` computes
    ``duration_sec = frames / sr if sr else 0.0``, so real audio legitimately comes
    back with no usable length. Coercing that to ``0.0`` (the old
    ``float(d.get("duration_sec") or 0.0)``) FABRICATES a measurement — and a
    fabricated 0.0 is byte-identical, everywhere downstream, to a genuine
    measurement of zero. The orchestrator builds its narration timeline out of this
    number and allocates no time at all for a chunk full of speech. ``None`` is the
    only value that says "unknown"; the gateway persists it as SQL NULL and the
    batch poll flags it (``JobStatus.duration_unmeasured``).

    The audio itself is untouched: an unmeasured clip is still uploaded and still
    completes the job. Dropping real audio over a missing measurement is strictly
    worse — the narrate stage raises ``StagePreconditionError`` on any failed job
    and kills the whole book.

    ``provider`` comes from the claim (``tts_jobs.provider``, server_default
    ``kokoro``) and is forwarded because the in-cluster synth target keys real
    behaviour off it: modality_gateway skips its ENTIRE BYO resolve when the body
    names a local provider (``kokoro``/``kitten``) and proxies straight to the
    platform GPU backend. Sent as null, the resolve runs, and a synth-SCOPED tenant
    is answered ``ModalityBYORequiredError`` → HTTP 402 — the one reason this worker
    has to be booted with ``TTS_SYNTH_ALLOW_SHARED_TOKEN`` and hand the fleet
    credential to the synth host. Filling the field makes that 402 branch
    unreachable; retiring the opt-in is a separate, later step.

    Behaviourally inert today, deliberately: every row is born ``kokoro``,
    ``TTS_DEFAULT_PROVIDER`` is ``kokoro`` on both deployed backends, and
    tts_backend's ``SynthesizeRequest`` accepts the same literal — so the value now
    stated explicitly is the value that was being defaulted to.

    Omitted rather than sent as ``None`` when absent (same shape as ``voice``): a
    claim from an older gateway has nothing to say, and the backend's own default is
    the right answer then.
    """
    body: dict[str, object] = {"text": text, "speed": speed}
    if voice:
        body["voice"] = voice
    if provider:
        body["provider"] = provider
    r = client.post(
        f"{SYNTH}/api/tts/synthesize",
        json=body,
        headers=_synth_headers(tenant_id),
        timeout=SYNTH_TIMEOUT,
    )
    if r.status_code in BUSY_STATUSES:
        raise SynthBusyError(r.status_code, _retry_after_seconds(r))
    if r.status_code == 402:
        raise SynthConfigError(_error_detail(r))
    r.raise_for_status()
    d = r.json()
    b64 = d.get("audio_base64")
    if not b64:
        raise RuntimeError("kokoro returned no audio_base64")
    return base64.b64decode(b64), _measured_duration(d.get("duration_sec"))


class SynthBusyError(Exception):
    """The synth target is saturated (503/429): retry later, this is not a failure."""

    def __init__(self, status_code: int, retry_after: float) -> None:
        super().__init__(f"synth busy: HTTP {status_code}, retry after {retry_after:g}s")
        self.status_code = status_code
        self.retry_after = retry_after


class SynthConfigError(Exception):
    """HTTP 402 from synth: the TENANT's configuration forbids this call.

    modality_gateway answers 402 when a byo_required tenant has no configured TTS
    provider — including (modality_gateway#50) when it names a local provider
    (kokoro/kitten). No retry can fix that; only the customer configuring a provider
    can. So the job is failed TERMINALLY (no requeue, no retry burn) with a reason a
    human can act on.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(
            "synth refused with HTTP 402 (customer configuration): the tenant must "
            f"configure its own TTS provider before use — {detail}"
        )


def _error_detail(r: httpx.Response) -> str:
    """The upstream's own error text (FastAPI ``detail``), else the raw body, bounded."""
    try:
        body = r.json()
    except Exception:  # noqa: BLE001 -- detail is best-effort
        return r.text[:300] or "(no detail)"
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])[:300]
    return str(body)[:300]


class BusyExhaustedError(Exception):
    """The per-job busy budget ran out — hand the claim back (refund), do not fail it."""

    def __init__(self, last: SynthBusyError, retries: int) -> None:
        super().__init__(f"GPU still busy after {retries} retries: {last}")
        self.last = last
        self.retries = retries


class LeaseLostError(Exception):
    """A heartbeat was refused (409): another worker may own the job now."""


def _clamp_retry_after(value: float) -> float:
    return max(BUSY_RETRY_AFTER_MIN, min(BUSY_RETRY_AFTER_MAX, value))


def _parse_retry_after_header(raw: str) -> float | None:
    """``Retry-After`` is delta-seconds OR an HTTP-date (RFC 9110 §10.2.3)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(raw)
        except _HTTP_DATE_PARSE_ERRORS:
            return None
        if when is None:
            return None
        value = when.timestamp() - time.time()
    return value if math.isfinite(value) else None


def _retry_after_seconds(r: httpx.Response) -> float:
    """How long the synth target asked us to wait: header, then body, then default.

    Always clamped to [1, 120] s — a 0 would hot-loop the GPU's admission gate, and an
    absurd value would park a claim far past any sane deadline.
    """
    header = r.headers.get("retry-after")
    if header is not None:
        parsed = _parse_retry_after_header(header)
        if parsed is not None:
            return _clamp_retry_after(parsed)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001 -- a busy body is optional; any parse error = no hint
        body = None
    if isinstance(body, dict):
        raw = body.get("retry_after")
        if isinstance(raw, int | float) and not isinstance(raw, bool) and math.isfinite(raw):
            return _clamp_retry_after(float(raw))
    return BUSY_RETRY_AFTER_DEFAULT


def _synthesize_until_not_busy(
    synth: httpx.Client, job: dict, lease_lost: threading.Event
) -> tuple[bytes, float | None]:
    """``_synthesize``, retried in place while the synth target says it is busy.

    Bounded twice — ``BUSY_MAX_RETRIES`` and ``BUSY_DEADLINE_SECONDS`` (wall clock from
    the first attempt) — and a wait that would cross the deadline is not started.
    The claim stays alive throughout because ``_process``'s heartbeat thread keeps
    beating; if a beat is refused the wait is cut short and ``LeaseLostError`` raised,
    since the job may already belong to someone else.
    """
    deadline = time.monotonic() + BUSY_DEADLINE_SECONDS
    retries = 0
    while True:
        try:
            return _synthesize(
                synth,
                job["text"],
                job.get("voice"),
                float(job.get("speed") or 1.0),
                job.get("provider"),
                # Absent on a claim from an older gateway: _synth_headers() then sends
                # the request unchanged rather than failing the job.
                job.get("tenant_id"),
            )
        except SynthBusyError as busy:
            if retries >= BUSY_MAX_RETRIES or time.monotonic() + busy.retry_after > deadline:
                raise BusyExhaustedError(busy, retries) from busy
            retries += 1
            _log(
                f"busy job={job['job_id'][:12]} HTTP {busy.status_code} — retry "
                f"{retries}/{BUSY_MAX_RETRIES} in {busy.retry_after:g}s"
            )
            # Event.wait, not time.sleep: a refused heartbeat wakes us immediately.
            if lease_lost.wait(busy.retry_after):
                raise LeaseLostError(f"lease lost while waiting on a busy GPU: {busy}") from busy


def _heartbeat_interval(lease_seconds: object) -> float:
    """Beat at least every lease/3, using the lease the claim reported when present."""
    if isinstance(lease_seconds, int | float) and not isinstance(lease_seconds, bool):
        lease = float(lease_seconds)
        if math.isfinite(lease) and lease > 0:
            return min(HEARTBEAT_SECONDS, lease / 3.0)
    return HEARTBEAT_SECONDS


def _heartbeat(
    client: httpx.Client,
    job_id: str,
    stop: threading.Event,
    lease_lost: threading.Event | None = None,
    interval: float | None = None,
) -> None:
    every = interval if interval is not None else HEARTBEAT_SECONDS
    while not stop.wait(every):
        try:
            # worker_id → the gateway renews only a claim THIS worker holds, and
            # answers 409 when it no longer does.
            r = client.post(
                f"{GATEWAY}/api/tts/jobs/{job_id}/heartbeat",
                headers=_GW_H,
                data={"worker_id": WORKER_ID},
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort keepalive
            _log(f"heartbeat error job={job_id[:12]}: {exc}")
            continue
        if getattr(r, "status_code", None) == 409:
            # The gateway no longer considers the job claimed (by anyone, or by us).
            _log(f"heartbeat refused job={job_id[:12]} — lease lost")
            if lease_lost is not None:
                lease_lost.set()
            return


def _release_job(gateway: httpx.Client, job_id: str) -> None:
    """Hand a still-held claim back so the gateway refunds the attempt."""
    try:
        r = gateway.post(
            f"{GATEWAY}/api/tts/jobs/{job_id}/release",
            headers=_GW_H,
            # reason=gpu_busy: the gateway also clears started_at so busy time never
            # counts toward the job's lifetime bound.
            data={"worker_id": WORKER_ID, "reason": "gpu_busy"},
            timeout=15,
        )
        _log(f"released job={job_id[:12]} (attempt refunded) → {r.status_code}")
    except Exception as exc:  # noqa: BLE001 -- lease expiry is the backstop
        _log(f"release failed job={job_id[:12]}: {exc} — lease expiry will reclaim it")


def _process(gateway: httpx.Client, synth: httpx.Client, job: dict) -> float:
    """Synthesize one claimed job via ``synth`` and report it via ``gateway``.

    Two clients on purpose: ``gateway`` signs every request with the worker's
    service cell token (see ``_gateway_client``); ``synth`` talks to a host the
    gateway does not own and must carry neither that token nor the gateway licence.

    Returns a cool-down (seconds) the claim loop should wait before claiming again:
    non-zero only after a GPU-busy release, so this worker does not immediately
    re-claim the job it just handed back and hammer a saturated lane.
    """
    job_id = job["job_id"]
    # Publish the claim BEFORE any work, so a SIGTERM at any point during synthesis
    # can hand it back instead of letting it burn an attempt.
    _CURRENT["job_id"] = job_id
    stop = threading.Event()
    lease_lost = threading.Event()
    hb = threading.Thread(
        target=_heartbeat,
        args=(gateway, job_id, stop, lease_lost, _heartbeat_interval(job.get("lease_seconds"))),
        daemon=True,
    )
    hb.start()
    cooldown = 0.0
    try:
        wav, dur = _synthesize_until_not_busy(synth, job, lease_lost)
        r = gateway.post(
            f"{GATEWAY}/api/tts/jobs/{job_id}/result",
            headers=_GW_H,
            files={"audio": ("clip.wav", wav, "audio/wav")},
            # worker_id lets the gateway reject this write if our lease already expired
            # and another worker reclaimed the job — we must not finish its claim.
            # OMITTED, not sent as "0.0", when the backend measured nothing: the
            # form field is optional at the gateway precisely so the absence
            # survives the wire instead of being re-fabricated one layer later.
            data=(
                {"duration_seconds": str(dur), "worker_id": WORKER_ID}
                if dur is not None
                else {"worker_id": WORKER_ID}
            ),
            timeout=120,
        )
        r.raise_for_status()
        measured = f"{dur:.1f}s" if dur is not None else "duration UNMEASURED"
        _log(f"done job={job_id[:12]} chars={len(job['text'])} {measured}")
    except BusyExhaustedError as exc:
        # A busy GPU is not this job's fault: refund the attempt, never /fail.
        _log(f"BUSY job={job_id[:12]}: {exc} — releasing the claim")
        _release_job(gateway, job_id)
        cooldown = exc.last.retry_after
    except LeaseLostError as exc:
        # The claim is not ours to fail or release any more; the owner decides.
        _log(f"ABANDON job={job_id[:12]}: {exc}")
    except SynthConfigError as exc:
        _log(f"FAIL (terminal) job={job_id[:12]}: {exc}")
        try:
            gateway.post(
                f"{GATEWAY}/api/tts/jobs/{job_id}/fail",
                headers=_GW_H,
                # terminal: an older gateway ignores the unknown field and requeues,
                # which is today's behaviour — never worse.
                data={"error": str(exc)[:500], "worker_id": WORKER_ID, "terminal": "true"},
                timeout=15,
            )
        except Exception as exc2:  # noqa: BLE001
            _log(f"could not report failure: {exc2}")
    except Exception as exc:  # noqa: BLE001 -- report the failure back to the queue
        _log(f"FAIL job={job_id[:12]}: {exc}")
        try:
            gateway.post(
                f"{GATEWAY}/api/tts/jobs/{job_id}/fail",
                headers=_GW_H,
                # Same guard as /result: a late failure from an expired lease must not
                # requeue (or poison) the claim another worker now holds.
                data={"error": str(exc)[:500], "worker_id": WORKER_ID},
                timeout=15,
            )
        except Exception as exc2:  # noqa: BLE001
            _log(f"could not report failure: {exc2}")
    finally:
        stop.set()
        # The job reached a verdict (done, failed or released) — there is no longer a claim to
        # hand back, so a later shutdown must not "release" it.
        _CURRENT.pop("job_id", None)
    return cooldown


def _warn_if_sharing_gateway_token() -> None:
    """Say out loud, at boot, that the fleet credential is going to the synth host.

    Two ways to reach that state now, and both are deliberate acts by an operator —
    so this is not the control (the control is that the DEFAULT refuses to start),
    it is the record. Silence would make an accepted exposure indistinguishable from
    a separated one:

    * opted in — TTS_SYNTH_ALLOW_SHARED_TOKEN, e.g. the mini2 worker, whose synth
      target 402s a synth-scoped tenant;
    * byte-identical — TTS_SYNTH_TOKEN set to the SAME value as the gateway token,
      the shape of a rushed remediation (`tts-gateway-secrets` holds exactly one
      token, so "just add TTS_SYNTH_TOKEN" resolves back to DOWNSTREAM_LICENSE_TOKEN).
      It starts, it works, and it separates nothing.
    """
    if not SYNTH_TOKEN_IS_SHARED:
        return

    if SYNTH_TOKEN_IS_EXPLICIT_DUPLICATE:
        cause = (
            "TTS_SYNTH_TOKEN is set but is byte-identical to TTS_GATEWAY_TOKEN, so "
            "nothing is separated — this is the no-op, not the fix"
        )
    else:
        cause = (
            "TTS_SYNTH_ALLOW_SHARED_TOKEN is set, so the gateway token is reused "
            "deliberately — this is an ACCEPTED exposure, not a fixed one"
        )

    _log(
        f"WARNING: {cause}. Synth calls to {SYNTH} carry the fleet-wide "
        f"DOWNSTREAM_LICENSE_TOKEN — the credential every w2v service authenticates "
        f"cross-service calls with, and which can claim/complete/fail TTS jobs for "
        f"EVERY tenant. Closing it needs a synth-scoped license the synth target will "
        f"actually accept."
    )


def main() -> None:
    lanes = "+".join(
        lane for lane, on in (("cell", CELL_SIGNER_CONFIGURED), ("licence", TOKEN)) if on
    )
    _log(f"worker {WORKER_ID} → gateway {GATEWAY} [{lanes}] via kokoro {SYNTH}")
    _warn_if_sharing_gateway_token()
    _install_shutdown_hooks()
    with _gateway_client(timeout=60) as gateway, _synth_client(timeout=60) as synth:
        while True:
            try:
                r = gateway.post(
                    f"{GATEWAY}/api/tts/claim", headers=_GW_H, json={"worker_id": WORKER_ID}
                )
                if r.status_code == 204:
                    time.sleep(POLL_IDLE_SECONDS)
                    continue
                r.raise_for_status()
                cooldown = _process(gateway, synth, r.json())
                if cooldown:
                    time.sleep(cooldown)
            except Exception as exc:  # noqa: BLE001 -- keep the loop alive through transients
                _log(f"claim-loop error: {exc}")
                time.sleep(POLL_IDLE_SECONDS)


if __name__ == "__main__":
    main()
