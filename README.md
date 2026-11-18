# TTS Gateway

Database-backed TTS claim-queue that fans audiobook and narration synthesis across many remote kokoro workers instead of one serialized box. Port 6320.

## What TTS Gateway Does

When the orchestrator's narrate stage needs a book turned into audio, it POSTs a batch of text chunks to `/api/tts/batches`. Each chunk carries the text to synthesize plus optional voice and speed. The gateway hashes every chunk into a stable `idempotency_key = sha256(tenant | voice | speed | preproc-version | text)`, inserts one `tts_jobs` row per unique chunk under a `(tenant_id, idempotency_key)` unique index with `ON CONFLICT DO UPDATE`, and returns `202 Accepted` with a `job_id` for every input chunk. Because submission is idempotent, a resumed narrate that re-submits the same book collapses onto the existing rows — many already `done` — so no synthesis is ever repeated. A re-attempt under a new `batch_id` re-tags the deduped rows onto the new batch (leaving status and result untouched) so the caller's poll still sees its own finished clips.

The work is drained by a pool of remote kokoro workers — thin pull-loops running as launchd units on the mini Macs, alongside the already-running kokoro backend at `tts.test:9202`. Each worker POSTs `/api/tts/claim` with its `worker_id`; the gateway calls the SECURITY DEFINER `claim_next_tts_job(worker_id, lease_seconds, max_attempts)` resolver, which atomically picks one claimable row with `FOR UPDATE SKIP LOCKED`, fails poison jobs that have exhausted `max_attempts`, otherwise moves the row to `claimed` with a fresh lease and an incremented attempt count, and returns the synthesis inputs. The claim `ORDER BY` prefers the tenant currently holding the fewest live leases, so N books drain concurrently and a short lesson never starves behind a six-hour audiobook. While synthesizing, the worker heartbeats `/api/tts/jobs/{job_id}/heartbeat` to renew its lease; a worker that dies silently has its job reclaimed once the lease expires.

On success the worker uploads the WAV to `/api/tts/jobs/{job_id}/result` as multipart with a `duration_seconds` field. The gateway writes the clip to its own MinIO bucket (`tts-gateway`) under the deterministic key `{tenant_id}/{idempotency_key}.wav`, marks the job `done`, and records the duration. On failure the worker reports `/api/tts/jobs/{job_id}/fail`; the gateway requeues the job if attempts remain, else marks it `failed`. The orchestrator polls `/api/tts/batches/{batch_id}` — one call returns every job's status, and for each `done` job the gateway presigns a fresh result URL at read time so the link never expires before narrate assembles the book.

The gateway is the only component that touches Postgres or MinIO; the bare-metal Macs never do. Adding a worker is exactly +1 unit of parallelism, bounded only by each box's single kokoro synth lock. The queue design is cloned from `html_render_service`'s proven `FOR UPDATE SKIP LOCKED` lease queue — the one structural change is that the worker pool is remote and drives claim/heartbeat/result over HTTP. The submit/poll surface is tenant-scoped (its session binds `app.current_tenant` for RLS enforce), while the worker surface is deliberately cross-tenant: the SECURITY DEFINER resolver binds no tenant, and a worker holds only a service-account license token, never database credentials.

## Features

- **Idempotent batch submit** — dedupes chunks on `sha256(tenant|voice|speed|preproc-version|text)` so a resumed narrate re-submitting the same book reuses existing jobs instead of re-synthesizing every chunk
- **Durable Postgres claim-queue** — one `tts_jobs` row per synthesis request survives pod restarts, so an in-flight book is never lost when a gateway replica dies
- **Atomic cross-tenant claim** — the `claim_next_tts_job` SECURITY DEFINER resolver picks one job with `FOR UPDATE SKIP LOCKED`, so many workers claim disjoint jobs without contention or double-work
- **Per-tenant fairness** — the claim ordering prefers the tenant with the fewest live leases so concurrent books drain in parallel and short jobs never starve behind long ones
- **Lease-based reclaim** — a claimed job whose lease expires (a dead or wedged worker) becomes claimable again, so no job is permanently stuck
- **Poison-job cap** — a chunk that crashes every worker is failed after `max_attempts` so it cannot starve the queue forever
- **Remote worker pool** — thin launchd pull-workers reuse each Mac's already-running kokoro (`tts.test:9202`), so a 16 GB box runs one model, and adding a box is one more registered worker
- **Presign-at-read result URLs** — done clips are presigned when the batch is polled, not when stored, so the URL is always fresh when the orchestrator assembles the audiobook
- **Preprocessing-versioned cache** — the `PREPROC_VERSION` token folds into the dedupe key so a tts_backend pre-processing change (e.g. the misaki G2P cutover) forces a clean re-synth instead of replaying stale audio
- **Non-fatal storage init** — the queue serves even if MinIO is briefly unreachable at startup; the bucket is re-ensured on first use rather than crashlooping the pod
- **License-token auth** — every request (orchestrator and worker) authenticates with `X-Tenant-Id: Bearer <license_token>` so the whole surface is tenant-isolated

## Authentication

Authentication is always on: `AuthMiddleware` from `w2v_common` is added unconditionally in `create_app()` (there is no disable flag). Both the orchestrator (submit/poll) and the remote kokoro workers (claim/heartbeat/result) authenticate with a license token carried in `X-Tenant-Id: Bearer <license_token>` — `Authorization` is reserved for upstream Keycloak JWTs. Tokens are introspected against the licenses service and cached ~5 minutes (positive cache TTL 300s), with a short negative cache and a stale-while-error fallback so a briefly slow licenses endpoint does not spuriously 401 a long-running narrate.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | Keycloak JWT | `Authorization: Bearer <jwt>` validated via Keycloak when `KEYCLOAK_URL` is set |
| 2 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, ~5-min cache |
| 3 | Trusted network | Synthetic in-cluster identity for private CIDRs — **off by default** (`AUTH_TRUST_INCLUSTER_IDENTITY=false`) |

Exempt paths (no token required): `/`, `/health`, `/healthz`, `/livez`, `/live`, `/readyz`, `/ready`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`, and the `/health/` prefix.

The submit/poll path binds `app.current_tenant` on its session so RLS enforce scopes each tenant to its own jobs; the worker claim path uses a raw (no-tenant) session because the SECURITY DEFINER resolver bypasses RLS and returns each job's tenant.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6320
uv run pytest tests/ -v
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Batches `/api/tts` (orchestrator-facing, tenant-scoped)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/batches` | Enqueue a book's chunks idempotently (returns `202` with a job ref per chunk) |
| GET | `/batches/{batch_id}` | Poll every job in the batch with status and presigned result URLs |

### Workers `/api/tts` (worker-facing, cross-tenant)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/claim` | Claim the next runnable job for a `worker_id` (`204` if the queue is empty) |
| POST | `/jobs/{job_id}/heartbeat` | Renew a claimed job's lease |
| POST | `/jobs/{job_id}/result` | Upload the synthesized WAV plus duration and mark the job done |
| POST | `/jobs/{job_id}/fail` | Report a synthesis failure (requeue if attempts remain, else poison) |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness / service info |
| GET | `/livez` | Liveness probe |
| GET | `/readyz` | Readiness probe (runs `SELECT 1` against Postgres) |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `6320` | Listen port (rejects `8000` via the w2v_common base) |
| `DATABASE_URL` | -- (required) | Owner DB URL used by alembic / the initContainer |
| `RUNTIME_DATABASE_URL` | -- | Restricted app-role URL for RLS enforce; falls back to `DATABASE_URL` |
| `QUEUE_LEASE_SECONDS` | `180` | Lease TTL; a worker must heartbeat within this or its job is reclaimed |
| `QUEUE_MAX_ATTEMPTS` | `4` | Claim attempts before a chunk is failed as poison |
| `BACKEND` | `local` | Storage backend (`minio` in cluster) |
| `MINIO_ENDPOINT` | `` | MinIO endpoint (`host:port`); required when backend is `minio` |
| `MINIO_ACCESS_KEY` | `` | MinIO access key |
| `MINIO_SECRET_KEY` | `` | MinIO secret key |
| `MINIO_BUCKET` | `tts-gateway` | Bucket for synthesized clips |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO |
| `MINIO_EXTERNAL_ENDPOINT` | -- | Rewrite presigned-URL host for clients outside the cluster |
| `PRESIGN_EXPIRY_SECONDS` | `3600` | Presigned result URL expiry |
| `LICENSE_INTROSPECT_URL` | discovery `licenses/license/introspect` | License validation endpoint (w2v_common) |
| `AUTH_TRUST_INCLUSTER_IDENTITY` | `false` | Opt-in trusted-network fail-open (w2v_common) |

## Architecture

```
src/
├── asgi.py                          # FastAPI app factory, AuthMiddleware, lifespan (MinIO init/close)
├── config.py                        # Pydantic settings: port 6320, DB URLs, QueueSettings, StorageSettings
├── api/
│   ├── schemas.py                   # Submit/poll + claim/result request/response models
│   └── routes/
│       ├── batches.py               # POST /batches (submit) + GET /batches/{id} (poll)
│       ├── workers.py               # claim / heartbeat / result / fail (worker-facing)
│       └── health.py                # /health, /livez, /readyz probes
├── database/
│   ├── models.py                    # TtsJob SQLAlchemy model + lease/dedupe indexes
│   ├── queue_resolvers.py           # SECURITY DEFINER claim_next_tts_job fn (registered on create_all)
│   ├── session.py                   # Async engine + tenant-scoped and raw session dependencies
│   └── migrations/
│       ├── env.py                   # Alembic environment
│       └── versions/
│           └── 0001_initial.py      # Initial tts_jobs schema + claim fn
└── services/
    ├── job_manager.py               # Queue lifecycle: submit / claim / heartbeat / complete / fail
    └── storage.py                   # MinIO clip storage (put + presign-at-read)
worker/
└── worker.py                        # Pull-worker. BOTH instances are cluster Deployments
                                     # (tts-gateway-worker → in-cluster modality-gateway;
                                     # tts-gateway-worker-mini1 → tts-mini1.test:9202).
                                     # It is NOT a bare-Mac process: it imports w2v_common.
```

## Testing

```bash
uv run pytest tests/ -v                     # Queue-core concurrency tests (real Postgres 18)
uv run pytest tests/test_queue.py -q        # 10 tests: idempotent submit, SKIP-LOCKED, lease reclaim, poison, fairness
uv run ruff check src/ tests/               # Lint
uv run ruff format --check src/ tests/      # Format check
```

Tests require a real Postgres 18 (psycopg3) with the claim function installed via `register_claim_fn`. Point `DATABASE_URL` (or `TEST_DATABASE_URL`) at it; the conftest defaults to `postgresql+psycopg://localhost:5434/tts_gateway_test`:

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=tts_gateway_test -p 5434:5432 postgres:18
```

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and `scripts/pip-audit.sh`.

No image is published from this repository; build it locally with the Dockerfile.

## Docker

```bash
docker build -t tts-gateway .
docker run -p 6320:6320 -e DATABASE_URL=postgresql+psycopg://host/tts_gateway tts-gateway
docker run -p 6320:6320 -e DATABASE_URL=postgresql+psycopg://host/tts_gateway -e BACKEND=minio -e MINIO_ENDPOINT=minio.test:9000 tts-gateway
```

## Dependencies

- **w2v_orchestrator** — submits narrate batches and polls for results (the sole submit/poll client)
- **kokoro TTS backend** (`tts.test:9202`) — the remote workers synthesize by calling its `POST /api/tts/synthesize`; the gateway never calls it directly
- **licenses service** — validates `X-Tenant-Id` license tokens via introspection
- **PostgreSQL 18** — durable `tts_jobs` queue via PgBouncer (`pgbouncer.pgbouncer:6432`), psycopg3 only
- **MinIO** — `tts-gateway` bucket for synthesized WAV clips
- **FastAPI** — web framework (`fastapi[all]`)
- **Uvicorn** — ASGI server
- **SQLAlchemy** (asyncio) — ORM and queue queries
- **psycopg** (binary, pool) — Postgres 18 driver
- **Alembic** — schema migrations
- **miniopy-async** — async MinIO client
- **httpx** — async HTTP client (worker synth calls)
- **Pydantic / pydantic-settings** — schema and settings validation
- **w2v-common** (`>=0.30.0`) — AuthMiddleware, StorageSettings, discovery, RLS helpers

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
