"""The /claim RESPONSE must carry the job's tenant — not merely the schema.

WHY THIS FILE EXISTS. The first version of this coverage asserted only that
``"tenant_id" in ClaimedJob.model_fields``. That is a vacuous floor: deleting the
``tenant_id=...`` argument in ``routes/workers.py::claim`` — the half that actually
ships the value to the worker — left every test green, because an optional field
defaults to ``None`` and the schema still declares it. The worker can attribute a
synth hop only to a tenant the gateway put ON THE WIRE, so the wire is what has to
be asserted.

These cases drive the real router and read the real HTTP body:

* a claimed row's tenant appears in the response payload, as a string;
* a row with no tenant serialises ``None`` rather than exploding — an older DB row
  or a mixed-version rollout must still hand the worker a usable job;
* the field is genuinely optional on the schema, so a newer worker paired with an
  older gateway keeps synthesizing (unattributed) instead of failing to parse.

No DB, no MinIO: ``claim_next`` and the session dependency are stubbed, exactly as
in tests/test_worker_authz.py, so what is exercised is the handler's serialisation.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.api.routes import workers
from src.database.session import get_raw_session

WORKER_SUBJECT = "11111111-1111-1111-1111-111111111111"
JOB_TENANT = uuid.UUID("3f1c2d4e-5a6b-4c7d-8e9f-0a1b2c3d4e5f")


@pytest.fixture
def claimed_row(monkeypatch) -> dict[str, object]:
    """The dict ``claim_next`` returns. Mutated per-test to vary the tenant."""
    row: dict[str, object] = {
        "job_id": "job-1",
        "voice": "af_heart",
        "speed": 1.0,
        "provider": "kokoro",
        "text": "chapter one",
        # A uuid.UUID, as the resolver really returns — the handler is responsible
        # for stringifying it. A str() fixture here would hide a serialisation bug.
        "tenant_id": JOB_TENANT,
    }

    async def _claim_next(_session, **_kw) -> dict[str, object]:
        return row

    monkeypatch.setattr(workers.jm, "claim_next", _claim_next)
    monkeypatch.setattr(
        workers,
        "get_settings",
        lambda: SimpleNamespace(
            max_result_bytes=1024,
            queue=SimpleNamespace(
                lease_seconds=180,
                max_attempts=4,
                max_job_lifetime_seconds=3600,
                max_busy_age_seconds=86400,
            ),
            worker_subject_id_set=lambda: frozenset({WORKER_SUBJECT}),
        ),
    )
    return row


@pytest_asyncio.fixture
async def api(claimed_row) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(workers.router)
    app.dependency_overrides[get_raw_session] = lambda: None
    set_identity_context(
        tenant_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        license_id="lic-1",
        subject_id=WORKER_SUBJECT,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tts.test"
    ) as client:
        yield client
    clear_identity_context()


async def test_claim_response_carries_the_jobs_tenant(api) -> None:
    """THE WIRE. Delete ``tenant_id=...`` in routes/workers.py::claim and this fails."""
    response = await api.post("/api/tts/claim", json={"worker_id": "mini2"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tenant_id"] == str(JOB_TENANT), (
        "the claim payload did not put the job's tenant on the wire — the worker has "
        f"nothing to attribute the synth hop with (got {body.get('tenant_id')!r})"
    )
    # The rest of the payload must be unharmed: a worker that cannot synthesize is a
    # worse outcome than one that cannot attribute.
    assert body["job_id"] == "job-1"
    assert body["text"] == "chapter one"


async def test_a_row_without_a_tenant_still_claims(api, claimed_row) -> None:
    """Fail safe on the gateway half too: an untenanted row must yield a usable job."""
    claimed_row["tenant_id"] = None

    response = await api.post("/api/tts/claim", json={"worker_id": "mini2"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tenant_id"] is None
    assert body["job_id"] == "job-1"


def test_tenant_id_is_optional_on_the_schema() -> None:
    """An older gateway (no tenant_id) must still parse in a newer worker."""
    from src.api.schemas import ClaimedJob

    field = ClaimedJob.model_fields["tenant_id"]
    assert not field.is_required()
    assert (
        ClaimedJob(job_id="j", voice="af_heart", speed=1.0, provider="kokoro", text="t").tenant_id
        is None
    )
