"""``ChunkIn.speed`` must be bounded at the API boundary.

The whole batch is inserted in ONE ``pg_insert`` into a ``Numeric(4,3)`` column
(max magnitude 9.999). An out-of-range speed (``12.0``/``inf``) overflows that
single statement → 500 → the ENTIRE book batch fails to enqueue. A ``nan``/``0.0``
stores fine but is later forwarded to kokoro, which asserts ``0.5 <= speed <= 2.0``
(see ``tts_backend/src/services/ssml_processor.py``), poisoning the job. So every
bad value must be rejected with 422 at the schema, never reaching the DB or kokoro.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.api.schemas import ChunkIn, SubmitBatchRequest


def _chunk(speed: float) -> ChunkIn:
    return ChunkIn(ref="r1", text="hello", speed=speed)


def test_default_speed_is_one() -> None:
    assert ChunkIn(ref="r1", text="hello").speed == 1.0


def test_legit_speed_accepted() -> None:
    assert _chunk(1.5).speed == 1.5
    assert _chunk(0.5).speed == 0.5
    assert _chunk(2.0).speed == 2.0


@pytest.mark.parametrize("bad", [12.0, 9.9999, 2.0001, 3.0])
def test_overflow_speed_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        _chunk(bad)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_speed_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        _chunk(bad)


def test_zero_and_negative_speed_rejected() -> None:
    # 0.0 previously slipped through submit_batch/worker's ``or 1.0`` coercion —
    # a silent footgun. It is now rejected (422) rather than silently becoming 1.0.
    with pytest.raises(ValidationError):
        _chunk(0.0)
    with pytest.raises(ValidationError):
        _chunk(-1.0)
    with pytest.raises(ValidationError):
        _chunk(0.25)  # below kokoro's 0.5 floor


def test_batch_with_one_bad_chunk_rejected() -> None:
    """One poison chunk must fail validation for the whole request body — before
    it ever reaches the single ``pg_insert`` that would 500 on overflow."""
    payload = {
        "batch_id": "book-1",
        "chunks": [
            {"ref": "a", "text": "ok", "speed": 1.0},
            {"ref": "b", "text": "boom", "speed": 12.0},
        ],
    }
    with pytest.raises(ValidationError):
        SubmitBatchRequest.model_validate(payload)


def test_kokoro_range_boundaries_are_inclusive() -> None:
    # Matches tts_backend's ``Field(default=1.0, ge=0.5, le=2.0)``.
    assert math.isclose(_chunk(0.5).speed, 0.5)
    assert math.isclose(_chunk(2.0).speed, 2.0)
