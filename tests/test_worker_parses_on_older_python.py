"""``worker/worker.py`` must still PARSE on Python 3.13.

THE TRAP. The bare multi-exception form ``except A, B:`` is PEP 758 and parses only
on Python >= 3.14. ``worker.py`` is the container entrypoint for BOTH worker
Deployments, so on a base-image downgrade that form is not a test failure — it is a
startup ``SyntaxError`` before the first line runs, taking the whole worker fleet
down with no log, no probe and no partial behaviour to debug from.

WHY A TEST AND NOT JUST PARENTHESES. ``except (A, B):`` does not survive here.
MEASURED on this file with this repo's pyproject (2026-09-15): ``ruff format``
rewrites it straight back to the bare form, on ruff 0.15.0 / 0.15.20 / 0.16.0 /
0.16.5 — the whole range the ``ruff>=0.15.0`` dev floor resolves to. So a
parenthesised fix silently un-fixes itself on the next format run. The code binds
the tuples to names (``_UUID_PARSE_ERRORS``, ``_NUMERIC_COERCION_ERRORS``) instead.

The rewrite is gated on ``target-version``: under ``py314`` (this repo) ruff
rewrites; under ``py313``, or in a scratch dir that loads no config, the same ruff
says "1 file left unchanged". Measuring outside this repo's config therefore gives
the opposite answer — an audit did exactly that and concluded parentheses were
safe. They are not, here.

But note what this test does and does not depend on: it asserts only that
``worker.py`` PARSES under the 3.13 grammar. It holds whatever the formatter does,
and it is the thing that actually protects the fleet — the formatter argument only
explains why the source is written the way it is.

``feature_version`` makes the check real on a 3.14 interpreter: ``compile()`` applies
the 3.13 grammar rules to source it is otherwise perfectly capable of parsing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

WORKER_SOURCE = Path(__file__).resolve().parents[1] / "worker" / "worker.py"


def test_the_bare_except_form_is_not_reintroduced() -> None:
    src = WORKER_SOURCE.read_text(encoding="utf-8")

    try:
        ast.parse(src, feature_version=(3, 13))
    except SyntaxError as exc:  # pragma: no cover - the failure path is the point
        pytest.fail(
            f"worker/worker.py no longer parses under Python 3.13 ({exc.msg}, line {exc.lineno}). "
            "A base-image downgrade would now CrashLoop every worker at import. Bind the "
            "exception group to a module-level tuple name — under this repo's "
            '`target-version = "py314"`, `except (A, B):` is rewritten back to the '
            "3.14-only bare form by `ruff format` (measured, see module docstring)."
        )


def test_the_control_confirms_the_check_can_fail() -> None:
    """A test that cannot fail is worse than no test: prove the 3.13 grammar is applied."""
    with pytest.raises(SyntaxError):
        ast.parse(
            "try:\n    pass\nexcept ValueError, TypeError:\n    pass\n", feature_version=(3, 13)
        )
