"""Thin web adapter over the canonical engine executor (`hta.modules.executor`).

The statistical implementation now lives in the engine and returns a Pydantic `TestResult`;
this adapter serialises it to the plain dict the SSE pipeline and the report assembler expect,
and turns an unrecognised test name into a graceful UNTESTABLE result so a run never 500s.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Make the engine (`hta`, under src/) importable.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hta.modules.executor import execute as _execute  # noqa: E402


def _error_result(test_name: str, message: str) -> dict[str, Any]:
    """A TestResult-shaped dict for the case the engine could not run (e.g. no test
    selected). Kept as a plain dict so it can carry a non-enum placeholder name."""
    return {
        "test_used": test_name or "—",
        "statistic": 0.0,
        "p_value": 1.0,
        "degrees_of_freedom": None,
        "effect_size": {"measure_name": "—", "value": 0.0,
                        "interpretation": "negligible", "ci_lower": 0.0, "ci_upper": 0.0},
        "assumption_checks": [{"assumption_name": "Execution", "status": "UNTESTABLE",
                               "note": message}],
        "confidence_interval": [0.0, 0.0],
        "is_significant": False,
        "power": None,
        "notes": [message],
    }


def execute(
    test_name: str,
    df: pd.DataFrame,
    outcome: str,
    group: Optional[str],
    predictor: Optional[str],
    design: dict[str, Any],
    selection: Any,
    predictors: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run `test_name` and return a TestResult-shaped dict."""
    try:
        result = _execute(test_name, df, outcome, group, predictor, design, selection,
                          predictors=predictors)
    except Exception as exc:
        return _error_result(test_name, f"{test_name or 'no test'} could not be run: {exc}")
    return result.model_dump(mode="json")
