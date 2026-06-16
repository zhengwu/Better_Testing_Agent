"""Analysis run endpoint — SSE streaming pipeline execution.

Dry-run mode streams the canned ``STUB_REPORT``; live mode runs the real pipeline:
profile (already written by set_variables) → select test (hta.modules.selector) →
execute (executor.py) → report (reporter.py).
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from web.backend.config import DRY_RUN
from web.backend.executor import execute
from web.backend.plots import plotspec_to_plotly
from web.backend.reporter import build_report, enrich_prose_with_llm
from web.backend.storage.local import LocalStorage
from web.backend.stubs import STUB_REPORT

# Make the statistical engine (`hta`, under src/) importable for the
# function-level imports used inside the live pipeline below.
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

router = APIRouter()
store = LocalStorage()

_NUMERIC_TYPES = ("CONTINUOUS", "ORDINAL", "COUNT")

DEFAULT_DESIGN: dict[str, Any] = {
    "design_type": "OBSERVATIONAL",
    "measurement_type": "BETWEEN_SUBJECTS",
    "is_randomized": False,
    "confounders": [],
    "notes": ["No design dialogue completed; assuming an observational, "
              "between-subjects design."],
}


def _sse(data: object) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _enrich_plots(report: dict) -> dict:  # type: ignore[type-arg]
    """Convert PlotSpec data dicts to plotly_json fields in-place."""
    for plot in report.get("plots", []):
        plot["plotly_json"] = plotspec_to_plotly(plot)
    return report


async def _run_dry_run(session_id: str) -> object:
    """Yield SSE progress events then the stub report."""
    stages = [
        ("selecting_test", "Selecting statistical test…"),
        ("executing_test",  "Computing Spearman rank correlation…"),
        ("generating_report", "Generating report…"),
    ]
    for stage, msg in stages:
        await asyncio.sleep(0.6)
        yield _sse({"type": "progress", "stage": stage, "message": msg})

    report = _enrich_plots(dict(STUB_REPORT))

    # Merge real profile if available
    if store.exists(session_id, "profile.json"):
        report["data_profile"] = json.loads(store.read(session_id, "profile.json"))

    # Merge real design if available
    if store.exists(session_id, "design.json"):
        report["study_design"] = json.loads(store.read(session_id, "design.json"))

    store.write_json(session_id, "report.json", report)
    store.set_status(session_id, "COMPLETE")

    yield _sse({"type": "result", "report": report})


def _choose_predictor(df: pd.DataFrame, cols: dict, outcome: str,  # type: ignore[type-arg]
                      group: Optional[str],
                      pool: Optional[list[str]] = None) -> Optional[str]:
    """Pick the numeric column most BET-dependent with the outcome.

    `pool`, when provided, restricts the search to those columns (used when
    the user selected 3+ variables and wants the strongest among them).
    Falls back to all numeric columns when pool is absent or empty."""
    if group:
        return None
    if outcome not in cols or cols[outcome].var_type not in _NUMERIC_TYPES:
        return None
    if pool:
        candidates = [c for c in pool
                      if c != outcome and c in cols and cols[c].var_type in _NUMERIC_TYPES]
    else:
        candidates = [c for c in df.columns
                      if c != outcome and cols[c].var_type in _NUMERIC_TYPES]
    if not candidates:
        return None
    from hta.bet_screen import maxbet
    y = pd.to_numeric(df[outcome], errors="coerce")
    best: Optional[tuple[str, float]] = None
    for c in candidates:
        sub = pd.DataFrame({"x": pd.to_numeric(df[c], errors="coerce"), "y": y}).dropna()
        if len(sub) < 8:
            continue
        try:
            z = maxbet(sub["x"].tolist(), sub["y"].tolist(), seed=0).bet_z
        except Exception:
            z = -1.0
        if best is None or z > best[1]:
            best = (c, z)
    return best[0] if best else candidates[0]


async def _run_live(session_id: str) -> object:
    """Run the real pipeline and stream progress + the final report."""
    from hta.modules.profiler import profile_column
    from hta.modules.selector import select

    variables = json.loads(store.read(session_id, "variables.json"))
    outcome = variables.get("outcome_variable")
    group = variables.get("group_variable") or None
    hypothesis = variables.get("hypothesis", "")

    df = pd.read_csv(io.BytesIO(store.read(session_id, "data.csv")))

    profile = (json.loads(store.read(session_id, "profile.json"))
               if store.exists(session_id, "profile.json") else None)
    if profile is None:
        from web.backend.api.sessions import _build_profile
        profile = _build_profile(df, outcome, group)
        store.write_json(session_id, "profile.json", profile)

    design = (json.loads(store.read(session_id, "design.json"))
              if store.exists(session_id, "design.json") else dict(DEFAULT_DESIGN))

    cols = {c: profile_column(c, df[c].apply(str).tolist()) for c in df.columns}
    raw = {c: df[c].apply(str).tolist() for c in df.columns}
    sel_vars = variables.get("selected_variables") or []
    # When 3+ variables are selected (no group), the full pool becomes the predictor list
    # for multi-predictor regression.  For 1–2 vars we keep the single-predictor path.
    pool = [v for v in sel_vars if v != outcome and v != group] if sel_vars else []
    multi_predictor_mode = len(pool) >= 2 and not group
    if multi_predictor_mode:
        # All pool members are jointly modelled; no single "best" predictor needed.
        predictors_list: list[str] | None = [
            v for v in pool if v in cols and cols[v].var_type in _NUMERIC_TYPES
        ]
        predictor = predictors_list[0] if predictors_list else None
        extra_predictors = list(predictors_list[1:]) if predictors_list else None
    else:
        predictors_list = None
        predictor = (variables.get("predictor_variable") or None) or _choose_predictor(
            df, cols, outcome, group, pool=pool or None)
        extra_predictors = [v for v in pool if v != predictor] or None

    # ── Step B: select test ───────────────────────────────────────────────────
    yield _sse({"type": "progress", "stage": "selecting_test",
                "message": "Selecting statistical test…"})
    await asyncio.sleep(0)
    selection = select(cols, outcome, group, predictor, hypothesis, raw,
                       predictors=predictors_list)
    test_name = selection.test
    if test_name in ("—", "", None):
        # Defensive fallback: pick a sensible test from what we have.
        test_name = "PEARSON_CORRELATION" if predictor else "—"
        selection.test = test_name

    # ── Step C: execute ───────────────────────────────────────────────────────
    yield _sse({"type": "progress", "stage": "executing_test",
                "message": f"Running {test_name.replace('_', ' ').title()}…"})
    await asyncio.sleep(0)
    test_result = execute(test_name, df, outcome, group, predictor, design, selection,
                          predictors=predictors_list)

    # ── Step D: report ────────────────────────────────────────────────────────
    yield _sse({"type": "progress", "stage": "generating_report",
                "message": "Generating report…"})
    await asyncio.sleep(0)
    report = build_report(profile, design, test_result, selection, df,
                          outcome, group, predictor, hypothesis,
                          extra_predictors=extra_predictors)
    _enrich_plots(report)

    # ── Step E: LLM prose enrichment ─────────────────────────────────────────
    # Run the sync Anthropic/OpenAI call in a thread-pool executor so the event
    # loop isn't blocked during the network round-trip (30 s timeout on the client).
    yield _sse({"type": "progress", "stage": "enriching_prose",
                "message": "Generating plain-language summary…"})
    loop = asyncio.get_event_loop()
    from functools import partial as _partial
    report = await loop.run_in_executor(
        None, _partial(enrich_prose_with_llm, report,
                       outcome=outcome, predictor=predictor,
                       extra_predictors=extra_predictors,
                       predictors=predictors_list))

    store.write_json(session_id, "report.json", report)
    store.set_status(session_id, "COMPLETE")
    yield _sse({"type": "result", "report": report})


@router.get("/sessions/{session_id}/preview-test")
async def preview_test(session_id: str) -> dict[str, Any]:
    """Return the test the selector would pick, without executing it.

    Used by the Step 4 Review screen to show the planned test + rationale before the user
    clicks Run, so they can go back and adjust variables if the selection looks wrong.
    """
    if not store.exists(session_id, "metadata.json"):
        raise HTTPException(status_code=404, detail="Session not found.")
    if not store.exists(session_id, "data.csv") or not store.exists(session_id, "variables.json"):
        raise HTTPException(status_code=409,
                            detail="Upload a CSV and set variables before previewing.")

    if DRY_RUN:
        return {
            "test_name": "SPEARMAN_CORRELATION",
            "rationale": "Dry-run mode: the stub dataset uses Spearman rank correlation.",
            "caveats": [],
        }

    variables = json.loads(store.read(session_id, "variables.json"))
    outcome = variables.get("outcome_variable", "")
    group = variables.get("group_variable") or None
    hypothesis = variables.get("hypothesis", "")

    if not outcome:
        raise HTTPException(status_code=409, detail="No outcome variable set.")

    df = pd.read_csv(io.BytesIO(store.read(session_id, "data.csv")))
    if outcome not in df.columns:
        raise HTTPException(status_code=409,
                            detail=f"Outcome column '{outcome}' not found in data.")

    from hta.modules.profiler import profile_column
    from hta.modules.selector import select

    cols = {c: profile_column(c, df[c].apply(str).tolist()) for c in df.columns}
    raw = {c: df[c].apply(str).tolist() for c in df.columns}
    sel_vars = variables.get("selected_variables") or []
    pool = [v for v in sel_vars if v != outcome and v != group] if sel_vars else []
    multi_pred = len(pool) >= 2 and not group
    if multi_pred:
        predictors_list: list[str] | None = [
            v for v in pool if v in cols and cols[v].var_type in _NUMERIC_TYPES
        ]
        predictor = predictors_list[0] if predictors_list else None
    else:
        predictors_list = None
        predictor = (variables.get("predictor_variable") or None) or _choose_predictor(
            df, cols, outcome, group, pool=pool or None)
    selection = select(cols, outcome, group, predictor, hypothesis, raw,
                       predictors=predictors_list)

    return {
        "test_name": selection.test,
        "rationale": selection.rationale,
        "caveats": selection.caveats,
    }


@router.post("/sessions/{session_id}/run")
async def run_analysis(session_id: str) -> StreamingResponse:
    if not store.exists(session_id, "metadata.json"):
        raise HTTPException(status_code=404, detail="Session not found.")

    store.set_status(session_id, "RUNNING")
    generator = _run_dry_run(session_id) if DRY_RUN else _run_live(session_id)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
