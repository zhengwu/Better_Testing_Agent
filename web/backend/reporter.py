"""Thin web adapter over the canonical engine reporter (`hta.modules.reporter`).

Parses the stored dict boundary (profile/design/test_result JSON) into Pydantic models,
calls the engine to build a typed `Report`, then serialises back to a dict and re-attaches
the presentation-only BET EDA plots (which carry `plotly_json` and are not part of the typed
`Report.plots`). The rare "no test could be selected" result cannot populate the strict
`TestResult` enum, so it is reported via a minimal fallback that still surfaces caveats/plots.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Make the engine (`hta`, under src/) importable.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hta.models.data import DataProfile  # noqa: E402
from hta.models.design import (  # noqa: E402
    MeasurementType,
    StudyDesign,
    StudyDesignType,
)
from hta.models.test import TestResult  # noqa: E402
from hta.modules.reporter import build_report as _build_report  # noqa: E402

from web.backend.config import (  # noqa: E402
    DRY_RUN, LLM_PROVIDER,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, IS_AZURE_OPENAI,
)


def _default_design() -> StudyDesign:
    return StudyDesign(
        design_type=StudyDesignType.OBSERVATIONAL,
        measurement_type=MeasurementType.BETWEEN_SUBJECTS,
        is_randomized=False,
        notes=["No design dialogue completed; assuming an observational design."],
    )


def _eda_var_names(p: dict[str, Any]) -> tuple[str, str]:
    return (
        p.get("x_label", "").split(" (")[0],
        p.get("y_label", "").split(" (")[0],
    )


def _predictor_qq_plot(df: pd.DataFrame, predictor: str) -> dict[str, Any] | None:
    """Normal Q–Q plot for the predictor variable (mirror of the engine's outcome QQ)."""
    try:
        from scipy import stats as _stats
        vals = pd.to_numeric(df[predictor], errors="coerce").dropna()
        if len(vals) < 8:
            return None
        (theoretical, sample), _ = _stats.probplot(vals)
        return {
            "plot_type": "qqplot",
            "title": f"Normal Q–Q plot — {predictor}",
            "x_label": "Theoretical quantiles",
            "y_label": "Sample quantiles",
            "data": {"theoretical": list(theoretical), "sample": list(sample)},
        }
    except Exception:
        return None


def _pair_bet_plot(df: pd.DataFrame, outcome: str, predictor: str) -> dict[str, Any] | None:
    """BET copula interaction plot for the exact outcome × predictor pair."""
    try:
        from hta.bet_screen import interaction_plot
        from web.backend.plots import plotspec_to_plotly

        x_vals = pd.to_numeric(df[outcome], errors="coerce").dropna()
        y_vals = pd.to_numeric(df[predictor], errors="coerce").dropna()
        idx = x_vals.index.intersection(y_vals.index)
        if len(idx) < 10:
            return None

        ip = interaction_plot(
            x_vals.loc[idx].tolist(), y_vals.loc[idx].tolist(),
            x_name=outcome, y_name=predictor, seed=0,
        )
        # Reuse the plotspec builder from sessions.py
        from web.backend.api.sessions import _interaction_plotspec
        spec = _interaction_plotspec(ip, color_by="interaction")
        spec["plotly_json"] = plotspec_to_plotly(spec)
        return spec
    except Exception:
        return None


def _attach_eda(
    report_dict: dict[str, Any],
    profile: dict[str, Any],
    df: pd.DataFrame | None = None,
    outcome: str | None = None,
    predictor: str | None = None,
    extra_predictors: list[str] | None = None,
) -> None:
    """Attach relevant BET EDA plots to the report.

    When outcome and predictor are both known:
    - Insert a Normal Q–Q plot for the predictor (if not already present).
    - Ensure the exact outcome × predictor BET copula plot is present.
    - Keep EDA pair plots that involve the main pair OR any extra selected
      variables (from a 3+ variable pool); drop unrelated Explore pairs.
    - Always keep the dependence network (it shows the full picture).
    - For each extra predictor, ensure its BET copula with the outcome is
      also present as a supplementary plot so all selected pairs are shown.
    """
    from web.backend.plots import plotspec_to_plotly

    eda: list[dict[str, Any]] = list(profile.get("eda_plots") or []) \
        if isinstance(profile, dict) else []

    if outcome and predictor and df is not None:
        target = {outcome, predictor}
        extra_targets = [{outcome, ep} for ep in (extra_predictors or [])]
        all_targets = [target] + extra_targets

        # Keep plots involving the main pair OR any extra selected pair, plus the network.
        eda = [
            p for p in eda
            if p.get("plot_type") == "bet_network"
            or any(t <= set(_eda_var_names(p)) for t in all_targets)
        ]

        # Ensure the exact pair BET copula is present for the primary pair.
        has_pair = any(
            p.get("plot_type") == "bet_interaction" and target == set(_eda_var_names(p))
            for p in eda
        )
        if not has_pair:
            pair_plot = _pair_bet_plot(df, outcome, predictor)
            if pair_plot:
                eda = [pair_plot] + eda

        # Ensure a BET copula exists for each extra predictor in the pool.
        # These are supplementary exploratory plots — label them clearly.
        for ep_target in extra_targets:
            has_ep = any(
                p.get("plot_type") == "bet_interaction" and ep_target == set(_eda_var_names(p))
                for p in eda
            )
            if not has_ep:
                ep_var = next(iter(ep_target - {outcome}), None)
                if ep_var:
                    ep_plot = _pair_bet_plot(df, outcome, ep_var)
                    if ep_plot:
                        ep_plot["title"] = "Supplementary: " + ep_plot.get("title", "")
                        eda.append(ep_plot)
            else:
                # Mark any existing EDA plot for this pair as supplementary too.
                for p in eda:
                    if (p.get("plot_type") == "bet_interaction"
                            and ep_target == set(_eda_var_names(p))
                            and not p.get("title", "").startswith("Supplementary")):
                        p["title"] = "Supplementary: " + p.get("title", "")

        # Insert predictor Q–Q plot after the existing test-result plots.
        existing = report_dict.get("plots", [])
        has_pred_qq = any(
            p.get("plot_type") == "qqplot" and predictor in p.get("title", "")
            for p in existing
        )
        if not has_pred_qq:
            pred_qq = _predictor_qq_plot(df, predictor)
            if pred_qq:
                pred_qq["plotly_json"] = plotspec_to_plotly(pred_qq)
                existing = list(existing) + [pred_qq]
                report_dict["plots"] = existing

    if eda:
        report_dict["plots"] = list(report_dict.get("plots", [])) + eda


def _degenerate_report(profile: dict[str, Any], design: dict[str, Any],
                       test_result: dict[str, Any], selection: Any,
                       hypothesis: str) -> dict[str, Any]:
    """Minimal report for the case where no valid test was selected (test_used == '—')."""
    caveats = [{"severity": "WARNING", "message": c,
                "recommendation": "Choose a group or predictor variable to form a test."}
               for c in (getattr(selection, "caveats", None) or [])]
    msg = ("No statistical test could be selected for this combination of variables. "
           "Pick a grouping variable (for a comparison) or a predictor (for an association).")
    report = {
        "data_profile": profile,
        "study_design": design,
        "test_result": test_result,
        "plain_language_summary": msg,
        "caveats": caveats,
        "plots": [],
        "methods_text": msg,
    }
    _attach_eda(report, profile)  # degenerate path: no variable filtering needed
    return report


def enrich_prose_with_llm(report: dict[str, Any],
                          outcome: str | None = None,
                          predictor: str | None = None,
                          extra_predictors: list[str] | None = None,
                          predictors: list[str] | None = None) -> dict[str, Any]:
    """Replace the deterministic report prose with a live LLM-generated version.

    Calls the configured LLM provider to rewrite `plain_language_summary` and
    `methods_text`. Falls back silently to the deterministic text on any error so
    the pipeline is never blocked by an LLM failure. No-ops in dry-run mode.
    """
    if DRY_RUN:
        return report

    tr = report.get("test_result", {})
    es = tr.get("effect_size", {})

    def _fmt(v: Any, decimals: int = 3) -> str:
        try:
            return f"{float(v):.{decimals}g}"
        except (TypeError, ValueError):
            return str(v)

    var_context = ""
    if outcome and predictors and len(predictors) >= 2:
        var_context = (
            f"Multiple regression model: outcome = {outcome}, "
            f"predictors jointly modelled = {', '.join(predictors)}. "
            "Interpret the effect size as overall model fit (R² or McFadden's R²), "
            "and per-predictor estimates as conditional (adjusted) effects."
        )
    elif outcome and predictor:
        var_context = f"Primary test pair: {outcome} (outcome) × {predictor} (predictor). "
        if extra_predictors:
            var_context += (
                f"The following variables were also selected by the user but were NOT part "
                f"of the formal statistical test — they appear as supplementary exploratory "
                f"plots only: {', '.join(extra_predictors)}. "
                "Do NOT imply these were formally tested or jointly modelled."
            )

    prompt = (
        f"A statistical test ({str(tr.get('test_used', '?')).replace('_', ' ')}) produced:\n"
        f"  p = {_fmt(tr.get('p_value'))},  "
        f"{es.get('measure_name', 'effect')} = {_fmt(es.get('value'))} "
        f"({es.get('interpretation', '')}, "
        f"95% CI [{_fmt(es.get('ci_lower'))}, {_fmt(es.get('ci_upper'))}]).\n"
        f"  Significant at α = 0.05: {'yes' if tr.get('is_significant') else 'no'}.\n"
        + (f"  {var_context}\n" if var_context else "") +
        f"\nExisting plain-language summary (deterministic):\n{report.get('plain_language_summary', '')}\n\n"
        f"Existing methods text (deterministic):\n{report.get('methods_text', '')}\n\n"
        "Rewrite both for a research paper. Return ONLY valid JSON with keys "
        '"plain" (2–3 sentences, non-statistician audience, lead with the key finding) '
        'and "methods" (3–5 sentences, passive voice, journal style). '
        "No markdown, no extra keys."
    )

    try:
        if LLM_PROVIDER == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=30.0)
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text
        else:
            if IS_AZURE_OPENAI:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=OPENAI_API_KEY,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_version=AZURE_OPENAI_API_VERSION,
                    timeout=30.0,
                )
            else:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=30.0)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600,
            )
            raw = resp.choices[0].message.content or ""

        import json as _json
        # Strip optional code-fence wrappers that some models emit.
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = _json.loads(clean)
        if parsed.get("plain"):
            report["plain_language_summary"] = parsed["plain"]
        if parsed.get("methods"):
            report["methods_text"] = parsed["methods"]
    except Exception:
        logger.exception("LLM prose enrichment failed — falling back to deterministic text")

    return report


def build_report(
    profile: dict[str, Any],
    design: dict[str, Any],
    test_result: dict[str, Any],
    selection: Any,
    df: pd.DataFrame,
    outcome: str,
    group: Optional[str],
    predictor: Optional[str],
    hypothesis: str,
    extra_predictors: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Assemble the report dict the SSE pipeline stores and streams."""
    try:
        result_model = TestResult.model_validate(test_result)
    except Exception:
        return _degenerate_report(profile, design, test_result, selection, hypothesis)

    try:
        design_model = StudyDesign.model_validate(design)
    except Exception:
        design_model = _default_design()
    profile_model = DataProfile.model_validate(profile)

    report = _build_report(profile_model, design_model, result_model, selection, df,
                           outcome, group, predictor, hypothesis)
    report_dict = report.model_dump(mode="json")
    _attach_eda(report_dict, profile, df=df, outcome=outcome, predictor=predictor,
                extra_predictors=extra_predictors)
    return report_dict
