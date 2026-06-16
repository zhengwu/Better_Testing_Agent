"""TestSelector — the deterministic test-selection decision tree (TECHNICAL_REPORT §6.2).

Purely deterministic: no LLM calls. Every distributional property is consumed as a graded
signal, never a binary significance gate. For an association of two continuous/ordinal
variables it also computes Pearson/Spearman and the real BET dependence test (p-value +
dependence region) so the selection is grounded in the data, not just a heuristic.

Pure standard library (plus the stdlib `hta.bet_screen` engine), so the playground and the
web backend can both import it with no third-party dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hta.bet_screen import maxbet, relationship_form
from hta.modules.profiler import _MISSING, EVENT_NAME_RE, _moments, _to_float, severity

if TYPE_CHECKING:
    from hta.modules.profiler import Column

LARGE_N = 30                      # CLT threshold (§6.1, proposed default)


def prefer_rank_based(outcome_type: str, n_min: int, nonnormality: str | None) -> bool:
    if outcome_type == "ORDINAL":
        return True
    if n_min >= LARGE_N:
        return False
    return nonnormality == "STRONG"


def _find_event_col(cols: dict[str, "Column"], exclude: set[str]) -> str | None:
    """A 0/1 event-/censoring-indicator column (the companion a survival analysis needs)."""
    for name, col in cols.items():
        if name in exclude or col.var_type != "BINARY":
            continue
        cats = {c.replace(".0", "") for c in col.categories}
        if EVENT_NAME_RE.search(name) and cats <= {"0", "1"}:
            return name
    return None


_WITHIN_RE = re.compile(r"\b(pair|paired|before|after|repeated|within|pre[- ]?post|matched)\b",
                        re.IGNORECASE)


@dataclass
class Selection:
    test: str
    rationale: str
    caveats: list[str] = field(default_factory=list)
    computed: dict[str, str] = field(default_factory=dict)
    region: list[tuple[int, int]] = field(default_factory=list)
    grid_size: int = 0


def _group_sizes(group_raw: list[str], rows_present: list[bool]) -> list[int]:
    counts: dict[str, int] = {}
    for v, ok in zip(group_raw, rows_present):
        if ok:
            counts[v] = counts.get(v, 0) + 1
    return list(counts.values())


_SEV_ORDER = {"NONE": 0, "MILD": 1, "STRONG": 2}


def _group_severity(o_raw: list[str], g_raw: list[str]) -> str:
    """Worst (most severe) per-group normality grade across groups with n ≥ 8."""
    groups: dict[str, list[float]] = {}
    for ov, gv in zip(o_raw, g_raw):
        f = _to_float(ov)
        if f is not None and gv.strip().lower() not in _MISSING:
            groups.setdefault(gv, []).append(f)
    worst = "NONE"
    for vals in groups.values():
        if len(vals) >= 8:
            _, _, sk, ku = _moments(vals)
            sev = severity(sk, ku)
            if _SEV_ORDER[sev] > _SEV_ORDER[worst]:
                worst = sev
    return worst


def select(
    cols: dict[str, "Column"],
    outcome: str,
    group: str | None,
    predictor: str | None,
    prompt: str,
    raw: dict[str, list[str]],
    predictors: list[str] | None = None,
) -> Selection:
    oc = cols[outcome]
    measurement = "WITHIN" if _WITHIN_RE.search(prompt) else "BETWEEN"

    # Multi-predictor regression routing: when 2+ numeric predictors are supplied and
    # there is no grouping variable, route to the appropriate regression family.
    # COUNT outcomes still go to Poisson/NegBin (below), but the formula will use all predictors.
    if predictors and len(predictors) >= 2 and not group:
        n_pred = len(predictors)
        pred_list = ", ".join(predictors)
        if oc.var_type in ("CONTINUOUS", "ORDINAL"):
            return Selection(
                "LINEAR_REGRESSION",
                f"{n_pred} numeric predictors ({pred_list}) with a continuous outcome "
                f"→ multiple linear regression (OLS). R² measures overall model fit.",
            )
        if oc.var_type in ("BINARY", "CATEGORICAL"):
            return Selection(
                "LOGISTIC_REGRESSION",
                f"{n_pred} numeric predictors ({pred_list}) with a binary/categorical outcome "
                f"→ logistic regression (MLE). McFadden's R² and per-predictor ORs reported.",
            )

    # Healthcare dispatch (subset): COUNT outcome → Poisson / NegBin.
    if oc.var_type == "COUNT" and oc.numeric:
        mean = sum(oc.numeric) / len(oc.numeric)
        var = sum((x - mean) ** 2 for x in oc.numeric) / max(1, len(oc.numeric) - 1)
        if var > 1.3 * mean:
            return Selection("NEGATIVE_BINOMIAL_REGRESSION",
                             f"Count outcome, overdispersed (var {var:.1f} > mean {mean:.1f}).")
        return Selection("POISSON_REGRESSION",
                         f"Count outcome, variance ≈ mean ({var:.1f} vs {mean:.1f}).")

    # Survival dispatch: a time-to-event outcome + a 0/1 event indicator → log-rank / Cox.
    if oc.var_type == "TIME_TO_EVENT":
        event = _find_event_col(cols, {outcome, group or "", predictor or ""})
        if event:
            if predictor and predictor in cols:
                return Selection("COX_REGRESSION",
                                 f"Time-to-event outcome with covariate {predictor} → Cox "
                                 f"proportional hazards (HR); event indicator '{event}'.")
            if group and group in cols:
                return Selection("LOG_RANK",
                                 f"Time-to-event outcome compared across {group} → log-rank "
                                 f"with Kaplan–Meier curves; event indicator '{event}'.")
            return Selection("LOG_RANK",
                             f"Time-to-event outcome → Kaplan–Meier / log-rank (event '{event}').")
        # No event indicator found → fall back to treating the duration as a continuous outcome.

    # ── grouped comparison ────────────────────────────────────────────────────
    if group and group in cols:
        sizes = _group_sizes(raw[group], [True] * len(raw[group]))
        # recompute sizes restricted to rows where outcome present
        sizes = _group_sizes(
            raw[group],
            [v.strip().lower() not in _MISSING for v in raw[outcome]],
        )
        n_groups = len(sizes)
        n_min = min(sizes) if sizes else 0

        if oc.var_type in ("CONTINUOUS", "ORDINAL", "TIME_TO_EVENT"):
            # Per-group (not pooled) normality severity — a pooled bimodal column would
            # look non-normal precisely when the groups differ, which is the wrong signal.
            grp_nonnorm = _group_severity(raw[outcome], raw[group])
            rank = prefer_rank_based(oc.var_type, n_min, grp_nonnorm)
            if n_groups == 2:
                if measurement == "WITHIN":
                    test = "WILCOXON_SIGNED_RANK" if rank else "PAIRED_T"
                else:
                    test = "MANN_WHITNEY_U" if rank else "WELCH_T"
            elif n_groups >= 3:
                test = "KRUSKAL_WALLIS" if rank else "WELCH_ANOVA"
            else:
                test = "WELCH_T"
            route = "rank-based" if rank else "parametric (Welch default, no variance pretest)"
            why = (f"{oc.var_type.lower()} outcome, {n_groups} group(s), {measurement.lower()}; "
                   f"worst-group normality {grp_nonnorm}, min group n={n_min} -> {route}.")
            sel = Selection(test, why)
        else:  # BINARY / CATEGORICAL outcome × group → contingency
            sel = _categorical(outcome, group, raw, measurement)

        if n_groups >= 3:
            sel.caveats.append("3+ groups: report post-hoc (Holm-adjusted) and family-wise error.")
        if n_min and n_min < 20:
            sel.caveats.append(f"Small smallest group (n={n_min}): interpret with caution.")
        return sel

    # ── association of two continuous/ordinal variables → BET path ────────────
    if (predictor and predictor in cols
            and oc.var_type in ("CONTINUOUS", "ORDINAL", "TIME_TO_EVENT")):
        pc = cols[predictor]
        if pc.var_type in ("CONTINUOUS", "ORDINAL", "COUNT"):
            return _association(oc, raw[outcome], raw[predictor])

    # Diagnostic accuracy: a binary reference standard discriminated by a continuous index.
    if (oc.var_type == "BINARY" and predictor and predictor in cols
            and cols[predictor].var_type in ("CONTINUOUS", "ORDINAL", "COUNT")):
        return Selection("ROC_AUC",
                         f"Binary outcome discriminated by the continuous index {predictor} "
                         "→ ROC / AUC.")

    if oc.var_type in ("BINARY", "CATEGORICAL"):
        return Selection("CHI_SQUARED",
                         "Categorical outcome with no second variable chosen — pick a group "
                         "or predictor column to form a comparison/association.")
    return Selection(
        "—",
        "Pick a group column (for a comparison) or a predictor column (for an association) "
        "to get a recommended test.",
    )


def _categorical(outcome: str, group: str, raw: dict[str, list[str]],
                 measurement: str) -> Selection:
    o_raw, g_raw = raw[outcome], raw[group]
    table: dict[tuple[str, str], int] = {}
    rkeys, ckeys = set(), set()
    for ov, gv in zip(o_raw, g_raw):
        if ov.strip().lower() in _MISSING or gv.strip().lower() in _MISSING:
            continue
        table[(ov, gv)] = table.get((ov, gv), 0) + 1
        rkeys.add(ov)
        ckeys.add(gv)
    r, c = len(rkeys), len(ckeys)
    n = sum(table.values())
    rsum = {rk: sum(v for (ov, _), v in table.items() if ov == rk) for rk in rkeys}
    csum = {ck: sum(v for (_, gv), v in table.items() if gv == ck) for ck in ckeys}
    min_expected = min((rsum[rk] * csum[ck] / n) for rk in rkeys for ck in ckeys) if n else 0
    shape = "2×2" if r == 2 and c == 2 else f"{r}×{c}"
    if measurement == "WITHIN" and r == 2 and c == 2:
        return Selection("MCNEMAR", "Paired binary outcome (2×2 within-subjects).")
    if min_expected >= 5:
        return Selection("CHI_SQUARED",
                         f"Categorical {shape} table, all expected counts ≥ 5 "
                         f"(min {min_expected:.1f}).")
    return Selection("FISHER_EXACT",
                     f"Categorical {shape} table, smallest expected count {min_expected:.1f} < 5.")


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(abs(z) / math.sqrt(2))


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else 0.0


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _fisher_z_p(r: float, n: int) -> float:
    if n <= 3 or abs(r) >= 1:
        return float("nan")
    z = math.atanh(r) * math.sqrt(n - 3)
    return 2 * _norm_sf(z)


def _association(oc: "Column", o_raw: list[str], p_raw: list[str]) -> Selection:
    """Row-aligned continuous/ordinal association: compute Pearson, Spearman, BET."""
    xy = []
    for a, b in zip(o_raw, p_raw):
        fa, fb = _to_float(a), _to_float(b)
        if fa is not None and fb is not None:
            xy.append((fa, fb))
    x = [a for a, _ in xy]
    y = [b for _, b in xy]
    n = len(x)
    if n < 8:
        return Selection("—", f"Only {n} complete numeric pairs — too few to analyse this pair.")
    bet = maxbet(x, y, seed=0)
    relationship = relationship_form(bet.form) if bet.significant else "linear"
    rank = prefer_rank_based(oc.var_type, n, oc.nonnormality)

    if relationship == "nonlinear":
        test = "MAXBET"
        why = (f"BET finds a nonlinear dependence ({bet.form.lower()}) — Pearson/Spearman would "
               f"miss or mislabel it. Use BET / a nonlinear model.")
    elif relationship == "monotone" or oc.var_type == "ORDINAL":
        test = "SPEARMAN_CORRELATION"
        why = "Monotone (or ordinal) association → Spearman's rank correlation."
    elif rank:
        test = "SPEARMAN_CORRELATION"
        why = "Small N with strong non-normality -> Spearman (rank-based)."
    else:
        test = "PEARSON_CORRELATION"
        why = "Approximately linear association at adequate N → Pearson's correlation."

    pr = _pearson(x, y)
    sp = _pearson(_rank(x), _rank(y))
    sel = Selection(test, why)
    sel.computed = {
        "Pearson r": f"{pr:+.3f} (p={_fisher_z_p(pr, n):.3g})",
        "Spearman ρ": f"{sp:+.3f} (p={_fisher_z_p(sp, n):.3g})",
        "BET": (f"{'REJECT independence' if bet.significant else 'no dependence'} "
                f"(p={bet.p_value:.3g}, z={bet.bet_z:.2f}, form={bet.form}, "
                f"interaction {bet.bid})"),
    }
    sel.region = bet.positive_region
    sel.grid_size = bet.grid_size
    if bet.nonlinear_only:
        sel.caveats.append(
            "BET-significant but Pearson and Spearman are both near zero: the dependence is "
            "nonlinear-only. Consider whether latent subgroups/subtypes drive it (run a "
            "within-subgroup analysis).")
    if n < 20:
        sel.caveats.append(f"Small sample (n={n}): correlation/BET estimates are unstable.")
    return sel


__all__ = ["LARGE_N", "Selection", "prefer_rank_based", "select"]
