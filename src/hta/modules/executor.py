"""TestExecutor — runs the selected statistical test and returns a `TestResult`.

`execute(test, df, outcome, group, predictor, design, selection) -> TestResult` is the
single entry point. Implementations are pure scipy + statsmodels; bootstrap CIs use stdlib
`random` (seed 42, n=1000) so they are deterministic and numpy-free.

Coverage in v0.1.0: the two-group t-tests, Mann–Whitney, paired t / Wilcoxon, one-way and
Welch ANOVA, Kruskal–Wallis, χ² / Fisher / McNemar, Pearson / Spearman, the pure-Python
MaxBET, and Poisson / negative-binomial regression. Tests that are in the enum but not yet
wired (survival, ROC/AUC, the reserved regressions) return an UNTESTABLE result rather than
raising, so the pipeline always produces a report; only an unrecognised test name raises.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional

import pandas as pd
from scipy import stats

from hta.models.test import StatisticalTest, TestResult
from hta.modules.causal import usable_adjustment_covariates
from hta.modules.profiler import EVENT_NAME_RE

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# Cohen (1988) interpretation thresholds, keyed by effect-size family.
_THRESH = {
    "d": (0.2, 0.5, 0.8),          # Cohen's d / d_z
    "r": (0.1, 0.3, 0.5),          # correlation / rank-biserial
    "v": (0.1, 0.3, 0.5),          # Cramér's V
    "eta2": (0.01, 0.06, 0.14),    # eta-squared
}

# |log(ratio)| thresholds for ratio effect measures (OR, HR, IRR).
# Symmetric for protective/harmful effects. Correspond roughly to:
#   small  → ratio in (0.80, 1.25]   ≤25 % departure from null
#   medium → ratio in (0.67, 1.50]   ≤50 % departure
#   large  → ratio < 0.50 or > 2.00  >100 % departure
_RATIO_THRESH = (0.22, 0.41, 0.69)


# ── small numeric helpers (stdlib) ────────────────────────────────────────────

def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _var(v: list[float]) -> float:
    n = len(v)
    if n < 2:
        return 0.0
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / (n - 1)


def _sd(v: list[float]) -> float:
    return math.sqrt(_var(v))


def _interpret_ratio(ratio: float) -> str:
    """Interpret ratio effect measures (OR, HR, IRR) by |log(ratio)| distance from null."""
    if ratio <= 0 or not math.isfinite(ratio):
        return "negligible"
    small, medium, large = _RATIO_THRESH
    d = abs(math.log(ratio))
    if d >= large:
        return "large"
    if d >= medium:
        return "medium"
    if d >= small:
        return "small"
    return "negligible"


def _interpret(value: float, family: str) -> str:
    if family == "ratio":
        return _interpret_ratio(value)
    small, medium, large = _THRESH[family]
    a = abs(value)
    if a >= large:
        return "large"
    if a >= medium:
        return "medium"
    if a >= small:
        return "small"
    return "negligible"


def _finite(x: float, default: float = 0.0) -> float:
    return float(x) if x is not None and math.isfinite(float(x)) else default


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _boot_two_group(a: list[float], b: list[float],
                    stat: Callable[[list[float], list[float]], float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        ra = rng.choices(a, k=len(a))
        rb = rng.choices(b, k=len(b))
        try:
            v = stat(ra, rb)
            if v is not None and math.isfinite(v):
                out.append(v)
        except Exception:
            pass
    out.sort()
    return (_pct(out, 0.025), _pct(out, 0.975)) if out else (float("nan"), float("nan"))


def _boot_paired(x: list[float], y: list[float],
                 stat: Callable[[list[float], list[float]], float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    m = len(x)
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(m) for _ in range(m)]
        rx = [x[i] for i in idx]
        ry = [y[i] for i in idx]
        try:
            v = stat(rx, ry)
            if v is not None and math.isfinite(v):
                out.append(v)
        except Exception:
            pass
    out.sort()
    return (_pct(out, 0.025), _pct(out, 0.975)) if out else (float("nan"), float("nan"))


def _cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp2 = ((na - 1) * _var(a) + (nb - 1) * _var(b)) / (na + nb - 2)
    sp = math.sqrt(sp2)
    return (_mean(a) - _mean(b)) / sp if sp > 0 else 0.0


def _dz_stat(x: list[float], y: list[float]) -> float:
    dd = [p - q for p, q in zip(x, y)]
    s = _sd(dd)
    return _mean(dd) / s if s > 0 else 0.0


def _eta_squared(groups: list[list[float]]) -> float:
    allv = [x for g in groups for x in g]
    if len(allv) < 2:
        return 0.0
    gm = _mean(allv)
    ss_b = sum(len(g) * (_mean(g) - gm) ** 2 for g in groups)
    ss_t = sum((x - gm) ** 2 for x in allv)
    return ss_b / ss_t if ss_t > 0 else 0.0


# ── sensitivity power (TECHNICAL_REPORT §5.5) ─────────────────────────────────

def _sensitivity_mde_d(
    n1: int, n2: int, alpha: float = 0.05, power: float = 0.80,
) -> Optional[float]:
    """Minimum detectable Cohen's d for a two-group comparison at the observed Ns
    (α=0.05, power=0.80) — *sensitivity* power, never observed/post-hoc power."""
    try:
        from statsmodels.stats.power import TTestIndPower
    except Exception:
        return None
    if n1 < 2 or n2 < 2:
        return None
    try:
        ratio = n2 / n1
        d = TTestIndPower().solve_power(
            effect_size=None, nobs1=n1, alpha=alpha, power=power, ratio=ratio,
            alternative="two-sided")
        return float(d) if d is not None and math.isfinite(float(d)) else None
    except Exception:
        return None


def _sensitivity_note(n1: int, n2: int) -> Optional[str]:
    mde = _sensitivity_mde_d(n1, n2)
    if mde is None:
        return None
    return (f"Sensitivity: minimum detectable Cohen's d = {mde:.2f} at "
            f"N={n1 + n2}, α=0.05, power=0.80.")


# ── data extraction ───────────────────────────────────────────────────────────

def _numeric(df: pd.DataFrame, col: str) -> list[float]:
    return [float(v) for v in pd.to_numeric(df[col], errors="coerce").dropna().tolist()]


def _group_arrays(
    df: pd.DataFrame, outcome: str, group: str,
) -> tuple[list[str], list[list[float]]]:
    sub = df[[outcome, group]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()
    labels: list[str] = []
    arrays: list[list[float]] = []
    for label, g in sub.groupby(group, sort=True):
        vals = [float(v) for v in g[outcome].tolist()]
        if vals:
            labels.append(str(label))
            arrays.append(vals)
    return labels, arrays


def _xy(df: pd.DataFrame, outcome: str, predictor: str) -> tuple[list[float], list[float]]:
    sub = df[[predictor, outcome]].copy()
    sub[predictor] = pd.to_numeric(sub[predictor], errors="coerce")
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()
    return ([float(v) for v in sub[predictor].tolist()],
            [float(v) for v in sub[outcome].tolist()])


# ── assembly helpers ──────────────────────────────────────────────────────────

def _check(name: str, status: str, note: str, *,
           test_used: Optional[str] = None, statistic: Optional[float] = None,
           p_value: Optional[float] = None) -> dict[str, Any]:
    d: dict[str, Any] = {"assumption_name": name, "status": status, "note": note}
    if test_used is not None:
        d["test_used"] = test_used
    if statistic is not None:
        d["statistic"] = round(float(statistic), 4)
    if p_value is not None:
        d["p_value"] = round(float(p_value), 4)
    return d


def _result(test_name: str, statistic: float, p_value: float, dof: Optional[float],
            measure: str, value: float, family: str, ci_lo: float, ci_hi: float,
            checks: list[dict[str, Any]], primary_ci: tuple[float, float],
            notes: list[str], power: Optional[float] = None) -> dict[str, Any]:
    value = _finite(value)
    ci_lo = _finite(ci_lo, value)
    ci_hi = _finite(ci_hi, value)
    return {
        "test_used": test_name,
        "statistic": round(_finite(statistic), 4),
        "p_value": _finite(p_value, 1.0),
        "degrees_of_freedom": (round(float(dof), 2) if dof is not None else None),
        "effect_size": {
            "measure_name": measure,
            "value": round(value, 4),
            "interpretation": _interpret(value, family),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        },
        "assumption_checks": checks,
        "confidence_interval": [round(_finite(primary_ci[0], value), 4),
                                round(_finite(primary_ci[1], value), 4)],
        "is_significant": bool(_finite(p_value, 1.0) < 0.05),
        "power": power,
        "notes": notes,
    }


def _not_implemented(test_name: str) -> dict[str, Any]:
    return _result(
        test_name, 0.0, 1.0, None, "—", 0.0, "r", 0.0, 0.0,
        [_check("Execution", "UNTESTABLE", f"{test_name} is not yet implemented.")],
        (0.0, 0.0),
        [f"{test_name} not yet implemented — use the dry-run stub for a demonstration."],
    )


def _normality_check(groups: list[list[float]]) -> dict[str, Any]:
    """Shapiro-Wilk per group; MET if all groups look normal, else VIOLATED."""
    worst_p = 1.0
    testable = False
    for g in groups:
        if 3 <= len(g) <= 5000:
            testable = True
            try:
                _, p = stats.shapiro(g)
                worst_p = min(worst_p, float(p))
            except Exception:
                pass
    if not testable:
        return _check("Normality", "UNTESTABLE", "Too few observations to test normality.")
    status = "MET" if worst_p > 0.05 else "VIOLATED"
    note = ("Each group is approximately normal (Shapiro-Wilk min p > 0.05)."
            if status == "MET" else
            "At least one group departs from normality (Shapiro-Wilk min p ≤ 0.05); "
            "the Welch/parametric result is reasonably robust at adequate N, but a "
            "rank-based test may be preferable.")
    return _check("Normality", status, note, test_used="Shapiro-Wilk", p_value=worst_p)


def _min_n_check(groups: list[list[float]], floor: int = 5) -> dict[str, Any]:
    smallest = min((len(g) for g in groups), default=0)
    status = "MET" if smallest >= floor else "VIOLATED"
    return _check("Minimum sample size", status,
                  f"Smallest group n = {smallest} (need ≥ {floor}).")


# ── post-hoc localisation (§6.3) ──────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _posthoc_note(sub: pd.DataFrame, outcome: str, group: str, kind: str) -> Optional[str]:
    """Pairwise post-hoc comparisons (§6.3): Games–Howell (Welch ANOVA), Tukey HSD (one-way
    ANOVA), or Dunn + Holm (Kruskal–Wallis). Returns a compact note, or None if the backing
    library is unavailable or the comparison cannot be computed."""
    try:
        pairs: list[tuple[str, str, float]] = []
        if kind == "dunn":
            import scikit_posthocs as sp
            m = sp.posthoc_dunn(sub, val_col=outcome, group_col=group, p_adjust="holm")
            cols = [str(col) for col in m.columns]
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((cols[i], cols[j], float(m.iloc[i, j])))
            label = "Dunn (Holm)"
        else:
            import pingouin as pg
            if kind == "gameshowell":
                res = pg.pairwise_gameshowell(data=sub, dv=outcome, between=group)
                pcol, label = "pval", "Games–Howell"
            else:
                res = pg.pairwise_tukey(data=sub, dv=outcome, between=group)
                pcol, label = "p_tukey", "Tukey HSD"
            pairs = [(str(row["A"]), str(row["B"]), float(row[pcol]))
                     for _, row in res.iterrows()]
        if not pairs:
            return None
        body = "; ".join(f"{a}–{b} p={_fmt_p(p)}" for a, b, p in pairs)
        return f"{label} post-hoc: {body}."
    except Exception:
        return None


def _welch_anova(sub: pd.DataFrame, outcome: str, group: str,
                 groups: list[list[float]]) -> tuple[float, float, Optional[float]]:
    """Welch's heteroscedastic one-way ANOVA via pingouin; falls back to a closed-form
    computation if pingouin is unavailable. Never Alexander–Govern (a different test)."""
    try:
        import pingouin as pg
        row = pg.welch_anova(data=sub, dv=outcome, between=group).iloc[0]
        return float(row["F"]), float(row["p_unc"]), float(row["ddof1"])
    except Exception:
        return _welch_anova_closed_form(groups)


def _welch_anova_closed_form(groups: list[list[float]]) -> tuple[float, float, Optional[float]]:
    k = len(groups)
    weights = [len(g) / _var(g) for g in groups if len(g) > 1 and _var(g) > 0]
    if k < 2 or len(weights) < k:                 # degenerate variance — last-resort fallback
        res = stats.f_oneway(*groups)
        return float(res.statistic), float(res.pvalue), float(k - 1)
    means = [_mean(g) for g in groups]
    sw = sum(weights)
    gbar = sum(w * m for w, m in zip(weights, means)) / sw
    num = sum(w * (m - gbar) ** 2 for w, m in zip(weights, means)) / (k - 1)
    tmp = sum((1 - w / sw) ** 2 / (len(g) - 1) for w, g in zip(weights, groups))
    f = num / (1 + 2 * (k - 2) / (k * k - 1) * tmp)
    df2 = (k * k - 1) / (3 * tmp) if tmp > 0 else float("inf")
    return f, float(stats.f.sf(f, k - 1, df2)), float(k - 1)


# ── contingency effect-size CIs and the R×C exact test (§6.4) ──────────────────

def _codes(values: list[str]) -> tuple[list[int], int]:
    cats = sorted(set(values))
    idx = {c: i for i, c in enumerate(cats)}
    return [idx[v] for v in values], len(cats)


def _table_chi2(tab: list[list[int]], n_rows: int, n_cols: int) -> tuple[float, float]:
    """Pearson χ² and grand total for a count table, skipping zero-expected cells."""
    tot = float(sum(sum(row) for row in tab))
    if tot <= 0:
        return 0.0, 0.0
    rowt = [sum(row) for row in tab]
    colt = [sum(tab[i][j] for i in range(n_rows)) for j in range(n_cols)]
    chi2 = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            e = rowt[i] * colt[j] / tot
            if e > 0:
                chi2 += (tab[i][j] - e) ** 2 / e
    return chi2, tot


def _bootstrap_cramers_v(o_codes: list[int], g_codes: list[int],
                         n_rows: int, n_cols: int) -> tuple[float, float]:
    """Percentile bootstrap CI for Cramér's V (resample the paired observations)."""
    n_dim = min(n_rows - 1, n_cols - 1)
    if n_dim < 1:
        return (0.0, 0.0)
    rng = random.Random(BOOTSTRAP_SEED)
    pairs = list(zip(o_codes, g_codes))
    n = len(pairs)
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        tab = [[0] * n_cols for _ in range(n_rows)]
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            tab[a][b] += 1
        chi2, tot = _table_chi2(tab, n_rows, n_cols)
        if tot > 0:
            v = math.sqrt(chi2 / (tot * n_dim))
            if math.isfinite(v):
                out.append(v)
    out.sort()
    return (_pct(out, 0.025), _pct(out, 0.975)) if out else (0.0, 0.0)


def _bootstrap_kruskal_eta(groups: list[list[float]]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    k = len(groups)
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        rs = [rng.choices(g, k=len(g)) for g in groups]
        try:
            h = float(stats.kruskal(*rs).statistic)
            n = sum(len(g) for g in rs)
            e = max(0.0, (h - k + 1) / (n - k)) if n > k else 0.0
            if math.isfinite(e):
                out.append(e)
        except Exception:
            pass
    out.sort()
    return (_pct(out, 0.025), _pct(out, 0.975)) if out else (0.0, 0.0)


def _or_phi_2x2(table: Any, chi2: float, n: int) -> tuple[float, float]:
    a, b = float(table.iloc[0, 0]), float(table.iloc[0, 1])
    c, d = float(table.iloc[1, 0]), float(table.iloc[1, 1])
    odds = (a * d) / (b * c) if b > 0 and c > 0 else float("nan")
    phi = math.sqrt(chi2 / n) if n else 0.0
    return _finite(odds, 1.0), phi


def _rxc_fisher_perm(o_codes: list[int], g_codes: list[int],
                     n_rows: int, n_cols: int, n_perm: int = 2000) -> tuple[float, int]:
    """Freeman–Halton simulated-exact p-value for an R×C table: permute the group labels
    (which fixes both margins) and compare the χ² statistic. Returns (p_value, n_perm)."""
    def chi2_of(gc: list[int]) -> float:
        tab = [[0] * n_cols for _ in range(n_rows)]
        for a, b in zip(o_codes, gc):
            tab[a][b] += 1
        return _table_chi2(tab, n_rows, n_cols)[0]

    obs = chi2_of(g_codes)
    rng = random.Random(BOOTSTRAP_SEED)
    perm = list(g_codes)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        if chi2_of(perm) >= obs - 1e-9:
            count += 1
    return (count + 1) / (n_perm + 1), n_perm


# ── per-test executors ────────────────────────────────────────────────────────

def _two_sample_t(df: pd.DataFrame, outcome: str, group: str, equal_var: bool) -> dict[str, Any]:
    labels, groups = _group_arrays(df, outcome, group)
    if len(groups) < 2:
        return _not_implemented("WELCH_T")
    a, b = groups[0], groups[1]
    res = stats.ttest_ind(a, b, equal_var=equal_var)
    d = _cohens_d(a, b)
    ci_d = _boot_two_group(a, b, _cohens_d)
    ci_md = _boot_two_group(a, b, lambda x, y: _mean(x) - _mean(y))
    name = "INDEPENDENT_T" if equal_var else "WELCH_T"
    checks = [
        _normality_check(groups),
        _min_n_check(groups),
        _check("Independence of observations", "UNTESTABLE",
               "Between-subjects design assumed; independence cannot be verified from data."),
    ]
    if equal_var:
        try:
            lev_stat, lev_p = stats.levene(a, b)
            checks.append(_check("Equal variances", "MET" if lev_p > 0.05 else "VIOLATED",
                                 "Levene's test for homogeneity of variance.",
                                 test_used="Levene", statistic=lev_stat, p_value=lev_p))
        except Exception:
            pass
    notes = [f"Mean difference = {_mean(a) - _mean(b):.3f} ({labels[0]} − {labels[1]})."]
    sens = _sensitivity_note(len(a), len(b))
    if sens:
        notes.append(sens)
    return _result(name, res.statistic, res.pvalue, getattr(res, "df", None),
                   "Cohen's d", d, "d", ci_d[0], ci_d[1], checks, ci_md, notes)


def _mann_whitney(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    labels, groups = _group_arrays(df, outcome, group)
    if len(groups) < 2:
        return _not_implemented("MANN_WHITNEY_U")
    a, b = groups[0], groups[1]

    def rb(x: list[float], y: list[float]) -> float:
        u = float(stats.mannwhitneyu(x, y, alternative="two-sided").statistic)
        return 1.0 - 2.0 * u / (len(x) * len(y))

    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = rb(a, b)
    ci = _boot_two_group(a, b, rb)
    checks = [_min_n_check(groups),
              _check("Independence of observations", "UNTESTABLE",
                     "Between-subjects design assumed.")]
    return _result("MANN_WHITNEY_U", res.statistic, res.pvalue, None,
                   "rank-biserial r", r, "r", ci[0], ci[1], checks, ci,
                   ["Distribution-free comparison of two independent groups."])


def _paired_t(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    labels, groups = _group_arrays(df, outcome, group)
    if len(groups) < 2:
        return _not_implemented("PAIRED_T")
    m = min(len(groups[0]), len(groups[1]))
    a, b = groups[0][:m], groups[1][:m]
    res = stats.ttest_rel(a, b)
    diffs = [x - y for x, y in zip(a, b)]
    dz = _dz_stat(a, b)
    ci_dz = _boot_paired(a, b, _dz_stat)
    ci_md = _boot_paired(a, b, lambda x, y: _mean([p - q for p, q in zip(x, y)]))
    checks = [_normality_check([diffs]), _min_n_check(groups, 5),
              _check("Paired structure", "UNTESTABLE",
                     f"Pairs aligned by row order; {m} pairs used.")]
    return _result("PAIRED_T", res.statistic, res.pvalue, float(m - 1),
                   "Cohen's d_z", dz, "d", ci_dz[0], ci_dz[1], checks, ci_md,
                   ["Within-subjects comparison of paired observations."])


def _wilcoxon(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    labels, groups = _group_arrays(df, outcome, group)
    if len(groups) < 2:
        return _not_implemented("WILCOXON_SIGNED_RANK")
    m = min(len(groups[0]), len(groups[1]))
    a, b = groups[0][:m], groups[1][:m]
    try:
        res = stats.wilcoxon(a, b)
    except ValueError:
        return _not_implemented("WILCOXON_SIGNED_RANK")
    n = m
    r = 1.0 - 2.0 * float(res.statistic) / (n * (n + 1) / 2.0) if n else 0.0
    checks = [_min_n_check(groups, 5),
              _check("Paired structure", "UNTESTABLE", f"{m} pairs (row-aligned).")]
    return _result("WILCOXON_SIGNED_RANK", res.statistic, res.pvalue, None,
                   "rank-biserial r", r, "r", -abs(r), abs(r), checks, (-abs(r), abs(r)),
                   ["Distribution-free within-subjects comparison."])


def _anova(df: pd.DataFrame, outcome: str, group: str, welch: bool) -> dict[str, Any]:
    labels, groups = _group_arrays(df, outcome, group)
    if len(groups) < 2:
        return _not_implemented("WELCH_ANOVA" if welch else "ONE_WAY_ANOVA")
    sub = df[[outcome, group]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()
    if welch:
        statistic, p, dof = _welch_anova(sub, outcome, group, groups)  # true Welch, not A–G
        name, posthoc = "WELCH_ANOVA", _posthoc_note(sub, outcome, group, "gameshowell")
    else:
        res = stats.f_oneway(*groups)
        statistic, p, dof = float(res.statistic), float(res.pvalue), float(len(groups) - 1)
        name, posthoc = "ONE_WAY_ANOVA", _posthoc_note(sub, outcome, group, "tukey")
    eta2 = _eta_squared(groups)
    ci = _bootstrap_eta(groups)
    checks = [
        _normality_check(groups),
        _min_n_check(groups),
        _check("Independence of observations", "UNTESTABLE", "Assumed by design."),
    ]
    notes = [f"{len(groups)} groups compared."]
    notes.append(posthoc if posthoc
                 else "Post-hoc localisation unavailable (needs pingouin/scikit-posthocs).")
    return _result(name, statistic, p, dof, "eta-squared", eta2, "eta2",
                   ci[0], ci[1], checks, ci, notes)


def _bootstrap_eta(groups: list[list[float]]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        rs = [rng.choices(g, k=len(g)) for g in groups]
        try:
            v = _eta_squared(rs)
            if math.isfinite(v):
                out.append(v)
        except Exception:
            pass
    out.sort()
    return (_pct(out, 0.025), _pct(out, 0.975)) if out else (0.0, 0.0)


def _kruskal(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    labels, groups = _group_arrays(df, outcome, group)
    if len(groups) < 2:
        return _not_implemented("KRUSKAL_WALLIS")
    res = stats.kruskal(*groups)
    k = len(groups)
    n = sum(len(g) for g in groups)
    eta2 = max(0.0, (float(res.statistic) - k + 1) / (n - k)) if n > k else 0.0
    ci = _bootstrap_kruskal_eta(groups)
    sub = df[[outcome, group]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()
    posthoc = _posthoc_note(sub, outcome, group, "dunn")
    checks = [
        _min_n_check(groups),
        _check("Independence of observations", "UNTESTABLE", "Assumed by design."),
    ]
    notes = ["Distribution-free omnibus test."]
    if posthoc:
        notes.append(posthoc)
    return _result("KRUSKAL_WALLIS", res.statistic, res.pvalue, float(k - 1),
                   "eta-squared (rank)", eta2, "eta2", ci[0], ci[1], checks, ci, notes)


def _chi_squared(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    table = pd.crosstab(df[outcome], df[group])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _not_implemented("CHI_SQUARED")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    r, c = table.shape
    v = math.sqrt(chi2 / (n * min(r - 1, c - 1))) if n and min(r - 1, c - 1) else 0.0
    min_exp = float(expected.min())
    sub = df[[outcome, group]].dropna()
    o_codes, n_rows = _codes(sub[outcome].astype(str).tolist())
    g_codes, n_cols = _codes(sub[group].astype(str).tolist())
    ci = _bootstrap_cramers_v(o_codes, g_codes, n_rows, n_cols)
    checks = [
        _check("Expected cell counts ≥ 5", "MET" if min_exp >= 5 else "VIOLATED",
               f"Smallest expected count = {min_exp:.2f}. "
               + ("" if min_exp >= 5 else "Fisher's exact is preferable.")),
        _check("Independence of observations", "UNTESTABLE", "Assumed by design."),
    ]
    notes = [f"{r}×{c} contingency table, N = {n}."]
    if (r, c) == (2, 2):
        odds, phi = _or_phi_2x2(table, float(chi2), n)
        notes.append(f"2×2: odds ratio = {odds:.3f}, φ = {phi:.3f}.")
    return _result("CHI_SQUARED", chi2, p, float(dof), "Cramér's V", v, "v",
                   ci[0], ci[1], checks, ci, notes)


def _fisher(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    table = pd.crosstab(df[outcome], df[group])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _not_implemented("FISHER_EXACT")
    if table.shape == (2, 2):
        odds, p = stats.fisher_exact(table.values)
        checks = [
            _check("2×2 table", "MET", "Outcome and group are both binary."),
            _check("Small expected counts", "UNTESTABLE",
                   "Fisher's exact is valid for any expected counts."),
        ]
        return _result("FISHER_EXACT", odds, p, None, "odds ratio", odds, "ratio",
                       odds, odds, checks, (odds, odds),
                       ["Exact test for a 2×2 contingency table."])
    # R×C: Freeman–Halton simulated-exact p-value + Cramér's V (odds ratio is undefined).
    sub = df[[outcome, group]].dropna()
    o_codes, n_rows = _codes(sub[outcome].astype(str).tolist())
    g_codes, n_cols = _codes(sub[group].astype(str).tolist())
    p, n_perm = _rxc_fisher_perm(o_codes, g_codes, n_rows, n_cols)
    chi2 = float(stats.chi2_contingency(table)[0])
    n = int(table.values.sum())
    n_dim = min(n_rows - 1, n_cols - 1)
    v = math.sqrt(chi2 / (n * n_dim)) if n and n_dim else 0.0
    ci = _bootstrap_cramers_v(o_codes, g_codes, n_rows, n_cols)
    checks = [
        _check("Exact test", "MET",
               f"Freeman–Halton via {n_perm} fixed-margin permutations "
               "(no large-sample χ² approximation)."),
        _check("Independence of observations", "UNTESTABLE", "Assumed by design."),
    ]
    return _result("FISHER_EXACT", chi2, p, None, "Cramér's V", v, "v",
                   ci[0], ci[1], checks, ci,
                   [f"{n_rows}×{n_cols} table, N = {n}; odds ratio is undefined for R×C "
                    "(Cramér's V reported)."])


def _mcnemar(df: pd.DataFrame, outcome: str, group: str) -> dict[str, Any]:
    from statsmodels.stats.contingency_tables import mcnemar
    table = pd.crosstab(df[outcome], df[group])
    if table.shape != (2, 2):
        return _not_implemented("MCNEMAR")
    arr = table.values
    res = mcnemar(arr)
    b_, c_ = float(arr[0, 1]), float(arr[1, 0])
    odds = (b_ / c_) if c_ > 0 else float("nan")
    checks = [_check("Paired binary structure", "UNTESTABLE", "Assumed by design."),
              _check("Discordant pairs ≥ 25",
                     "MET" if (b_ + c_) >= 25 else "MARGINAL",
                     f"{int(b_ + c_)} discordant pairs.")]
    return _result("MCNEMAR", res.statistic, res.pvalue, None, "odds ratio (discordant)",
                   _finite(odds, 1.0), "ratio", _finite(odds, 1.0), _finite(odds, 1.0),
                   checks, (_finite(odds, 1.0), _finite(odds, 1.0)),
                   ["McNemar's test for paired binary data."])


def _pearson(df: pd.DataFrame, outcome: str, predictor: str, selection: Any) -> dict[str, Any]:
    x, y = _xy(df, outcome, predictor)
    if len(x) < 3:
        return _not_implemented("PEARSON_CORRELATION")
    res = stats.pearsonr(x, y)
    try:
        ci = res.confidence_interval(0.95)
        ci_lo, ci_hi = float(ci.low), float(ci.high)
    except Exception:
        ci_lo, ci_hi = float("nan"), float("nan")
    checks = _corr_checks(x, y, selection, linear=True)
    return _result("PEARSON_CORRELATION", res.statistic, res.pvalue, float(len(x) - 2),
                   "Pearson's r", res.statistic, "r", ci_lo, ci_hi, checks, (ci_lo, ci_hi),
                   [f"Linear association between {predictor} and {outcome} (n = {len(x)})."])


def _spearman(df: pd.DataFrame, outcome: str, predictor: str, selection: Any) -> dict[str, Any]:
    x, y = _xy(df, outcome, predictor)
    if len(x) < 3:
        return _not_implemented("SPEARMAN_CORRELATION")
    res = stats.spearmanr(x, y)
    ci = _boot_paired(x, y, lambda a, b: float(stats.spearmanr(a, b).statistic))
    checks = _corr_checks(x, y, selection, linear=False)
    return _result("SPEARMAN_CORRELATION", res.statistic, res.pvalue, None,
                   "Spearman's ρ", res.statistic, "r", ci[0], ci[1], checks, ci,
                   [f"Monotone (rank) association: {predictor} vs {outcome} (n = {len(x)})."])


def _maxbet(df: pd.DataFrame, outcome: str, predictor: str) -> dict[str, Any]:
    from hta.bet_screen import maxbet
    x, y = _xy(df, outcome, predictor)
    if len(x) < 8:
        return _not_implemented("MAXBET")
    res = maxbet(x, y, seed=0)
    n = len(x)
    effect = res.bet_z / math.sqrt(n) if n else 0.0
    ci = _boot_paired(x, y, lambda a, b: maxbet(a, b, seed=0).bet_z / math.sqrt(len(a)))
    checks = [
        _check("Minimum sample size", "MET" if n >= 8 else "VIOLATED", f"n = {n} (need ≥ 8)."),
        _check("Continuity", "MET", "Ties are jittered before the copula transform."),
    ]
    notes = [f"Dominant interaction {res.bid} → {res.form} dependence.",
             "BET detects nonlinear dependence that Pearson/Spearman can miss."]
    if res.region_description:
        notes.append(res.region_description)
    return _result("MAXBET", res.bet_z, res.p_value, None, "BET symmetry (|S|/n)",
                   effect, "r", ci[0], ci[1], checks, ci, notes)


def _corr_checks(
    x: list[float], y: list[float], selection: Any, linear: bool,
) -> list[dict[str, Any]]:
    bet = (selection.computed or {}).get("BET", "") if selection is not None else ""
    if linear:
        nonlinear = "form=PARABOLIC" in bet or "form=SINUSOIDAL" in bet or \
                    "form=CHECKERBOARD" in bet or "form=COMPLEX" in bet
        lin = _check("Linearity", "VIOLATED" if nonlinear else "MET",
                     "BET screen: " + (bet or "no nonlinear pattern detected."))
    else:
        lin = _check("Monotonicity", "MET",
                     "A monotone (rank) relationship is the estimand; linearity not required.")
    return [lin, _check("Sample size", "MET" if len(x) >= 10 else "MARGINAL",
                        f"n = {len(x)} paired observations.")]


def _formula_term(col: str) -> str:
    """Wrap a column name in Q() so statsmodels handles spaces and special chars."""
    return f"Q('{col}')"


_Q_RE = __import__("re").compile(r"Q\('(.+?)'\)")


def _clean_param_name(name: str) -> str:
    """Strip statsmodels Q('...') quoting from a parameter label."""
    m = _Q_RE.fullmatch(name.strip())
    return m.group(1) if m else name


def _glm_count(df: pd.DataFrame, outcome: str, regressor: Optional[str],
               neg_binom: bool,
               regressors: Optional[list[str]] = None) -> dict[str, Any]:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    name = "NEGATIVE_BINOMIAL_REGRESSION" if neg_binom else "POISSON_REGRESSION"
    preds = regressors if regressors and len(regressors) >= 2 else (
        [regressor] if regressor else []
    )
    if not preds:
        return _not_implemented(name)
    cols_needed = [outcome] + preds
    d = df[cols_needed].copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    for p in preds:
        d[p] = pd.to_numeric(d[p], errors="coerce")
    d = d.dropna()
    if len(d) < max(5, len(preds) + 2):
        return _not_implemented(name)
    rhs = " + ".join(_formula_term(p) for p in preds)
    formula = f"{_formula_term(outcome)} ~ {rhs}"
    family = sm.families.NegativeBinomial() if neg_binom else sm.families.Poisson()
    try:
        model = smf.glm(formula, data=d, family=family).fit()
    except Exception:
        return _not_implemented(name)
    params = model.params.drop("Intercept", errors="ignore")
    if len(params) == 0:
        return _not_implemented(name)
    # Primary effect: first (or only) predictor; log model-level deviance p as omnibus
    key = params.index[0]
    coef = float(params.iloc[0])
    irr = math.exp(coef)
    conf = model.conf_int()
    irr_lo = math.exp(float(conf.loc[key].iloc[0]))
    irr_hi = math.exp(float(conf.loc[key].iloc[1]))
    p = float(model.pvalues[key])
    stat = float(model.tvalues[key])
    checks = [_check("Count outcome", "MET", "Outcome treated as non-negative counts."),
              _check("Overdispersion",
                     "MET" if neg_binom else "MARGINAL",
                     "Negative-binomial relaxes the variance = mean assumption." if neg_binom
                     else "Verify variance ≈ mean; prefer negative binomial if overdispersed.")]
    notes = [f"IRR = exp(β) for {preds[0]}; CI back-transformed from the log scale."]
    if len(preds) > 1:
        extra = []
        for k in params.index[1:]:
            try:
                irr_k = math.exp(float(params[k]))
                p_k = float(model.pvalues[k])
                label = _clean_param_name(k)
                extra.append(f"{label}: IRR={irr_k:.3f} (p={_fmt_p(p_k)})")
            except Exception:
                pass
        if extra:
            notes.append("Other predictors — " + "; ".join(extra) + ".")
    return _result(name, stat, p, None, "incidence-rate ratio", irr, "ratio",
                   irr_lo, irr_hi, checks, (irr_lo, irr_hi), notes)


def _linear_regression(df: pd.DataFrame, outcome: str,
                       predictors: list[str]) -> dict[str, Any]:
    """Multiple linear regression (OLS) via statsmodels.

    Reports the overall model F-test and R², with per-predictor coefficients and
    standardised betas in the notes. Effect size is R² (treated as eta² for interpretation).
    """
    import statsmodels.formula.api as smf
    cols_needed = [outcome] + predictors
    d = df[cols_needed].copy()
    for col in cols_needed:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    n_min = len(predictors) + 2
    if len(d) < max(10, n_min):
        return _not_implemented("LINEAR_REGRESSION")
    rhs = " + ".join(_formula_term(p) for p in predictors)
    formula = f"{_formula_term(outcome)} ~ {rhs}"
    try:
        model = smf.ols(formula, data=d).fit()
    except Exception:
        return _not_implemented("LINEAR_REGRESSION")
    r2 = float(model.rsquared)
    r2_adj = float(model.rsquared_adj)
    f_stat = float(model.fvalue) if model.fvalue is not None else 0.0
    f_p = float(model.f_pvalue) if model.f_pvalue is not None else 1.0
    n, k = len(d), len(predictors)
    # Cohen's f² = R² / (1 − R²) — "large" if f² ≥ 0.35 (Cohen 1988)
    f2 = r2 / (1 - r2) if r2 < 1.0 else float("inf")
    # Bootstrap CI for R²
    rng = random.Random(BOOTSTRAP_SEED)
    boot_r2: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = d.iloc[idx]
        try:
            import statsmodels.formula.api as _smf
            m = _smf.ols(formula, data=sub).fit()
            v = float(m.rsquared)
            if math.isfinite(v):
                boot_r2.append(v)
        except Exception:
            pass
    boot_r2.sort()
    ci_lo = _pct(boot_r2, 0.025) if boot_r2 else 0.0
    ci_hi = _pct(boot_r2, 0.975) if boot_r2 else r2
    # Assumption checks
    checks = [
        _check("Minimum sample size", "MET" if n >= 10 else "VIOLATED",
               f"n = {n}, {k} predictors (need n > 10·k for reliable estimates)."),
        _check("Independence of observations", "UNTESTABLE", "Assumed by design."),
        _check("Linearity / homoscedasticity", "UNTESTABLE",
               "Inspect residual vs. fitted and Q–Q plots."),
    ]
    # Per-predictor coefficient table in notes
    coef_lines: list[str] = []
    params = model.params.drop("Intercept", errors="ignore")
    pvals = model.pvalues.drop("Intercept", errors="ignore")
    conf = model.conf_int().drop("Intercept", errors="ignore")
    for col_k in params.index:
        b = float(params[col_k])
        p_k = float(pvals[col_k])
        lo_k = float(conf.loc[col_k].iloc[0])
        hi_k = float(conf.loc[col_k].iloc[1])
        label = _clean_param_name(col_k)
        coef_lines.append(f"{label}: β={b:.3f} [{lo_k:.3f}, {hi_k:.3f}] p={_fmt_p(p_k)}")
    notes = [
        f"OLS regression: outcome = {outcome}, {k} predictors.",
        f"Model: R² = {r2:.3f}, adjusted R² = {r2_adj:.3f}, F({k}, {n - k - 1}) = {f_stat:.2f}, p = {_fmt_p(f_p)}.",
        f"Cohen's f² = {f2:.3f} (large ≥ 0.35).",
        "Per-predictor (unstandardised β, 95% CI, p): " + "; ".join(coef_lines) + ".",
    ]
    return _result("LINEAR_REGRESSION", f_stat, f_p, float(n - k - 1),
                   "R²", r2, "eta2", ci_lo, ci_hi, checks, (ci_lo, ci_hi), notes)


def _logistic_regression(df: pd.DataFrame, outcome: str,
                         predictors: list[str]) -> dict[str, Any]:
    """Binary logistic regression (MLE) via statsmodels.

    Reports McFadden's pseudo-R², the overall likelihood-ratio test, and per-predictor
    odds ratios with 95% CIs and Wald p-values.
    """
    import statsmodels.formula.api as smf
    cols_needed = [outcome] + predictors
    d = df[cols_needed].copy()
    for col in cols_needed:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    if len(d) < max(10, len(predictors) + 2):
        return _not_implemented("LOGISTIC_REGRESSION")
    # Map outcome to 0/1 (two unique numeric values required)
    uniq = sorted(d[outcome].unique())
    if len(uniq) != 2:
        return _not_implemented("LOGISTIC_REGRESSION")
    d = d.copy()
    d["_y01"] = (d[outcome] == uniq[1]).astype(int)
    rhs = " + ".join(_formula_term(p) for p in predictors)
    formula = f"_y01 ~ {rhs}"
    try:
        model = smf.logit(formula, data=d).fit(disp=0)
    except Exception:
        return _not_implemented("LOGISTIC_REGRESSION")
    # McFadden's pseudo-R²
    ll_model = float(model.llf)
    ll_null = float(model.llnull)
    mcfadden = 1.0 - (ll_model / ll_null) if ll_null != 0 else 0.0
    mcfadden = max(0.0, mcfadden)
    # Overall LR test p-value
    lr_stat = -2.0 * (ll_null - ll_model)
    from scipy.stats import chi2 as _chi2
    lr_p = float(1.0 - _chi2.cdf(lr_stat, df=len(predictors)))
    n, k = len(d), len(predictors)
    # Bootstrap CI for McFadden's R²
    rng = random.Random(BOOTSTRAP_SEED)
    boot_r2: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = d.iloc[idx]
        if sub["_y01"].nunique() < 2:
            continue
        try:
            import statsmodels.formula.api as _smf
            m = _smf.logit(formula, data=sub).fit(disp=0)
            v = max(0.0, 1.0 - float(m.llf) / float(m.llnull)) if m.llnull != 0 else 0.0
            if math.isfinite(v):
                boot_r2.append(v)
        except Exception:
            pass
    boot_r2.sort()
    ci_lo = _pct(boot_r2, 0.025) if boot_r2 else 0.0
    ci_hi = _pct(boot_r2, 0.975) if boot_r2 else mcfadden
    checks = [
        _check("Binary outcome", "MET",
               f"Outcome mapped to 0 ({uniq[0]}) / 1 ({uniq[1]})."),
        _check("Minimum sample size", "MET" if n >= 10 else "VIOLATED",
               f"n = {n}, {k} predictors."),
        _check("Complete separation", "UNTESTABLE",
               "Check for perfect separation if the model fails to converge."),
    ]
    # Per-predictor OR table
    params = model.params.drop("Intercept", errors="ignore")
    pvals = model.pvalues.drop("Intercept", errors="ignore")
    conf = model.conf_int().drop("Intercept", errors="ignore")
    coef_lines: list[str] = []
    for col_k in params.index:
        b = float(params[col_k])
        p_k = float(pvals[col_k])
        or_k = math.exp(b)
        lo_k = math.exp(float(conf.loc[col_k].iloc[0]))
        hi_k = math.exp(float(conf.loc[col_k].iloc[1]))
        label = _clean_param_name(col_k)
        coef_lines.append(f"{label}: OR={or_k:.3f} [{lo_k:.3f}, {hi_k:.3f}] p={_fmt_p(p_k)}")
    notes = [
        f"Logistic regression: outcome = {outcome} (1 = '{uniq[1]}'), {k} predictors.",
        f"Model: McFadden's R² = {mcfadden:.3f}, LR χ²({k}) = {lr_stat:.2f}, p = {_fmt_p(lr_p)}.",
        "Per-predictor odds ratios (OR, 95% CI, p): " + "; ".join(coef_lines) + ".",
    ]
    return _result("LOGISTIC_REGRESSION", lr_stat, lr_p, float(k),
                   "McFadden's R²", mcfadden, "eta2", ci_lo, ci_hi, checks, (ci_lo, ci_hi), notes)


# ── survival (§6.5b) and diagnostic accuracy (§6.5c) ─────────────────────────

def _find_event_column(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    """A 0/1 event-/censoring-indicator column (the companion a survival analysis needs)."""
    for col in df.columns:
        if col in exclude or not EVENT_NAME_RE.search(str(col)):
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        uniq = {float(v) for v in vals.unique().tolist()}
        if uniq and uniq <= {0.0, 1.0}:
            return str(col)
    return None


def _survival_frame(df: pd.DataFrame, outcome: str, event_col: str,
                    extra: list[str]) -> pd.DataFrame:
    sub = df[[outcome, event_col, *extra]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub[event_col] = pd.to_numeric(sub[event_col], errors="coerce")
    return sub[sub[outcome] >= 0].dropna(subset=[outcome, event_col])


def _km_medians(sub: pd.DataFrame, outcome: str, event_col: str, group: str,
                labels: list[str]) -> list[str]:
    from lifelines import KaplanMeierFitter
    out: list[str] = []
    for lab in labels:
        g = sub[sub[group].astype(str) == lab]
        try:
            m = KaplanMeierFitter().fit(g[outcome], g[event_col]).median_survival_time_
            out.append(f"{lab}={float(m):.2f}")
        except Exception:
            pass
    return out


def _logrank(df: pd.DataFrame, outcome: str, group: str, event_col: str) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    sub = _survival_frame(df, outcome, event_col, [group]).dropna(subset=[group])
    labels = sorted(str(x) for x in sub[group].unique())
    if len(labels) < 2:
        return _not_implemented("LOG_RANK")
    checks = [
        _check("Non-informative censoring", "UNTESTABLE",
               "Censoring is assumed unrelated to prognosis; verify dropout is non-informative."),
        _check("Proportional hazards", "UNTESTABLE",
               "A single HR assumes proportional hazards; inspect the KM curves for crossing."),
    ]
    medians = _km_medians(sub, outcome, event_col, group, labels)
    notes = [f"Median survival per group: {', '.join(medians)}." if medians
             else "Median survival not reached in some groups.",
             "Kaplan–Meier + log-rank; censoring assumed non-informative."]
    if len(labels) == 2:
        a = sub[sub[group].astype(str) == labels[0]]
        b = sub[sub[group].astype(str) == labels[1]]
        lr = logrank_test(a[outcome], b[outcome], a[event_col], b[event_col])
        d = sub.assign(_g=(sub[group].astype(str) == labels[1]).astype(int))
        try:
            row = CoxPHFitter().fit(d[[outcome, event_col, "_g"]], duration_col=outcome,
                                    event_col=event_col).summary.loc["_g"]
            hr = float(row["exp(coef)"])
            hr_lo, hr_hi = float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"])
        except Exception:
            hr = hr_lo = hr_hi = float("nan")
        notes.insert(0, f"Hazard ratio ({labels[1]} vs {labels[0]}) = {_finite(hr, 1.0):.3f}.")
        return _result("LOG_RANK", float(lr.test_statistic), float(lr.p_value), None,
                       "hazard ratio", _finite(hr, 1.0), "ratio", _finite(hr_lo, hr),
                       _finite(hr_hi, hr), checks, (_finite(hr_lo, hr), _finite(hr_hi, hr)), notes)
    mlr = multivariate_logrank_test(sub[outcome], sub[group], sub[event_col])
    notes.insert(0, f"{len(labels)}-group omnibus log-rank (no single hazard ratio).")
    return _result("LOG_RANK", float(mlr.test_statistic), float(mlr.p_value),
                   float(len(labels) - 1), "hazard ratio", 1.0, "ratio", 1.0, 1.0, checks,
                   (1.0, 1.0), notes)


def _cox(df: pd.DataFrame, outcome: str, covariate: str, event_col: str) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    sub = _survival_frame(df, outcome, event_col, [covariate]).dropna(subset=[covariate])
    numeric = pd.to_numeric(sub[covariate], errors="coerce")
    if numeric.notna().all() and int(numeric.nunique()) > 1:
        sub = sub.assign(_x=numeric)
    else:
        labels = sorted(str(x) for x in sub[covariate].unique())
        if len(labels) < 2:
            return _not_implemented("COX_REGRESSION")
        sub = sub.assign(_x=(sub[covariate].astype(str) == labels[-1]).astype(int))
    d = sub[[outcome, event_col, "_x"]]
    try:
        cph = CoxPHFitter().fit(d, duration_col=outcome, event_col=event_col)
    except Exception:
        return _not_implemented("COX_REGRESSION")
    row = cph.summary.loc["_x"]
    hr, p, z = float(row["exp(coef)"]), float(row["p"]), float(row["z"])
    hr_lo, hr_hi = float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"])
    ph_status, ph_p, ph_note = "UNTESTABLE", None, "Proportional-hazards test unavailable."
    try:
        ph_p = float(min(proportional_hazard_test(cph, d).p_value))
        ph_status = "VIOLATED" if ph_p < 0.05 else "MET"
        ph_note = (f"Scaled Schoenfeld residual test p = {ph_p:.3f}. "
                   + ("Non-proportional hazards — a single HR is misleading; consider "
                      "time-varying effects or RMST." if ph_p < 0.05
                      else "No evidence against proportional hazards."))
    except Exception:
        pass
    checks = [_check("Proportional hazards", ph_status, ph_note, p_value=ph_p),
              _check("Non-informative censoring", "UNTESTABLE",
                     "Censoring assumed unrelated to prognosis.")]
    return _result("COX_REGRESSION", z, p, None, "hazard ratio", hr, "ratio", hr_lo, hr_hi,
                   checks, (hr_lo, hr_hi),
                   [f"Hazard ratio = exp(β) for {covariate}; CI back-transformed from the "
                    "log scale."])


def _boot_auc(y: list[int], score: list[float]) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(y)
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        try:
            out.append(float(roc_auc_score(yy, [score[i] for i in idx])))
        except Exception:
            pass
    out.sort()
    return (_pct(out, 0.025), _pct(out, 0.975)) if out else (float("nan"), float("nan"))


def _roc_auc(df: pd.DataFrame, outcome: str, predictor: str) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score, roc_curve
    sub = df[[outcome, predictor]].copy()
    sub[predictor] = pd.to_numeric(sub[predictor], errors="coerce")
    sub = sub.dropna()
    labels = sorted(str(x) for x in sub[outcome].unique())
    if len(labels) != 2 or len(sub) < 5:
        return _not_implemented("ROC_AUC")
    pos = labels[-1]
    y = [1 if str(v) == pos else 0 for v in sub[outcome].tolist()]
    score = [float(v) for v in sub[predictor].tolist()]
    try:
        auc = float(roc_auc_score(y, score))
    except Exception:
        return _not_implemented("ROC_AUC")
    fpr, tpr, _thr = roc_curve(y, score)
    j = max(range(len(fpr)), key=lambda i: tpr[i] - fpr[i])
    sens, spec = float(tpr[j]), float(1 - fpr[j])
    pos_s = [s for s, yy in zip(score, y) if yy == 1]
    neg_s = [s for s, yy in zip(score, y) if yy == 0]
    p = float(stats.mannwhitneyu(pos_s, neg_s, alternative="two-sided").pvalue)
    ci = _boot_auc(y, score)
    checks = [_check("Binary reference standard", "MET",
                     f"Discriminating '{pos}' from '{labels[0]}'."),
              _check("Sample size", "MET" if len(sub) >= 20 else "MARGINAL", f"n = {len(sub)}.")]
    notes = [f"AUC for {predictor} discriminating '{pos}' (Mann–Whitney equivalent p-value).",
             f"Youden-optimal threshold: sensitivity = {sens:.2f}, specificity = {spec:.2f}.",
             "95% CI via 1000-sample bootstrap (DeLong is the analytic alternative)."]
    return _result("ROC_AUC", auc, p, None, "AUC", auc, "r",
                   _finite(ci[0], auc), _finite(ci[1], auc), checks,
                   (_finite(ci[0], auc), _finite(ci[1], auc)), notes)


# ── confounder-adjusted estimate (§5.3) ──────────────────────────────────────

def _adjusted_estimate(test_name: str, df: pd.DataFrame, outcome: str,
                       group: Optional[str], predictor: Optional[str],
                       covars: list[str]) -> Optional[str]:
    """A confounder-adjusted estimate that the elicited confounders actually move:
    partial correlation for the association tests, ANCOVA for the continuous group
    comparisons. Returns a note, or None when adjustment does not apply / cannot be run."""
    if not covars:
        return None
    cov_label = ", ".join(covars)
    try:
        import pingouin as pg
        if test_name in ("PEARSON_CORRELATION", "SPEARMAN_CORRELATION") and predictor:
            method = "spearman" if test_name == "SPEARMAN_CORRELATION" else "pearson"
            sub = df[[outcome, predictor, *covars]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < len(covars) + 3:
                return None
            row = pg.partial_corr(data=sub, x=predictor, y=outcome, covar=list(covars),
                                  method=method).iloc[0]
            r, p = float(row["r"]), float(row["p_val"])
            return (f"Adjusted for {cov_label}: partial {method} correlation r = {r:.3f} "
                    f"(p = {_fmt_p(p)}) — compare with the unadjusted estimate above.")
        if test_name in ("WELCH_T", "INDEPENDENT_T", "WELCH_ANOVA", "ONE_WAY_ANOVA") and group:
            sub = df[[outcome, group, *covars]].copy()
            for col in (outcome, *covars):
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna()
            if len(sub) < len(covars) + 4:
                return None
            anc = pg.ancova(data=sub, dv=outcome, between=group, covar=list(covars))
            match = anc[anc["Source"] == group]
            if match.empty:
                return None
            row = match.iloc[0]
            f, p, np2 = float(row["F"]), float(row["p_unc"]), float(row["np2"])
            tail = f", partial η² = {np2:.3f}" if math.isfinite(np2) else ""
            return (f"Adjusted for {cov_label} (ANCOVA): {group} effect F = {f:.2f}, "
                    f"p = {_fmt_p(p)}{tail}.")
    except Exception:
        return None
    return None


# ── dispatch ──────────────────────────────────────────────────────────────────

def _dispatch(
    test_name: str, df: pd.DataFrame, outcome: str, group: Optional[str],
    predictor: Optional[str], selection: Any,
    predictors: Optional[list[str]] = None,
) -> dict[str, Any]:
    if test_name == "WELCH_T":
        return _two_sample_t(df, outcome, group, equal_var=False)  # type: ignore[arg-type]
    if test_name == "INDEPENDENT_T":
        return _two_sample_t(df, outcome, group, equal_var=True)   # type: ignore[arg-type]
    if test_name == "MANN_WHITNEY_U":
        return _mann_whitney(df, outcome, group)                   # type: ignore[arg-type]
    if test_name == "PAIRED_T":
        return _paired_t(df, outcome, group)                       # type: ignore[arg-type]
    if test_name == "WILCOXON_SIGNED_RANK":
        return _wilcoxon(df, outcome, group)                       # type: ignore[arg-type]
    if test_name == "WELCH_ANOVA":
        return _anova(df, outcome, group, welch=True)              # type: ignore[arg-type]
    if test_name == "ONE_WAY_ANOVA":
        return _anova(df, outcome, group, welch=False)             # type: ignore[arg-type]
    if test_name == "KRUSKAL_WALLIS":
        return _kruskal(df, outcome, group)                        # type: ignore[arg-type]
    if test_name == "CHI_SQUARED":
        return _chi_squared(df, outcome, group)                    # type: ignore[arg-type]
    if test_name == "FISHER_EXACT":
        return _fisher(df, outcome, group)                         # type: ignore[arg-type]
    if test_name == "MCNEMAR":
        return _mcnemar(df, outcome, group)                        # type: ignore[arg-type]
    if test_name == "PEARSON_CORRELATION":
        return _pearson(df, outcome, predictor, selection)         # type: ignore[arg-type]
    if test_name == "SPEARMAN_CORRELATION":
        return _spearman(df, outcome, predictor, selection)        # type: ignore[arg-type]
    if test_name == "MAXBET":
        return _maxbet(df, outcome, predictor)                     # type: ignore[arg-type]
    if test_name == "POISSON_REGRESSION":
        return _glm_count(df, outcome, predictor or group, neg_binom=False,
                          regressors=predictors)
    if test_name == "NEGATIVE_BINOMIAL_REGRESSION":
        return _glm_count(df, outcome, predictor or group, neg_binom=True,
                          regressors=predictors)
    if test_name == "LINEAR_REGRESSION":
        preds = predictors or ([predictor] if predictor else None)
        if not preds:
            return _not_implemented("LINEAR_REGRESSION")
        return _linear_regression(df, outcome, preds)
    if test_name == "LOGISTIC_REGRESSION":
        preds = predictors or ([predictor] if predictor else None)
        if not preds:
            return _not_implemented("LOGISTIC_REGRESSION")
        return _logistic_regression(df, outcome, preds)
    if test_name == "LOG_RANK":
        ev = _find_event_column(df, {outcome, group or "", predictor or ""})
        return _logrank(df, outcome, group, ev) if (ev and group) else _not_implemented("LOG_RANK")
    if test_name == "COX_REGRESSION":
        ev = _find_event_column(df, {outcome, group or "", predictor or ""})
        cov = predictor or group
        return _cox(df, outcome, cov, ev) if (ev and cov) else _not_implemented("COX_REGRESSION")
    if test_name == "ROC_AUC":
        return _roc_auc(df, outcome, predictor) if predictor else _not_implemented("ROC_AUC")
    return _not_implemented(test_name)


# ── public entry point ────────────────────────────────────────────────────────

def execute(
    test: StatisticalTest | str,
    df: pd.DataFrame,
    outcome: str,
    group: Optional[str],
    predictor: Optional[str],
    design: Any = None,
    selection: Any = None,
    predictors: Optional[list[str]] = None,
) -> TestResult:
    """Run `test` on the data and return a validated `TestResult`.

    `predictors` (optional list) enables multi-predictor regression; when present it
    takes precedence over the single `predictor` argument for regression tests.

    Raises `ValueError` if `test` is not a recognised `StatisticalTest`; any error from
    the individual test is caught and surfaced as an UNTESTABLE result so the pipeline
    still produces a report.
    """
    test_name = test.value if isinstance(test, StatisticalTest) else str(test)
    # Validate up front: an unknown name cannot populate the strict `test_used` enum.
    StatisticalTest(test_name)
    try:
        d = _dispatch(test_name, df, outcome, group, predictor, selection, predictors)
    except Exception as exc:  # never crash the whole run on a single-test failure
        d = _result(test_name, 0.0, 1.0, None, "—", 0.0, "r", 0.0, 0.0,
                    [_check("Execution", "UNTESTABLE", f"Test failed: {exc}")],
                    (0.0, 0.0), [f"Execution error: {exc}"])
        return TestResult.model_validate(d)
    # Confounder adjustment (§5.3) — skip for regression (confounders are already in the model).
    if test_name not in ("LINEAR_REGRESSION", "LOGISTIC_REGRESSION",
                         "POISSON_REGRESSION", "NEGATIVE_BINOMIAL_REGRESSION"):
        try:
            exclude = {x for x in (outcome, group, predictor) if x}
            covars = usable_adjustment_covariates(design, df, exclude)
            note = _adjusted_estimate(test_name, df, outcome, group, predictor, covars)
            if note:
                d["notes"].append(note)
        except Exception:
            pass
    return TestResult.model_validate(d)
