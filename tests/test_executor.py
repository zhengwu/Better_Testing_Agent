"""Tests for the engine executor (`hta.modules.executor`).

Each selectable test must return a validated `TestResult` with the right `test_used`, a
p-value in range, and a populated effect size. Two cases cross-check the wiring against a
direct scipy computation; the survival/diagnostic/reserved tests must degrade to an
UNTESTABLE result, and an unknown test name must raise.
"""

from __future__ import annotations

import math
import random

import pandas as pd
import pytest
from scipy import stats

from hta.models.test import StatisticalTest, TestResult
from hta.modules.executor import execute


def _two_groups() -> pd.DataFrame:
    a = [10.2, 11.4, 9.1, 12.3, 10.8, 11.0, 13.1, 9.6, 10.5, 12.0, 10.1, 11.7]
    b = [20.2, 21.4, 19.1, 22.3, 20.8, 21.0, 23.1, 19.6, 20.5, 22.0, 20.3, 21.6]
    return pd.DataFrame({"arm": ["A"] * len(a) + ["B"] * len(b), "score": a + b})


def _three_groups() -> pd.DataFrame:
    rows = []
    for label, base in (("A", 10.0), ("B", 14.0), ("C", 18.0)):
        for i in range(10):
            rows.append({"arm": label, "score": base + (i % 5) * 0.7})
    return pd.DataFrame(rows)


def test_welch_t_matches_scipy() -> None:
    df = _two_groups()
    a = df[df.arm == "A"].score.tolist()
    b = df[df.arm == "B"].score.tolist()
    expected = stats.ttest_ind(a, b, equal_var=False)
    res = execute("WELCH_T", df, "score", "arm", None)
    assert isinstance(res, TestResult)
    assert res.test_used == StatisticalTest.WELCH_T
    # The executor stores `statistic` rounded to 4 dp; allow for that.
    assert res.statistic == pytest.approx(expected.statistic, abs=1e-3)
    assert res.p_value == pytest.approx(expected.pvalue, rel=1e-3)
    assert res.is_significant is True
    assert res.effect_size.measure_name == "Cohen's d"
    assert any("Sensitivity" in n for n in res.notes)  # §5.5 sensitivity power


def test_independent_t_has_levene_check() -> None:
    res = execute("INDEPENDENT_T", _two_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.INDEPENDENT_T
    assert any(c.assumption_name == "Equal variances" for c in res.assumption_checks)


def test_mann_whitney() -> None:
    res = execute("MANN_WHITNEY_U", _two_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.MANN_WHITNEY_U
    assert res.effect_size.measure_name == "rank-biserial r"
    assert 0.0 <= res.p_value <= 1.0


def test_paired_t() -> None:
    res = execute("PAIRED_T", _two_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.PAIRED_T
    assert res.effect_size.measure_name == "Cohen's d_z"


def test_wilcoxon() -> None:
    res = execute("WILCOXON_SIGNED_RANK", _two_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.WILCOXON_SIGNED_RANK


def test_welch_anova_three_groups() -> None:
    res = execute("WELCH_ANOVA", _three_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.WELCH_ANOVA
    assert res.effect_size.measure_name == "eta-squared"


def test_one_way_anova() -> None:
    res = execute("ONE_WAY_ANOVA", _three_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.ONE_WAY_ANOVA
    assert res.degrees_of_freedom == pytest.approx(2.0)


def test_kruskal() -> None:
    res = execute("KRUSKAL_WALLIS", _three_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.KRUSKAL_WALLIS


def test_chi_squared_rxc() -> None:
    df = pd.DataFrame({
        "treat": (["A"] * 30) + (["B"] * 30) + (["C"] * 30),
        "out": ((["x"] * 10 + ["y"] * 10 + ["z"] * 10) * 3),
    })
    res = execute("CHI_SQUARED", df, "out", "treat", None)
    assert res.test_used == StatisticalTest.CHI_SQUARED
    assert res.effect_size.measure_name == "Cramér's V"


def test_fisher_exact_2x2() -> None:
    df = pd.DataFrame({"treat": ["A", "A", "A", "B", "B", "B", "A", "B"],
                       "out": ["x", "x", "y", "y", "y", "x", "x", "y"]})
    res = execute("FISHER_EXACT", df, "out", "treat", None)
    assert res.test_used == StatisticalTest.FISHER_EXACT
    assert res.effect_size.measure_name == "odds ratio"


def test_mcnemar_2x2() -> None:
    df = pd.DataFrame({"before": (["+"] * 20) + (["-"] * 20),
                       "after": (["+"] * 10 + ["-"] * 10) + (["+"] * 15 + ["-"] * 5)})
    res = execute("MCNEMAR", df, "after", "before", None)
    assert res.test_used == StatisticalTest.MCNEMAR


def test_pearson_matches_scipy() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 19.9]
    df = pd.DataFrame({"x": x, "y": y})
    expected = stats.pearsonr(x, y)
    res = execute("PEARSON_CORRELATION", df, "y", None, "x", selection=None)
    assert res.test_used == StatisticalTest.PEARSON_CORRELATION
    assert res.statistic == pytest.approx(expected.statistic, abs=1e-3)
    assert res.p_value == pytest.approx(expected.pvalue, abs=1e-4)


def test_spearman() -> None:
    df = pd.DataFrame({"x": list(range(12)), "y": [v ** 1.5 for v in range(12)]})
    res = execute("SPEARMAN_CORRELATION", df, "y", None, "x", selection=None)
    assert res.test_used == StatisticalTest.SPEARMAN_CORRELATION
    assert res.effect_size.value == pytest.approx(1.0, abs=1e-9)  # monotone → ρ = 1


def test_maxbet_parabola() -> None:
    n = 120
    xs = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
    df = pd.DataFrame({"x": xs, "y": [x * x for x in xs]})
    res = execute("MAXBET", df, "y", None, "x")
    assert res.test_used == StatisticalTest.MAXBET
    assert res.is_significant is True  # strong nonlinear dependence


def test_poisson_regression() -> None:
    df = pd.DataFrame({"x": list(range(1, 21)) * 2,
                       "events": [max(0, round(0.6 * v)) for v in list(range(1, 21)) * 2]})
    res = execute("POISSON_REGRESSION", df, "events", None, "x")
    assert res.test_used == StatisticalTest.POISSON_REGRESSION
    assert res.effect_size.measure_name == "incidence-rate ratio"


def test_negative_binomial_regression() -> None:
    counts = [0, 1, 0, 5, 2, 14, 0, 9, 1, 30, 3, 25, 0, 7, 40, 2, 18, 1, 22, 0] * 2
    df = pd.DataFrame({"x": list(range(1, 21)) * 2, "events": counts})
    res = execute("NEGATIVE_BINOMIAL_REGRESSION", df, "events", None, "x")
    assert res.test_used == StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION


def test_linear_regression() -> None:
    rng = random.Random(42)
    n = 50
    x1 = [rng.uniform(0, 10) for _ in range(n)]
    x2 = [rng.uniform(0, 5) for _ in range(n)]
    y = [2.0 * x1[i] - 1.5 * x2[i] + rng.gauss(0, 1) for i in range(n)]
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    res = execute("LINEAR_REGRESSION", df, "y", None, None, None, None, ["x1", "x2"])
    assert res.test_used == StatisticalTest.LINEAR_REGRESSION
    assert isinstance(res.coefficient_table, list) and len(res.coefficient_table) > 0
    for row in res.coefficient_table:
        assert set(row.keys()) >= {"predictor", "estimate", "ci_lower", "ci_upper", "p_value"}
        assert all(math.isfinite(row[k]) for k in ("estimate", "ci_lower", "ci_upper", "p_value"))


def test_logistic_regression() -> None:
    rng = random.Random(7)
    n = 80
    x = [rng.uniform(-3, 3) for _ in range(n)]
    y = [1 if x[i] + rng.gauss(0, 0.5) > 0 else 0 for i in range(n)]
    df = pd.DataFrame({"y": y, "x": x})
    res = execute("LOGISTIC_REGRESSION", df, "y", None, "x")
    assert res.test_used == StatisticalTest.LOGISTIC_REGRESSION
    assert isinstance(res.coefficient_table, list) and len(res.coefficient_table) > 0
    for row in res.coefficient_table:
        assert set(row.keys()) >= {"predictor", "estimate", "ci_lower", "ci_upper", "p_value"}
        assert all(math.isfinite(row[k]) for k in ("estimate", "ci_lower", "ci_upper", "p_value"))


def test_unimplemented_test_is_untestable() -> None:
    """Survival/diagnostic tests are in the enum but not wired — they must not crash."""
    res = execute("LOG_RANK", _two_groups(), "score", "arm", None)
    assert res.test_used == StatisticalTest.LOG_RANK
    assert any(c.status.value == "UNTESTABLE" for c in res.assumption_checks)


def test_unknown_test_raises() -> None:
    with pytest.raises(ValueError):
        execute("NOT_A_REAL_TEST", _two_groups(), "score", "arm", None)


def test_single_group_degrades_gracefully() -> None:
    df = pd.DataFrame({"arm": ["A"] * 10, "score": [float(i) for i in range(10)]})
    res = execute("WELCH_T", df, "score", "arm", None)
    assert isinstance(res, TestResult)
    assert any(c.status.value == "UNTESTABLE" for c in res.assumption_checks)


def test_result_is_json_serialisable() -> None:
    res = execute("WELCH_T", _two_groups(), "score", "arm", None)
    dumped = res.model_dump(mode="json")
    assert dumped["test_used"] == "WELCH_T"
    assert math.isfinite(dumped["p_value"])


# ── Phase 2: post-hoc, real CIs, R×C exact ────────────────────────────────────

def test_welch_anova_is_true_welch_with_posthoc() -> None:
    import pingouin as pg
    df = _three_groups()
    res = execute("WELCH_ANOVA", df, "score", "arm", None)
    expected_f = float(pg.welch_anova(data=df, dv="score", between="arm").iloc[0]["F"])
    # The statistic must be Welch's F (pingouin), not scipy's Alexander–Govern.
    assert res.statistic == pytest.approx(expected_f, abs=1e-3)
    assert any("Games–Howell" in n for n in res.notes)
    assert res.confidence_interval[0] != res.confidence_interval[1]  # real bootstrap CI


def test_welch_anova_closed_form_matches_pingouin() -> None:
    import pingouin as pg

    from hta.modules.executor import _group_arrays, _welch_anova_closed_form
    df = _three_groups()
    _, groups = _group_arrays(df, "score", "arm")
    f, p, _df1 = _welch_anova_closed_form(groups)
    expected = pg.welch_anova(data=df, dv="score", between="arm").iloc[0]
    assert f == pytest.approx(float(expected["F"]), rel=1e-6)
    assert p == pytest.approx(float(expected["p_unc"]), rel=1e-6)


def test_one_way_anova_posthoc_in_notes() -> None:
    res = execute("ONE_WAY_ANOVA", _three_groups(), "score", "arm", None)
    assert any("Tukey" in n for n in res.notes)


def test_kruskal_posthoc_and_real_ci() -> None:
    res = execute("KRUSKAL_WALLIS", _three_groups(), "score", "arm", None)
    assert any("Dunn" in n for n in res.notes)
    assert res.confidence_interval[0] != res.confidence_interval[1]


def test_chi_squared_2x2_reports_odds_ratio() -> None:
    df = pd.DataFrame({"t": (["A"] * 20) + (["B"] * 20),
                       "out": (["x"] * 14 + ["y"] * 6) + (["x"] * 5 + ["y"] * 15)})
    res = execute("CHI_SQUARED", df, "out", "t", None)
    assert res.effect_size.measure_name == "Cramér's V"
    assert any("odds ratio" in n and "φ" in n for n in res.notes)
    assert res.confidence_interval[0] != res.confidence_interval[1]


def test_fisher_rxc_freeman_halton() -> None:
    df = pd.DataFrame({
        "t": (["A"] * 12) + (["B"] * 12) + (["C"] * 12),
        "out": (["x"] * 8 + ["y"] * 2 + ["z"] * 2) + (["x"] * 2 + ["y"] * 8 + ["z"] * 2)
               + (["x"] * 2 + ["y"] * 2 + ["z"] * 8),
    })
    res = execute("FISHER_EXACT", df, "out", "t", None)
    assert res.test_used == StatisticalTest.FISHER_EXACT
    assert res.effect_size.measure_name == "Cramér's V"  # OR undefined for R×C
    assert 0.0 <= res.p_value <= 1.0
    assert any("Freeman" in c.note for c in res.assumption_checks)


# ── Phase 3: confounder-adjusted estimates (§5.3) ─────────────────────────────

def test_adjusted_partial_correlation_note() -> None:
    # z drives both x and y; the partial correlation must differ from the raw one.
    z = [float(i % 15) for i in range(60)]
    df = pd.DataFrame({"x": [zi + (i % 3) * 0.4 for i, zi in enumerate(z)],
                       "y": [zi + (i % 4) * 0.3 for i, zi in enumerate(z)], "z": z})
    design = {"confounders": [{"name": "z", "is_measured": True,
                               "adjustment_recommended": True}]}
    res = execute("SPEARMAN_CORRELATION", df, "y", None, "x", design, None)
    assert any("Adjusted for z" in n and "partial" in n for n in res.notes)


def test_adjusted_ancova_note() -> None:
    rows = []
    for label, base in (("A", 10.0), ("B", 14.0)):
        for i in range(15):
            rows.append({"g": label, "y": base + (i % 6) * 0.7 + 0.5 * (i % 4), "c": float(i % 6)})
    df = pd.DataFrame(rows)
    design = {"confounders": [{"name": "c", "is_measured": True,
                              "adjustment_recommended": True}]}
    res = execute("WELCH_T", df, "y", "g", None, design, None)
    assert any("ANCOVA" in n for n in res.notes)


def test_no_adjustment_without_confounders() -> None:
    res = execute("WELCH_T", _two_groups(), "score", "arm", None)
    assert not any("Adjusted for" in n for n in res.notes)


# ── Phase 4: survival (§6.5b) and diagnostic accuracy (§6.5c) ─────────────────

def _survival_df() -> pd.DataFrame:
    rng = random.Random(0)
    rows = []
    for arm, scale in (("control", 9.0), ("treated", 20.0)):
        for _ in range(40):
            rows.append({"survival_time": round(rng.expovariate(1 / scale), 3),
                         "status": 1 if rng.random() < 0.85 else 0, "arm": arm})
    return pd.DataFrame(rows)


def test_log_rank() -> None:
    res = execute("LOG_RANK", _survival_df(), "survival_time", "arm", None)
    assert res.test_used == StatisticalTest.LOG_RANK
    assert res.effect_size.measure_name == "hazard ratio"
    assert 0.0 <= res.p_value <= 1.0
    assert any("Hazard ratio" in n or "Median survival" in n for n in res.notes)


def test_log_rank_without_event_column_is_untestable() -> None:
    df = _survival_df().drop(columns=["status"])
    res = execute("LOG_RANK", df, "survival_time", "arm", None)
    assert any(c.status.value == "UNTESTABLE" for c in res.assumption_checks)


def test_cox_regression_with_ph_check() -> None:
    rng = random.Random(1)
    rows = []
    for _ in range(90):
        age = rng.uniform(40, 80)
        rows.append({"fu_time": round(rng.expovariate(1 / max(2.0, 30 - 0.25 * (age - 40))), 3),
                     "death": 1 if rng.random() < 0.8 else 0, "age": round(age, 1)})
    res = execute("COX_REGRESSION", pd.DataFrame(rows), "fu_time", None, "age")
    assert res.test_used == StatisticalTest.COX_REGRESSION
    assert res.effect_size.measure_name == "hazard ratio"
    assert any(c.assumption_name == "Proportional hazards" for c in res.assumption_checks)


def test_roc_auc() -> None:
    rng = random.Random(2)
    rows = [{"disease": 1 if i < 50 else 0,
             "biomarker": round(rng.gauss(1.5 if i < 50 else 0.0, 1.0), 3)} for i in range(100)]
    res = execute("ROC_AUC", pd.DataFrame(rows), "disease", None, "biomarker")
    assert res.test_used == StatisticalTest.ROC_AUC
    assert res.effect_size.measure_name == "AUC"
    assert 0.6 < res.effect_size.value <= 1.0   # the biomarker discriminates disease
    assert any("sensitivity" in n for n in res.notes)
