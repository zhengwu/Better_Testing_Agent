"""Pydantic models for statistical test selection, assumption checking, and results."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StatisticalTest(str, Enum):
    """Enumeration of all statistical tests the agent can select and execute."""

    INDEPENDENT_T = "INDEPENDENT_T"  # equal-variance Student's t — explicit override only
    PAIRED_T = "PAIRED_T"
    WELCH_T = "WELCH_T"  # default for 2-group between-subjects continuous (no variance pretest)
    ONE_WAY_ANOVA = "ONE_WAY_ANOVA"  # pooled-variance ANOVA — explicit override only
    WELCH_ANOVA = "WELCH_ANOVA"  # default for 3+ group between-subjects continuous
    KRUSKAL_WALLIS = "KRUSKAL_WALLIS"
    MANN_WHITNEY_U = "MANN_WHITNEY_U"
    WILCOXON_SIGNED_RANK = "WILCOXON_SIGNED_RANK"
    CHI_SQUARED = "CHI_SQUARED"
    FISHER_EXACT = "FISHER_EXACT"
    MCNEMAR = "MCNEMAR"
    PEARSON_CORRELATION = "PEARSON_CORRELATION"
    SPEARMAN_CORRELATION = "SPEARMAN_CORRELATION"
    MAXBET = "MAXBET"  # nonlinear independence (BET); default BET-family choice
    BEAST = "BEAST"  # data-adaptive BET variant; reserved for explicit override
    # Count / rate outcomes (healthcare incidence) — effect size is the incidence-rate ratio.
    POISSON_REGRESSION = "POISSON_REGRESSION"  # default for counts/rates when not overdispersed
    NEGATIVE_BINOMIAL_REGRESSION = "NEGATIVE_BINOMIAL_REGRESSION"  # default when overdispersed
    # Time-to-event / survival outcomes — effect size is the hazard ratio.
    LOG_RANK = "LOG_RANK"  # group comparison of survival curves (no covariates)
    COX_REGRESSION = "COX_REGRESSION"  # survival with covariate adjustment (hazard ratio)
    # Diagnostic-accuracy evaluation — discrimination via the ROC area under the curve.
    ROC_AUC = "ROC_AUC"  # AUC with DeLong CI; DeLong test to compare two AUCs
    # Multiple-predictor regression — selectable when 2+ predictors are provided.
    LINEAR_REGRESSION = "LINEAR_REGRESSION"    # continuous outcome, OLS; effect size = R²
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"  # binary outcome, MLE; effect size = McFadden's R²
    # Reserved for v0.2.0 — clustered / longitudinal data.
    LINEAR_MIXED_MODEL = "LINEAR_MIXED_MODEL"
    GENERALIZED_ESTIMATING_EQUATIONS = "GENERALIZED_ESTIMATING_EQUATIONS"


class AssumptionStatus(str, Enum):
    """Outcome of checking a statistical assumption."""

    MET = "MET"
    VIOLATED = "VIOLATED"
    UNTESTABLE = "UNTESTABLE"
    MARGINAL = "MARGINAL"


class AssumptionCheck(BaseModel):
    """Result of checking a single statistical assumption."""

    assumption_name: str
    status: AssumptionStatus
    test_used: Optional[str] = None
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    note: str


class EffectSize(BaseModel):
    """Standardised effect size estimate with confidence interval.

    measure_name examples: "Cohen's d", "Cramér's V", "rank-biserial r",
    "eta-squared", "odds ratio". Healthcare effect measures use the same shape:
    "incidence-rate ratio" (counts/rates), "hazard ratio" (survival), "risk ratio",
    "AUC" (diagnostic discrimination). Ratio measures report the CI on the ratio
    scale (back-transformed from the log scale).
    interpretation follows Cohen (1988) conventions where applicable; for ratio
    measures it describes direction/magnitude (e.g. "protective", "harmful").
    """

    measure_name: str
    value: float
    interpretation: str  # e.g. "small", "medium", "large"
    ci_lower: float
    ci_upper: float


class TestResult(BaseModel):
    """Complete output of a statistical test, including effect size and assumption checks.

    is_significant uses a fixed alpha of 0.05; downstream consumers may apply
    their own threshold.
    """

    test_used: StatisticalTest
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[float] = None
    effect_size: EffectSize
    assumption_checks: list[AssumptionCheck] = Field(default_factory=list)
    confidence_interval: tuple[float, float]
    is_significant: bool  # p_value < 0.05
    power: Optional[float] = None
    notes: list[str] = Field(default_factory=list)
