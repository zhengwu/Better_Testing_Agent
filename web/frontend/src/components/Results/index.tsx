import ReactMarkdown from 'react-markdown'
import type { Report, CoefficientRow } from '../../types/api'
import { AssumptionBadge, CaveatBadge, SignificanceBadge } from '../shared/Badge'
import { CopyButton } from '../shared/CopyButton'
import { exportHtmlUrl } from '../../api/client'
import PlotViewer from './PlotViewer'

const TEST_LABELS: Record<string, string> = {
  WELCH_T: "Welch's t-test", INDEPENDENT_T: "Student's t-test",
  PAIRED_T: 'Paired t-test', MANN_WHITNEY_U: 'Mann–Whitney U',
  WILCOXON_SIGNED_RANK: 'Wilcoxon signed-rank', ONE_WAY_ANOVA: 'One-way ANOVA',
  KRUSKAL_WALLIS: 'Kruskal–Wallis', CHI_SQUARED: 'Chi-squared',
  FISHER_EXACT: "Fisher's exact", MCNEMAR: 'McNemar',
  PEARSON_CORRELATION: 'Pearson correlation', SPEARMAN_CORRELATION: 'Spearman correlation',
  MAXBET: 'MaxBET (nonlinear independence)',
  WELCH_ANOVA: "Welch's ANOVA",
  POISSON_REGRESSION: 'Poisson regression', NEGATIVE_BINOMIAL_REGRESSION: 'Negative binomial regression',
  LINEAR_REGRESSION: 'Multiple linear regression',
  LOGISTIC_REGRESSION: 'Logistic regression',
  ROC_AUC: 'AUC (ROC)',
  LOG_RANK: 'Log-rank test',
  COX_REGRESSION: 'Cox proportional-hazards',
  BEAST: 'BEAST (adaptive BET)',
}

const STAT_LABELS: Record<string, string> = {
  WELCH_T: 't', INDEPENDENT_T: 't', PAIRED_T: 't',
  MANN_WHITNEY_U: 'U', WILCOXON_SIGNED_RANK: 'W',
  WELCH_ANOVA: 'F', ONE_WAY_ANOVA: 'F', KRUSKAL_WALLIS: 'H',
  CHI_SQUARED: 'χ²', FISHER_EXACT: 'χ²', MCNEMAR: 'χ²',
  PEARSON_CORRELATION: 'r', SPEARMAN_CORRELATION: 'ρ',
  MAXBET: 'Z',
  POISSON_REGRESSION: 'z', NEGATIVE_BINOMIAL_REGRESSION: 'z',
  LINEAR_REGRESSION: 'F', LOGISTIC_REGRESSION: 'LR χ²',
  ROC_AUC: 'Z',
  LOG_RANK: 'χ²',
  COX_REGRESSION: 'z',
  BEAST: 'Z',
}

function fmt(n: number, decimals = 3) {
  return n.toFixed(decimals)
}

function fmtP(p: number) {
  return p < 0.001 ? '< 0.001' : p.toFixed(3)
}

function CoefficientTable({ rows, testUsed }: { rows: CoefficientRow[]; testUsed: string }) {
  const isLogistic = testUsed === 'LOGISTIC_REGRESSION'
  const estimateLabel = isLogistic ? 'OR' : 'β'
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 overflow-x-auto">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
        Per-predictor coefficients
      </p>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left text-xs text-slate-500 border-b border-slate-200">
            <th className="pb-2 pr-4 font-medium">Predictor</th>
            <th className="pb-2 pr-4 font-medium">{estimateLabel}</th>
            <th className="pb-2 pr-4 font-medium">95% CI</th>
            <th className="pb-2 font-medium">p-value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-slate-100 last:border-0">
              <td className="py-2 pr-4 font-medium text-slate-800">{row.predictor}</td>
              <td className="py-2 pr-4 text-slate-700">{fmt(row.estimate, 3)}</td>
              <td className="py-2 pr-4 text-slate-700">
                [{fmt(row.ci_lower, 3)}, {fmt(row.ci_upper, 3)}]
              </td>
              <td className={`py-2 font-medium ${row.p_value < 0.05 ? 'text-green-700' : 'text-slate-700'}`}>
                {fmtP(row.p_value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

interface Props { report: Report; sessionId: string }

export default function ResultsView({ report, sessionId }: Props) {
  const { test_result: tr, study_design: sd, data_profile: dp } = report

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* ── Left panel ────────────────────────────────────────────────── */}
      <aside className="lg:w-72 shrink-0 space-y-4">

        {/* Data profile */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Data profile</p>
          <div className="space-y-1.5 text-sm">
            <p><span className="text-slate-500">N =</span> <strong>{dp.variables[0]?.n_observations ?? '—'}</strong></p>
            <p><span className="text-slate-500">Variables:</span> <strong>{dp.variables.length}</strong></p>
            {dp.notes.map((n, i) => (
              <p key={i} className="text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded">⚠ {n}</p>
            ))}
          </div>
        </div>

        {/* Test selected */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Test selected</p>
          <p className="font-bold text-slate-900">{TEST_LABELS[tr.test_used] ?? tr.test_used}</p>
          <details className="mt-2">
            <summary className="text-xs text-brand cursor-pointer hover:underline">Why this test?</summary>
            <p className="mt-2 text-xs text-slate-600 leading-relaxed">
              {report.methods_text.split('.')[0]}.
            </p>
          </details>
        </div>

        {/* Assumption checks */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Assumption checks</p>
          <div className="space-y-2.5">
            {tr.assumption_checks.map((ac, i) => (
              <div key={i}>
                <div className="flex items-start justify-between gap-2">
                  <p className="text-xs font-medium text-slate-700">{ac.assumption_name}</p>
                  <AssumptionBadge status={ac.status} />
                </div>
                <p className="text-xs text-slate-500 mt-0.5">{ac.note}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Caveats */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Caveats</p>
          <div className="space-y-3">
            {report.caveats.map((c, i) => (
              <div key={i}>
                <div className="flex items-center gap-2 mb-1">
                  <CaveatBadge severity={c.severity} />
                </div>
                <p className="text-xs text-slate-700">{c.message}</p>
                <p className="text-xs text-slate-500 mt-0.5 italic">{c.recommendation}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Study design */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Study design</p>
          <div className="space-y-1.5 text-xs text-slate-600">
            <p><span className="text-slate-400">Type:</span> {sd.design_type.replace('_', ' ')}</p>
            <p><span className="text-slate-400">Measurement:</span> {sd.measurement_type.replace('_', ' ')}</p>
            <p><span className="text-slate-400">Randomised:</span> {sd.is_randomized ? 'Yes' : 'No'}</p>
          </div>
        </div>
      </aside>

      {/* ── Right panel ───────────────────────────────────────────────── */}
      <div className="flex-1 space-y-5">

        {/* Primary result */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Primary result</p>
              <h3 className="text-2xl font-bold text-slate-900">{TEST_LABELS[tr.test_used] ?? tr.test_used}</h3>
            </div>
            <SignificanceBadge significant={tr.is_significant} />
          </div>
          <div className="mt-4 flex flex-wrap gap-6 text-sm">
            <div>
              <p className="text-slate-500 text-xs">Statistic</p>
              <p className="text-xl font-bold text-slate-900">
                {STAT_LABELS[tr.test_used] ?? 'stat'} = {fmt(tr.statistic, 2)}
              </p>
            </div>
            {tr.degrees_of_freedom != null && (
              <div>
                <p className="text-slate-500 text-xs">df</p>
                <p className="text-xl font-bold text-slate-900">{fmt(tr.degrees_of_freedom, 1)}</p>
              </div>
            )}
            <div>
              <p className="text-slate-500 text-xs">p-value</p>
              <p className={`text-xl font-bold ${tr.is_significant ? 'text-green-700' : 'text-slate-900'}`}>
                {tr.p_value < 0.001 ? '< 0.001' : fmt(tr.p_value)}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">95% CI</p>
              <p className="text-xl font-bold text-slate-900">[{fmt(tr.confidence_interval[0], 1)}, {fmt(tr.confidence_interval[1], 1)}]</p>
            </div>
          </div>
        </div>

        {/* Effect size */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Effect size</p>
          <div className="flex items-baseline gap-3 flex-wrap">
            <p className="text-2xl font-bold text-slate-900">
              {tr.effect_size.measure_name} = {fmt(tr.effect_size.value, 2)}
            </p>
            <span className="px-2.5 py-0.5 bg-indigo-100 text-brand rounded-full text-sm font-medium capitalize">
              {tr.effect_size.interpretation}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1.5">
            95% CI [{fmt(tr.effect_size.ci_lower, 2)}, {fmt(tr.effect_size.ci_upper, 2)}]
          </p>
          {tr.notes.map((n, i) => (
            <p key={i} className="mt-3 text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg">{n}</p>
          ))}
        </div>

        {/* Coefficient table (regression tests only) */}
        {tr.coefficient_table && tr.coefficient_table.length > 0 && (
          <CoefficientTable rows={tr.coefficient_table} testUsed={tr.test_used} />
        )}

        {/* Plain-language summary */}
        <div className="bg-indigo-50 rounded-xl border border-indigo-200 p-6">
          <p className="text-xs font-semibold text-brand uppercase tracking-wide mb-2">Plain-language summary</p>
          <div className="text-slate-800 leading-relaxed prose prose-sm max-w-none prose-p:mb-2 prose-p:last:mb-0">
            <ReactMarkdown>{report.plain_language_summary}</ReactMarkdown>
          </div>
        </div>

        {/* Plots */}
        {report.plots.length > 0 && <PlotViewer plots={report.plots} />}

        {/* Methods text */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Methods text</p>
            <CopyButton text={report.methods_text} />
          </div>
          <p className="text-sm text-slate-700 leading-relaxed font-mono bg-slate-50 p-4 rounded-lg">
            {report.methods_text}
          </p>
        </div>

        {/* Export */}
        <div className="flex flex-wrap gap-3 justify-end">
          <button
            onClick={() => {
              const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = `hta-report-${sessionId.slice(0, 8)}.json`
              a.click()
              URL.revokeObjectURL(url)
            }}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 text-slate-700 rounded-xl text-sm font-medium hover:bg-slate-50 transition-colors"
          >
            ⬇ Download JSON
          </button>
          <a
            href={exportHtmlUrl(sessionId)}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-brand text-white rounded-xl text-sm font-medium hover:bg-brand-dark transition-colors"
          >
            ⬇ Download Report (HTML)
          </a>
        </div>
      </div>
    </div>
  )
}
