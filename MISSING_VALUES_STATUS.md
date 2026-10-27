# Handoff: missing-value (unequal residual df) eBayes path

**Audience:** the next engineer/agent picking up `limma_py`.
**Date:** 2026-06-25.
**TL;DR:** The headline feature is **done and tested**. The full pipeline
(`lmFit → contrasts_fit → eBayes → toptable`) now runs on expression matrices with
`NaN`s. The one remaining item is the gold-standard verification (numeric
cross-check against R `limma`), which is **deferred only because R is not installed
here** — not because of any known defect.

See `MISSING_VALUES_TODO.md` for the original design note and the rationale behind
choosing solution **(b)** (faithful port) over **(a)** (legacy moment estimator).

---

## What works now

When the expression matrix contains missing values, `lmFit`'s per-gene QR branch
produces a *per-gene* `df_residual` vector. Previously `squeezeVar` raised
`NotImplementedError` on unequal df. Now:

- **Default path (`legacy=None`)** routes unequal df through the new
  `fitFDistUnequalDF1` (profile-likelihood estimator, R's modern `legacy=FALSE`
  default). Produces finite `t` / `P.Value` / `adj.P.Val`.
- **`eBayes(fit, legacy=True)`** routes unequal df through the existing moment
  estimator `fitFDist` (R's pre-3.27 `legacy=TRUE` behavior). Also works.
- **Complete-case data (uniform df)** auto-selects the legacy branch, so existing
  results are **unchanged** (guarded by a test asserting default == legacy=True).
- **Edge cases** (a gene missing an entire group; a gene with df ≤ 1) propagate
  `NaN` to `toptable` without crashing.

## Files changed (uncommitted)

| File | Change |
|------|--------|
| `limma_py/fitFDist.py` | Added `fitFDistUnequalDF1(...)` + `_subset_unit_vector` helper. Imports now include `polygamma`, `gammaln`, `minimize_scalar`. |
| `limma_py/squeezeVar.py` | Signature gained `span=None`; unequal-df `else` branch now calls `fitFDistUnequalDF1` instead of raising. `span is not None` forces `legacy=False`. |
| `tests/test_missing_values.py` | **New.** 9 tests, all passing. |
| `MISSING_VALUES_TODO.md` | Status updated to mark §1(a) and §1(b) done. |

These are **not committed** — the user has not asked to commit. `git status` will
also show auto-generated `*.pyc`/egg-info churn; ignore it.

The new estimator is intentionally **not** in the public API (`limma_py/__init__.py`
exports only `lmFit, eBayes, toptable, classify_tests_f, contrasts_fit,
make_contrasts`), consistent with `fitFDist`/`squeezeVar` being internal. Import it
directly for tests: `from limma_py.fitFDist import fitFDistUnequalDF1`.

## ⚠️ The one gotcha you must not break

This module defines `logmdigamma(a) = digamma(a) - log(a)`, which is the **negative**
of R limma's `logmdigamma` (`log(a) - digamma(a)`). The R `fitFDistUnequalDF1`
source uses `+ logmdigamma` in three places; the Python port **flips those signs**
to match the local convention. The three flipped spots in `fitFDist.py`:

- `e = z - logmdigamma(d1)`  (line ~306)
- `d2s20 = d2 * np.exp(emean + logmdigamma(d2))`  (inside `minus_twice_log_lik`)
- `s20 = np.exp(emean + logmdigamma(d2))`  (line ~342)

Get this wrong and the estimator is **silently biased** (it returned df2≈6.5 vs a
true 10 before the fix). The bias is *not* caught by "does it run" tests — only by
the Monte Carlo recovery test. If you refactor `logmdigamma` or touch these lines,
re-run `test_recovers_parameters_unequal_df`.

## Tests (`tests/test_missing_values.py`) — 9 passing

Unit tests on `fitFDistUnequalDF1`:
- `test_recovers_parameters_unequal_df` — **the key correctness test.** 60k-draw
  Monte Carlo from a scaled-F with df1 ∈ {4,5,6}; asserts df2≈10 (rel 0.1),
  scale≈2 (rel 0.05).
- `test_agrees_with_legacy_on_equal_df` — on equal df, new estimator ≈ `fitFDist`.
- `test_nan_and_zero_df_are_ignored` — NaN x and df1<0.01 get zero weight.
- `test_too_few_informative_returns_nan` — <2 informative → `{nan, nan}`.
- `test_covariate_and_robust_not_implemented` — both raise `NotImplementedError`.

End-to-end pipeline tests:
- `test_missing_values_pipeline_runs_and_is_finite` — default path, df {4,5,6}.
- `test_legacy_true_path_runs_on_missing_values`.
- `test_complete_case_uses_legacy_branch` — regression guard, default == legacy.
- `test_edge_genes_propagate_nan_without_crash`.

Run: `python -m pytest tests/test_missing_values.py -q` → `9 passed`.

## ⭐ Outstanding verification (the real next step)

**Numeric cross-check against R `limma` on an identical fixture.** This is the
gold standard and the only meaningful gap. R was not installed in this environment
so it was substituted with Monte Carlo recovery + legacy-agreement tests. To close
it:

1. Build one fixture matrix with injected NaNs (reuse `_inject_missing` /
   `_make_expr` from the test file, or export a CSV).
2. In R: `lmFit` → `contrasts.fit` → `eBayes` → `topTable`, for **both**
   `eBayes(legacy=TRUE)` and the default.
3. Compare against Python output: `logFC`, `t`, `P.Value`, `df.prior` (`df_prior`),
   `s2.prior` (`var_prior` / `scale`). Expect agreement to within optimizer
   tolerance.
4. Add as a (skippable-if-no-R) test, e.g. pin R's numbers as constants so it runs
   without R in CI.

R source for reference:
`https://code.bioconductor.org/browse/limma/raw/devel/R/fitFDistUnequalDF1.R`.

## Not done (intentionally out of scope, both raise `NotImplementedError`)

- **Robust EB** (`robust=True`) — needs `fitFDistRobust`/`fitFDistRobustly`.
  Orthogonal to missing values. (`MISSING_VALUES_TODO.md` §2.)
- **Variance trend / covariate** under the unequal-df estimator — port the lowess
  trend from R into `fitFDistUnequalDF1`. Until then, trended fits on missing-value
  data must use `eBayes(..., legacy=True)`. (§3.)
