# Handoff: missing-value (unequal residual df) eBayes path

**Audience:** the next engineer/agent picking up `limma_py`.
**Date:** 2026-06-25.
**TL;DR:** The headline feature is **done and tested**. The full pipeline
(`lmFit → contrasts_fit → eBayes → toptable`) now runs on expression matrices with
`NaN`s, and is **verified against R limma 3.68.5** (agreement to ~1e-11 on both the
default and `legacy=True` paths; see `tests/test_r_reference.py`).

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

Run: `python -m pytest tests -q` → `13 passed` (9 here + 4 R-reference).

## ✅ Verification against R limma (done 2026-10-27)

`tests/test_r_reference.py` (4 tests) compares the Python output with pinned R
outputs in `tests/data/r_reference/`, so it runs without R:

- `fixA.csv`: 500 genes × 8 samples, 75 genes with 1–3 NaNs (df ∈ {3,4,5,6}).
  Full `lmFit → contrasts.fit → eBayes` for default and `legacy=TRUE`; checks
  `logFC`, `t`, `P.Value`, `B`, `s2.post`, `df.total`, `df.residual`, `df.prior`,
  `s2.prior` at `rtol=1e-8`. Observed max rel diff ~1e-11.
- `fixB.csv`: 3000 scaled-F variances with df1 ∈ {2..8}, plus zeros, NaNs and
  zero-df entries; `fitFDistUnequalDF1` and `fitFDist` directly.

To regenerate: `make_fixtures.py`, then `Rscript generate.R` in that directory
(an R env exists at `/opt/miniforge3/envs/rlimma`).

**Optimizer tolerance:** with scipy's default `xatol` (1e-5) the default path
differed from R by ~3e-5 in `df.prior`. Solving both to `tol=1e-12` gave
agreement to ~1e-7, which confirmed the port is correct and the gap was only
where each optimizer stopped. Both use Brent's method, so `fitFDistUnequalDF1`
now passes R's default `tol` (`eps**0.25`) and reproduces R's output almost
exactly.

R source for reference:
`https://code.bioconductor.org/browse/limma/raw/devel/R/fitFDistUnequalDF1.R`.

## Not done (intentionally out of scope, both raise `NotImplementedError`)

- **Robust EB** (`robust=True`) — needs `fitFDistRobust`/`fitFDistRobustly`.
  Orthogonal to missing values. (`MISSING_VALUES_TODO.md` §2.)
- **Variance trend / covariate** under the unequal-df estimator — port the lowess
  trend from R into `fitFDistUnequalDF1`. Until then, trended fits on missing-value
  data must use `eBayes(..., legacy=True)`. (§3.)
