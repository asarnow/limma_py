# Implementation note: handle missing values (unequal residual df) in the eBayes path

**Audience:** an engineer/agent extending `limma_py`.
**Status:** ✅ **§1(b) implemented** (2026-06-23). The default `eBayes`/`squeezeVar`
path now handles unequal residual df via a faithful port of R limma's
`fitFDistUnequalDF1`. The `legacy=True` workaround (§1a) is also confirmed working
and covered by tests. Both are verified against R limma 3.68.5
(`tests/test_r_reference.py`, 2026-10-27). Still open: robust EB (§2) and the variance-trend/covariate
case under the unequal-df estimator (both raise `NotImplementedError`).

## What was implemented

- `limma_py/fitFDist.py`: new `fitFDistUnequalDF1(x, df1, covariate, span, robust,
  prior_weights)` — profile-likelihood estimator of the scaled-F prior `df2`/scale,
  ported from R `R/fitFDistUnequalDF1.R`. Non-robust, no-covariate path only.
- `limma_py/squeezeVar.py`: the unequal-df `else` branch now calls
  `fitFDistUnequalDF1` instead of raising; added a `span` argument (forces the new
  estimator when set), mirroring R.
- `tests/test_missing_values.py`: MC parameter-recovery, equal-df agreement with
  legacy `fitFDist`, NaN/zero-df robustness, the full missing-value pipeline,
  `legacy=True`, complete-case regression guard, and NaN-propagation edge cases.

> **Gotcha for future edits:** this module's `logmdigamma(a)` is defined as
> `digamma(a) - log(a)`, i.e. the **negative** of R limma's `logmdigamma`
> (`log(a) - digamma(a)`). The R `fitFDistUnequalDF1` source uses `+ logmdigamma`;
> the Python port flips those signs to match the local convention. Get this wrong
> and the estimator is silently biased (verified via the MC recovery test).

## Goal

Let the full pipeline (`lmFit` → `contrasts_fit` → `eBayes` → `toptable`) run on
expression matrices that contain `NaN`s (missing quantifications), which is the
norm for label-free proteomics / AP-MS data. Today the pipeline only works when
every feature is observed in every sample.

## Where it breaks today

`lmFit` already tolerates missing values. When the expression matrix has any
non-finite entry, `lm_series` takes the **per-gene QR branch**
(`limma_py/lmfit.py:245-321`): it fits each gene on just its observed samples,
so `df_residual` becomes a *per-gene* vector (e.g. 6, 5, 4 for genes missing 0,
1, 2 values with a 2-column design). That part is correct.

The pipeline then dies in `squeezeVar`:

- `limma_py/squeezeVar.py:26-46` — `legacy` is auto-selected as
  `legacy = (min(df>0) == max(df>0))`. With missing values the residual dfs
  differ, so `legacy` becomes `False`, and the `else` branch raises:

  ```python
  raise NotImplementedError(
      "Handling unequal degrees of freedom (fitFDistUnequalDF1) not implemented.")
  ```

So **the only hard blocker is `squeezeVar`’s unequal-df branch.** This was
observed live: complete-case data (uniform df) works; the same data with any
missing value raises the above.

## What already supports unequal df (do NOT rewrite these)

These were checked against the source and are already vector-df aware, so the
change is small and localized:

- `fitFDist` (`limma_py/fitFDist.py:49-180`) **already accepts a vector `df1`**:
  it filters invalid/zero df via the `ok` mask (`:63-91`), forms
  `e = log(x) - logmdigamma(df1/2)` per gene (`:99-101`), and applies the
  per-gene correction `evar -= mean(polygamma(1, df1/2))` (`:160`). This is R’s
  *legacy* moment estimator and it is valid for unequal df.
- `_ebayes` (`limma_py/ebayes.py:136-234`): `df_total = df_residual + df_prior`
  (`:187`), `df_pooled = sum(df_residual)` (`:190`), p-values from `df_total`
  (`:197-198`), lods kernel from `df_total` (`:228-232`) — all handle a vector
  `df_residual`.
- `_squeezeVar` (`limma_py/squeezeVar.py:61-85`): shrinkage formula is written
  for vector `df`/`df_prior`.
- `tmixture_vector` (`limma_py/ebayes.py:260-318`): already "unifies" unequal df
  to `MaxDF` (`:282-288`).
- `contrasts_fit` (`limma_py/contrasts.py`): uses per-gene `stdev_unscaled`
  (correct for the orthogonal means-model contrast) and masks NA coefficients
  (`:156-210`).

## Required changes

### 1. `squeezeVar`: stop crashing on unequal df (primary fix) — ✅ DONE (both (a) and (b))

Two ways to do this; ship at least (a), ideally (b). Both are now in place: (a) is
reachable via `eBayes(fit, legacy=True)`, and (b) is the default (`legacy=None`).

**(a) Minimal — route unequal df through the existing legacy `fitFDist`.**
`fitFDist` already does the right thing for vector `df1`. The unequal-df `else`
branch in `squeezeVar` (`:43-46`) should call it instead of raising:

```python
else:  # unequal df
    if robust:
        raise NotImplementedError("fitFDistRobust not implemented")  # see §2
    fit = fitFDist(var, df1=df, covariate=covariate)
    df_prior = fit['df2']
```

This reproduces limma’s pre-3.27 (`legacy=TRUE`) behavior and unblocks
missing-value data immediately. **Likely already reachable today by calling
`eBayes(fit, legacy=True)`** — traced through the code but *not* executed end to
end (kernel state was lost before the run). Confirm with the test in §4 before
documenting it as a supported workaround.

**(b) Faithful — implement `fitFDistUnequalDF1` for the default path.**
Modern limma (`legacy=FALSE`, the default) uses `fitFDistUnequalDF1`, which is
more accurate than the moment estimator when df vary because each gene’s
`log F` has a df-dependent variance/bias. Port it from R limma
(`R/fitFDist.R`, function `fitFDistUnequalDF1`): it computes per-gene weights
from `trigamma(df1/2)` and solves for the prior `df2`/scale by an iterative
profile likelihood rather than a single moment match. Keep the `covariate`
(variance-trend) support consistent with the existing `fitFDist`. Until this
exists, make `legacy=None` *default to the legacy estimator* (a) rather than
raising, and emit a one-time warning that the unequal-df EB estimator is the
legacy one.

### 2. Robust path (`fitFDistRobust`) — separate, optional

`squeezeVar(..., robust=True)` raises at `:36-38` regardless of df. This is
orthogonal to missing values; only implement if robust EB is needed. Port
`fitFDistRobust`/`fitFDistRobustly` from R limma. Track separately.

### 3. Edge cases to handle (not just "don’t crash")

- **`df_residual == 0` genes** (fit used up all residual df): `squeezeVar:22-23`
  already zeroes their `var`; `fitFDist`’s `ok` mask excludes them. Confirm
  `_squeezeVar` returns the prior (var_prior) for these and that downstream
  `t`/`p` are finite or cleanly `NaN`.
- **Genes unfit in a whole group** (e.g. all MERS samples missing): per-gene
  fit yields `NaN` sigma / a rank-deficient design for the contrast. Decide:
  either drop them before `eBayes` or guarantee `NaN` propagates to
  `toptable` (no crash, row kept as `NaN`). Recommended: document that callers
  should pre-filter to ≥ N replicates per group; the comparison this note came
  from used ≥3/4 per group.
- **Partial-NA coefficient warning**: `lmFit` (`lmfit.py:650-654`) only warns in
  a code path that the per-gene branch doesn’t reach; verify the warning still
  fires (or is intentionally dropped) for partially-estimable genes.

### 4. Tests to add (`limma_py/test_functions.py` or a new test module)

- Build a small genes×samples matrix, two groups, **inject missing values** so
  residual df vary (e.g. dfs of {6,5,4}). Assert:
  - `eBayes` (default `legacy=None`) returns finite `t`/`P.Value`/`adj.P.Val`
    with no `NotImplementedError`.
  - `eBayes(legacy=True)` runs and matches the §1(a) path.
  - Results match R `limma` on the same matrix within tolerance — run R’s
    `lmFit`/`eBayes`/`topTable` once and check `logFC`, `t`, `P.Value`,
    `df.prior`, `s2.prior`. Test both `legacy=TRUE` (vs §1a) and the default
    (vs §1b once implemented). ✅ Done: `tests/test_r_reference.py`.
- Keep a complete-case fixture as a regression guard (must stay identical).
- Edge fixtures: a `df_residual==0` gene and an all-missing-in-one-group gene.

## Immediate workaround (for users, today)

Force the legacy estimator and pre-filter to a complete-enough set:

```python
fit = limma_py.eBayes(fit, legacy=True)   # routes to fitFDist (vector df1)
```

Without it, restrict the input to complete cases (all samples observed), which
gives uniform df and uses the working equal-df path. The notebook that prompted
this note used complete cases (2683 of ~5400 protein groups) for exactly this
reason; the better outcome is §1 so the ≥3-per-group set (3125 groups) can be
used directly.

## References

- R limma source: `R/squeezeVar.R`, `R/fitFDist.R`
  (`fitFDist`, `fitFDistUnequalDF1`, `fitFDistRobustly`).
- Smyth (2004), *Stat. Appl. Genet. Mol. Biol.* 3(1) Art. 3 — original eBayes.
- Phipson et al. (2016), *Ann. Appl. Stat.* 10(2):946 — robust EB / trend.
