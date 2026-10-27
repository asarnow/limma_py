"""Tests for the unequal-df (missing value) eBayes path.

These cover the ``squeezeVar`` -> ``fitFDistUnequalDF1`` route that is taken when
the expression matrix contains missing values, so residual degrees of freedom
vary between genes. See ``MISSING_VALUES_TODO.md``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as fdist

import limma_py as limma
from limma_py.fitFDist import fitFDist, fitFDistUnequalDF1


DESIGN = pd.DataFrame({
    "G1": [1, 1, 1, 1, 0, 0, 0, 0],
    "G2": [0, 0, 0, 0, 1, 1, 1, 1],
})


def _make_expr(seed=0, ngenes=200):
    rng = np.random.default_rng(seed)
    expr = pd.DataFrame(rng.standard_normal((ngenes, 8)))
    expr.iloc[:20, 4:] += 2.0  # genuine signal in the first 20 genes
    return expr


def _run(expr, legacy=None):
    fit = limma.lmFit(expr, DESIGN)
    fit = limma.contrasts_fit(fit, limma.make_contrasts("G2-G1", levels=DESIGN))
    return limma.eBayes(fit, legacy=legacy)


def _inject_missing(expr):
    em = expr.copy().astype(float)
    em.iloc[5, 1] = np.nan              # gene with df 5
    em.iloc[6, 1] = np.nan              # gene with df 4
    em.iloc[6, 5] = np.nan
    em.iloc[7, 2] = np.nan              # another gene with df 5
    return em


# ---------------------------------------------------------------------------
# fitFDistUnequalDF1 unit tests
# ---------------------------------------------------------------------------

def test_recovers_parameters_unequal_df():
    """Monte Carlo: estimator recovers the true scaled-F parameters."""
    rng = np.random.default_rng(42)
    n = 60000
    true_df2, true_s20 = 10.0, 2.0
    df1 = rng.choice([4.0, 5.0, 6.0], size=n)
    x = true_s20 * fdist.rvs(df1, true_df2, size=n, random_state=rng)
    out = fitFDistUnequalDF1(x, df1=df1)
    assert out["df2"] == pytest.approx(true_df2, rel=0.1)
    assert out["scale"] == pytest.approx(true_s20, rel=0.05)


def test_agrees_with_legacy_on_equal_df():
    """On equal-df data the new estimator should match legacy fitFDist."""
    rng = np.random.default_rng(7)
    n = 40000
    df1 = np.full(n, 5.0)
    x = 2.0 * fdist.rvs(df1, 8.0, size=n, random_state=rng)
    new = fitFDistUnequalDF1(x, df1=df1)
    leg = fitFDist(x, df1=df1)
    assert new["df2"] == pytest.approx(float(leg["df2"]), rel=0.05)
    assert float(new["scale"]) == pytest.approx(float(np.mean(leg["scale"])), rel=0.02)


def test_nan_and_zero_df_are_ignored():
    """NaN x values and tiny df1 must not poison the estimate."""
    rng = np.random.default_rng(1)
    n = 20000
    df1 = np.full(n, 5.0)
    x = 1.5 * fdist.rvs(df1, 12.0, size=n, random_state=rng)
    clean = fitFDistUnequalDF1(x.copy(), df1=df1.copy())

    x2 = x.copy()
    df1_2 = df1.copy()
    x2[:50] = np.nan          # missing variances
    df1_2[50:60] = 0.0        # zero residual df (treated as uninformative)
    out = fitFDistUnequalDF1(x2, df1=df1_2)
    assert out["df2"] == pytest.approx(clean["df2"], rel=0.05)


def test_too_few_informative_returns_nan():
    out = fitFDistUnequalDF1(np.array([0.0, 0.0, 1.0]), df1=np.array([5.0, 5.0, 5.0]))
    assert np.isnan(out["scale"]) and np.isnan(out["df2"])


def test_covariate_and_robust_not_implemented():
    x = np.abs(np.random.default_rng(0).standard_normal(50)) + 0.1
    df1 = np.full(50, 5.0)
    with pytest.raises(NotImplementedError):
        fitFDistUnequalDF1(x, df1=df1, covariate=np.arange(50.0))
    with pytest.raises(NotImplementedError):
        fitFDistUnequalDF1(x, df1=df1, robust=True)


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------

def test_missing_values_pipeline_runs_and_is_finite():
    """Default eBayes (legacy=None) must handle unequal df without raising."""
    em = _inject_missing(_make_expr())
    fit = _run(em)  # would previously raise NotImplementedError
    assert set(np.unique(fit["df_residual"])) == {4.0, 5.0, 6.0}
    assert np.all(np.isfinite(np.asarray(fit["t"])))
    assert np.all(np.isfinite(np.asarray(fit["p_value"])))


def test_legacy_true_path_runs_on_missing_values():
    """eBayes(legacy=True) routes unequal df through fitFDist (vector df1)."""
    em = _inject_missing(_make_expr())
    fit = _run(em, legacy=True)
    assert np.all(np.isfinite(np.asarray(fit["t"])))


def test_complete_case_uses_legacy_branch():
    """Equal-df data auto-selects the legacy branch, so the new code must not
    change complete-case results: default and legacy=True must be identical."""
    expr = _make_expr()
    assert set(np.unique(_run(expr)["df_residual"])) == {6.0}
    default = np.asarray(_run(expr)["t"])
    legacy = np.asarray(_run(expr, legacy=True)["t"])
    np.testing.assert_array_equal(default, legacy)


def test_edge_genes_propagate_nan_without_crash():
    """A group-wide missing gene and a df<=1 gene must yield NaN, not a crash."""
    expr = _make_expr()
    em = expr.copy().astype(float)
    em.iloc[100, 4:] = np.nan   # all of group 2 missing -> contrast not estimable
    em.iloc[101, 2:] = np.nan   # only 2 observations left
    fit = _run(em)
    t = np.asarray(fit["t"]).ravel()
    assert np.isnan(t[100])
    assert np.isnan(t[101])
    # Other genes are unaffected and finite.
    assert np.isfinite(t[0])
