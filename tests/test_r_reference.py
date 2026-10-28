"""Numeric cross-check of the missing-value eBayes path against R limma.

Reference outputs in ``tests/data/r_reference`` were produced by ``generate.R``
(R 4.5.3, limma 3.68.5) on fixtures from ``make_fixtures.py``, so this test runs
without R. Both the default (``fitFDistUnequalDF1``) and ``legacy=True``
(``fitFDist``) estimators are checked.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import limma_py as limma
from limma_py.fitFDist import fitFDist, fitFDistUnequalDF1


DATA = Path(__file__).parent / "data" / "r_reference"
HYPER = pd.read_csv(DATA / "R_hyperparameters.csv").set_index(["fixture", "method"])

DESIGN = pd.DataFrame({
    "G1": [1, 1, 1, 1, 0, 0, 0, 0],
    "G2": [0, 0, 0, 0, 1, 1, 1, 1],
})

# Python fit key for each column of the R output.
COLUMNS = {
    "logFC": "coefficients",
    "t": "t",
    "P.Value": "p_value",
    "B": "lods",
    "s2.post": "s2_post",
    "df.total": "df_total",
    "df.residual": "df_residual",
}


def _fit_fixA(legacy):
    expr = pd.read_csv(DATA / "fixA.csv")
    fit = limma.lmFit(expr, DESIGN)
    fit = limma.contrasts_fit(fit, limma.make_contrasts("G2-G1", levels=DESIGN))
    return limma.eBayes(fit, legacy=legacy)


@pytest.mark.parametrize("method, legacy", [("default", None), ("legacy", True)])
def test_pipeline_matches_r(method, legacy):
    eb = _fit_fixA(legacy)
    ref = pd.read_csv(DATA / f"R_fixA_{method}.csv")
    assert len(np.unique(ref["df.residual"])) > 1  # fixture really has unequal df

    hyper = HYPER.loc[("fixA", method)]
    assert float(np.ravel(eb["df_prior"])[0]) == pytest.approx(hyper["df2"], rel=1e-8)
    assert float(np.ravel(eb["s2_prior"])[0]) == pytest.approx(hyper["scale"], rel=1e-8)

    for rcol, pykey in COLUMNS.items():
        np.testing.assert_allclose(
            np.asarray(eb[pykey], dtype=float).ravel(), ref[rcol].to_numpy(),
            rtol=1e-8, atol=1e-10, err_msg=rcol)


def test_fitFDistUnequalDF1_matches_r():
    fix = pd.read_csv(DATA / "fixB.csv")
    out = fitFDistUnequalDF1(fix["x"].to_numpy(), df1=fix["df1"].to_numpy())
    hyper = HYPER.loc[("fixB", "default")]
    assert out["df2"] == pytest.approx(hyper["df2"], rel=1e-8)
    assert out["scale"] == pytest.approx(hyper["scale"], rel=1e-8)


def test_fitFDist_matches_r():
    fix = pd.read_csv(DATA / "fixB.csv")
    out = fitFDist(fix["x"].to_numpy(), df1=fix["df1"].to_numpy())
    hyper = HYPER.loc[("fixB", "legacy")]
    assert float(out["df2"]) == pytest.approx(hyper["df2"], rel=1e-8)
    assert float(np.mean(out["scale"])) == pytest.approx(hyper["scale"], rel=1e-8)
