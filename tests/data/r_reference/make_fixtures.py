import numpy as np, pandas as pd
rng = np.random.default_rng(2026)
# Fixture A: proteomics-like, 500 genes x 8 samples, ~15% genes w/ 1-3 random NaNs,
# heteroscedastic gene variances so the prior is non-trivial.
n = 500
sd = np.sqrt(0.5 * 8 / rng.chisquare(8, n))
X = rng.standard_normal((n, 8)) * sd[:, None] + rng.normal(20, 2, n)[:, None]
X[:40, 4:] += 1.5
for g in rng.choice(n, 75, replace=False):
    k = rng.integers(1, 4)
    X[g, rng.choice(8, k, replace=False)] = np.nan
# keep >=2 obs per group so every gene is estimable
for g in range(n):
    for grp in (slice(0, 4), slice(4, 8)):
        row = X[g, grp]
        if np.sum(np.isfinite(row)) < 2:
            idx = np.where(~np.isfinite(row))[0][0]
            row[idx] = 20.0 + rng.standard_normal()
pd.DataFrame(X, columns=[f"S{i}" for i in range(8)]).to_csv("fixA.csv", index=False)
# Fixture B: raw F-dist vector for fitFDistUnequalDF1 unit check
df1 = rng.choice([2., 3., 4., 5., 6., 8.], 3000)
from scipy.stats import f
x = 1.7 * f.rvs(df1, 7.0, random_state=rng)
x[:5] = 0.0; x[5:8] = np.nan; df1[8:12] = 0.0
pd.DataFrame({"x": x, "df1": df1}).to_csv("fixB.csv", index=False)
