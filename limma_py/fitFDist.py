import numpy as np
from scipy.special import digamma, polygamma
from scipy.linalg import lstsq, qr
from .utils.ns import ns


def logmdigamma(a):
    """Equivalent to R's logmdigamma: digamma(a) - log(a)"""
    return digamma(a) - np.log(a)


def trigammaInverse(x):
    """Exact reproduction of R's trigammaInverse function"""
    x = np.asarray(x)
    y = np.full_like(x, np.nan)
    valid = ~np.isnan(x) & (x >= 0)
    if not np.any(valid):
        return y

    x_valid = x[valid]
    y_valid = np.zeros_like(x_valid)

    # Handle large values (x > 1e7)
    large = x_valid > 1e7
    y_valid[large] = 1.0 / np.sqrt(x_valid[large])

    # Handle small values (x < 1e-6)
    small = x_valid < 1e-6
    y_valid[small] = 1.0 / x_valid[small]

    # Use Newton's method for intermediate values
    other = ~large & ~small
    if np.any(other):
        x_other = x_valid[other]
        y_other = 0.5 + 1.0 / x_other
        for _ in range(50):
            tri = polygamma(1, y_other)
            tri2 = polygamma(2, y_other)
            dif = tri * (1 - tri / x_other) / tri2
            y_other += dif
            if np.max(-dif / y_other) < 1e-8:
                break
        y_valid[other] = y_other

    y[valid] = y_valid
    return y


def fitFDist(x, df1, covariate=None):
    """Complete replication of R's fitFDist function"""
    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    df1 = np.asarray(df1, dtype=np.float64)
    n = len(x)

    # Input validation
    if n == 0:
        return {'scale': np.nan, 'df2': np.nan}
    if n == 1:
        return {'scale': x[0], 'df2': 0}

    # Check df1 validity
    ok = np.isfinite(df1) & (df1 > 1e-15)
    if df1.size == 1:
        if not ok:
            return {'scale': np.nan, 'df2': np.nan}
        ok = np.full(n, True)
    else:
        if len(df1) != n:
            raise ValueError("x and df1 have different lengths")

    # Filter invalid data
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = np.sum(ok)
    if nok == 1:
        return {'scale': np.full(n, x[ok][0]), 'df2': 0}
    if nok == 0:
        return {'scale': np.nan, 'df2': np.nan}

    # Record covariates for invalid data (for later prediction)
    covariate_notok = None
    if covariate is not None:
        covariate_notok = covariate[~ok].copy()

    # Filter data (keep only valid portion)
    x = x[ok]
    if df1.size > 1:
        df1 = df1[ok]
    if covariate is not None:
        covariate = covariate[ok]
    n = len(x)  # Update n to valid data count

    # Zero-value offset handling (identical to R)
    m = np.median(x)
    if m == 0:
        m = 1
    x = np.maximum(x, 1e-5 * m)

    # Core transformation: e = log(x) - logmdigamma(df1/2)
    z = np.log(x)
    e = z - logmdigamma(df1 / 2)

    # Calculate emean and evar (two cases: with/without covariate)
    if covariate is None:
        # No covariate: simple mean and sample variance
        emean = np.mean(e)
        evar = np.var(e, ddof=1)  # Unbiased variance (divide by n-1)
    else:
        # With covariate: spline fitting
        # Dynamically calculate spline degrees of freedom (consistent with R)
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))

        # If spline degrees of freedom insufficient, recursively call without covariate
        if splinedf < 2:
            result = fitFDist(x, df1, covariate=None)
            # Extend result to original length
            scale = np.full(n, result['scale'][0] if np.isscalar(result['scale']) else result['scale'][0])
            return {'scale': scale, 'df2': result['df2']}

        # Build spline design matrix
        try:
            design = ns(x=covariate, df=splinedf, intercept=True)
            design = np.asarray(design, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Spline design matrix construction failed: {e}")

        # Linear model fitting using QR decomposition (consistent with R's lm.fit)
        Q, R = qr(design, mode='full')
        rank = np.linalg.matrix_rank(R)

        # Calculate coefficients
        fit_coef = lstsq(R, Q.T @ e, cond=None)[0]
        emean_fitted = design @ fit_coef

        # Critical fix: account for the degrees of freedom when computing the residual variance
        residuals = e - emean_fitted
        residual_df = n - rank  # Residual degrees of freedom
        if residual_df > 0:
            evar = np.sum(residuals ** 2) / residual_df  # Divide by the residual df
        else:
            evar = 0.0

        # Predict emean for invalid data (align with original length)
        if covariate_notok is not None and len(covariate_notok) > 0:
            design_notok = ns(x=covariate_notok, df=splinedf, intercept=True)
            design_notok = np.asarray(design_notok, dtype=np.float64)
            emean_notok = design_notok @ fit_coef
            # Merge emean for valid and invalid data (restore original order)
            emean_full = np.empty(n, dtype=np.float64)
            emean_full[ok] = emean_fitted
            emean_full[~ok] = emean_notok
            emean = emean_full
        else:
            # Use fitted values directly if no invalid data (extend to original length)
            emean = np.full(n, np.nan)
            emean[ok] = emean_fitted

    # Adjust evar: subtract mean of trigamma(df1/2) (consistent with R)
    evar = evar - np.mean(polygamma(1, df1 / 2))

    # Estimate df2 and scale
    if evar > 0:
        df2 = 2 * trigammaInverse(evar)
        # Handle array case for df2 (ensure broadcast compatibility)
        if np.isscalar(df2):
            s20 = np.exp(emean + logmdigamma(df2 / 2))
        else:
            s20 = np.exp(emean + logmdigamma(df2 / 2))
    else:
        df2 = np.inf
        if covariate is None:
            # Without covariate, scale is mean of x (consistent with R)
            s20 = np.mean(x)
            s20 = np.full(n, s20)  # Extend to original length
        else:
            # With covariate, scale is exp(emean)
            s20 = np.exp(emean)

    return {'scale': s20, 'df2': df2}

