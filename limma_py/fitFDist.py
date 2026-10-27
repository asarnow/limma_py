import numpy as np
from scipy.special import digamma, polygamma, gammaln
from scipy.linalg import lstsq, qr
from scipy.optimize import minimize_scalar
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


def _subset_unit_vector(x, i):
    """Subset a vector, treating a length-1 vector as a vector of equal values.

    Faithful port of R limma's ``.subsetUnitVector``.
    """
    x = np.asarray(x)
    if x.size == 1:
        return x
    return x[i]


def fitFDistUnequalDF1(x, df1, covariate=None, span=None, robust=False, prior_weights=None):
    """Robust estimation of the parameters of a scaled F-distribution given df1.

    This is the modern (``legacy=False``) hyperparameter estimator used by
    ``squeezeVar``/``eBayes``. Unlike the moment-based :func:`fitFDist`, it gives
    special attention to the possibility that ``df1`` varies substantially between
    observations (e.g. missing-value data with per-gene residual df). It estimates
    the prior scale and ``df2`` by maximizing the profile log-likelihood of a
    scaled F-distribution.

    Faithful port of R limma's ``fitFDistUnequalDF1`` (Gordon Smyth and Lizhong
    Chen, ``R/fitFDistUnequalDF1.R``).

    Parameters
    ----------
    x : array-like
        Sample variances (one per gene).
    df1 : float or array-like
        Numerator degrees of freedom (per-gene residual df). May be a scalar or a
        vector of length ``len(x)``.
    covariate : array-like, optional
        Covariate for a variance trend (e.g. average log-expression). Not yet
        supported in this estimator; use ``legacy=True`` for trended fits.
    span : float, optional
        Span for the lowess trend fit (only relevant when ``covariate`` is given).
    robust : bool, optional
        Whether to down-weight outliers. Not yet implemented for this estimator.
    prior_weights : array-like, optional
        Non-negative prior weights, one per gene.

    Returns
    -------
    dict
        ``{'scale': s20, 'df2': df2}`` (non-robust), or additionally
        ``'df2_outlier'``/``'df2_shrunk'`` for the robust path.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size

    # Check df1. Can be a unit (length-1) vector.
    df1 = np.asarray(df1, dtype=np.float64)
    if df1.size not in (1, n):
        raise ValueError("x and df1 are different lengths")
    if np.any(np.isnan(df1)):
        raise ValueError("NA df1 values")
    # Recycle df1 to length n so per-gene masking below is well defined.
    df1 = np.full(n, float(df1)) if df1.size == 1 else df1.astype(np.float64).copy()

    # Check covariate
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if covariate.size != n:
            raise ValueError("x and covariate are different lengths")
        if np.any(np.isnan(covariate)):
            raise ValueError("covariate contains NA values")

    # Check prior.weights
    if prior_weights is not None:
        prior_weights = np.asarray(prior_weights, dtype=np.float64).copy()
        if prior_weights.size != n:
            raise ValueError("x and prior.weights are different lengths")
        m = np.min(prior_weights)
        if np.isnan(m):
            raise ValueError("prior.weights contain NA values")
        if m < 0:
            raise ValueError("prior.weights are negative")

    # Check for NAs in x: zero out the value and give it zero prior weight.
    if np.any(np.isnan(x)):
        i = np.isnan(x)
        if prior_weights is None:
            prior_weights = (~i).astype(np.float64)
        else:
            prior_weights[i] = 0.0
        x[i] = 0.0

    # Treat small df1 values as un-informative.
    if np.min(df1) < 0.01:
        i = df1 < 0.01
        if prior_weights is None:
            prior_weights = (~i).astype(np.float64)
        else:
            prior_weights[i] = 0.0
        df1[i] = 1.0

    PriorWeights = prior_weights is not None

    # Check there are some informative x values.
    i = x > 0
    if PriorWeights:
        i[prior_weights == 0] = False
    n_informative = int(np.sum(i))
    if n_informative < 2:
        return {"scale": np.nan, "df2": np.nan}
    if n_informative == 2:
        covariate = None
        robust = False
        prior_weights = None
        PriorWeights = False

    # Avoid exactly zero x values for moment estimation.
    m = np.median(x[i])
    xpos = np.maximum(x, 1e-12 * m)

    # Work with log(F).
    z = np.log(xpos)

    # Average log(F) adjusted for d1. NB: this module's ``logmdigamma`` is
    # digamma(a)-log(a), i.e. the negative of R limma's ``logmdigamma`` (which is
    # log(a)-digamma(a)), so the signs here are flipped relative to the R source.
    d1 = df1 / 2.0
    e = z - logmdigamma(d1)
    w = 1.0 / polygamma(1, d1)  # trigamma
    if PriorWeights:
        w = w * prior_weights
    if covariate is None:
        emean = np.sum(w * e) / np.sum(w)
    else:
        raise NotImplementedError(
            "Variance-trend (covariate) fitting is not implemented for the "
            "unequal-df estimator. Use eBayes(..., legacy=True) for a trended "
            "fit on missing-value data."
        )

    # Log-likelihood function (parameterized so par in (0, 1) maps to d2 in (0, Inf)).
    d1x = d1 * xpos
    if PriorWeights:
        def minus_twice_log_lik(par):
            d2 = par / (1.0 - par)
            d2s20 = d2 * np.exp(emean + logmdigamma(d2))
            return -2.0 * np.sum(prior_weights * (
                -(d1 + d2) * np.log1p(d1x / d2s20)
                - d1 * np.log(d2s20)
                + gammaln(d1 + d2) - gammaln(d2)))
    else:
        def minus_twice_log_lik(par):
            d2 = par / (1.0 - par)
            d2s20 = d2 * np.exp(emean + logmdigamma(d2))
            return -2.0 * np.sum(
                -(d1 + d2) * np.log1p(d1x / d2s20)
                - d1 * np.log(d2s20)
                + gammaln(d1 + d2) - gammaln(d2))

    # Optimization over par in (1/2, 0.9998), matching R's optimize().
    res = minimize_scalar(minus_twice_log_lik, bounds=(0.5, 0.9998), method="bounded")
    par_min = res.x
    d2 = par_min / (1.0 - par_min)
    s20 = np.exp(emean + logmdigamma(d2))

    # Finish here if robust=False.
    if not robust:
        return {"scale": s20, "df2": 2.0 * d2}

    raise NotImplementedError(
        "Robust estimation (fitFDistUnequalDF1 robust path) is not implemented; "
        "call eBayes/squeezeVar with robust=False."
    )

