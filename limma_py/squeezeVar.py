import numpy as np
from .fitFDist import fitFDist, fitFDistUnequalDF1


def squeezeVar(var, df, covariate=None, span=None, robust=False, winsor_tail_p=(0.05, 0.1), legacy=None):
    var = np.asarray(var, dtype=float)  # Ensure float dtype
    n = var.size  # Number of genes

    # Error when there are no observations
    if n == 0:
        raise ValueError("var is empty")

    # With fewer than 3 samples empirical Bayes provides no benefit; return the raw variance directly
    if n < 3:
        return {"var_post": var,
                "var_prior": var,
                "df_prior": 0.0}

    df = np.asarray(df, dtype=float)

    # Handle invalid degrees of freedom (df == 0)
    if df.size > 1:  # Equivalent to R's length(df) > 1
        var[df == 0] = 0.0  # Zero out var at every position where df == 0

    # span is only implemented for the new hyperparameter function
    if span is not None:
        legacy = False

    # Assume df is already an np.ndarray(dtype=float)
    if legacy is None:  # Corresponds to R's is.null(legacy)
        dfp = df[df > 0]  # Keep only valid degrees of freedom
        if dfp.size == 0:
            # Edge case: no valid degrees of freedom; fall through to the new method
            legacy = False
        else:
            legacy = bool(np.min(dfp) == np.max(dfp))

    if legacy:  # Legacy method (equal df)
        if robust:  # Robust estimation (outlier-resistant)
            raise NotImplementedError(
                "Handling unequal degrees of freedom (fitFDistRobust) not implemented."
            )

        else:
            fit = fitFDist(var, df1=df, covariate=covariate)
            df_prior = fit['df2']
    else:  # New method (unequal df)
        fit = fitFDistUnequalDF1(var, df1=df, covariate=covariate, span=span, robust=robust)
        df_prior = fit.get('df2_shrunk')
        if df_prior is None:
            df_prior = fit['df2']

    if np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior df")

    var_post = _squeezeVar(var=var, df=df,
                            var_prior=fit['scale'],
                            df_prior=df_prior)

    return {
        "df_prior": df_prior,
        "var_prior": fit["scale"],
        "var_post": var_post
    }

def _squeezeVar(var, df, var_prior, df_prior):
    m = np.max(df_prior)  # Maximum prior degrees of freedom
    if np.isfinite(m):
        return (df * var + df_prior * var_prior) / (df + df_prior)

    n = var.size  # Number of genes
    if np.asarray(var_prior).size == n:
        var_post = var_prior
    else:
        var_post = np.resize(var_prior, n)

    m = np.min(df_prior)  # Minimum prior degrees of freedom
    if m > 1e100:
        return var_post  # Treated as infinite; return directly

    # Assume var, df, df_prior, and var_post are np.ndarrays of matching (or broadcastable) length
    i = np.isfinite(df_prior)  # Boolean mask: genes with finite prior df
    if df.size > 1:  # Corresponds to R's length(df) > 1
        df = df[i]  # Extract the matching residual df
    df_prior_finite = df_prior[i]  # Extract the matching prior df

    # Apply the shrinkage formula only to genes with finite df
    var_post[i] = (df * var[i] + df_prior_finite * var_post[i]) / (df + df_prior_finite)

    return var_post