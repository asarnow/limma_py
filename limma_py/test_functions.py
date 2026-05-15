import numpy as np
import pandas as pd
from scipy import stats



def classify_tests_f(object, cor_matrix=None, df=None, p_value=0.01, fstat_only=False):
    if isinstance(object, dict):
        tstat = np.asarray(object["t"], dtype=float)
    else:
        tstat = np.asarray(object, dtype=float)
    tstat = pd.DataFrame(tstat)

    if isinstance(object, list):
        if object.get("t") is None:
            raise ValueError("tstat cannot be extracted from object")

    if cor_matrix is None and object.get("cov_coefficients") is not None:
        cov = object["cov_coefficients"]
        cov = np.asarray(cov)
        n = cov.shape[0]
        i = np.arange(n) + n * np.arange(n)
        if np.min(cov.flat[i]) == 0:
            j = i[cov.flat[i] == 0]
            cov.flat[j] = 1
        std = np.sqrt(np.diag(cov))
        cor_matrix = cov / np.outer(std, std)

    if df is None and object.get("df_prior") is not None and object.get("df_residual") is not None:
        df_prior = object["df_prior"]
        if np.isscalar(df_prior):
            df_prior_val = df_prior  # Use the scalar as-is
        else:
            # Take the first element of the array and convert to a Python scalar
            df_prior_val = np.asarray(df_prior)[0].item()

        df_residual = object["df_residual"]
        if np.isscalar(df_residual):
            df_residual_val = df_residual  # Use the scalar as-is
        else:
            # Take the first element of the array and convert to a Python scalar
            df_residual_val = np.asarray(df_residual)[0].item()

        df = float(df_prior_val + df_residual_val)
    if tstat.ndim == 1:
        tstat = tstat.reshape(-1, 1)
    else:
        if tstat.ndim == 1:
            tstat = tstat.reshape(-1, 1)

    ngenes = tstat.shape[0]  # Number of genes (rows)
    ntests = tstat.shape[1]  # Number of tests (columns)

    if ntests == 1:
        if fstat_only:
            fstat = tstat ** 2
            # Use a Series/DataFrame to carry custom attributes
            fstat = pd.Series(fstat.to_numpy().ravel())
            fstat.attrs["df1"] = 1
            fstat.attrs["df2"] = df
            return fstat
        else:
            p = 2 * stats.t.sf(np.abs(tstat), df)
            sign_t = np.sign(tstat)
            flag = sign_t * (p < p_value)
            return pd.DataFrame(flag, columns=["test"])

    if cor_matrix is None:
        r = ntests  # With no correlation matrix, effective rank = number of tests
        Q = np.diag(np.ones(r)) / np.sqrt(r)  # Orthogonal transform (default: standardized diagonal matrix)
    else:
        E = np.linalg.eigh(cor_matrix)  # Eigendecomposition of the (symmetric) correlation matrix
        r = np.sum(E[0] / E[0][0] > 1e-08)  # Effective rank: count of eigenvalues whose ratio to the largest > 1e-08
        Q = np.dot(E[1][:, :r], np.diag(1 / np.sqrt(E[0][:r]))) / np.sqrt(r)  # Build the transform matrix

    if fstat_only:
        # Compute the F-statistic
        fstat = np.sum((tstat.values @ Q) ** 2, axis=1)  # F-statistic = sum of squares of transformed t-statistics
        # Attach attributes
        fstat = pd.Series(fstat, index=tstat.index)  # Convert to Series, preserving the index
        fstat.attrs["df1"] = r  # Numerator df = effective rank r
        fstat.attrs["df2"] = df  # Denominator df = df
        return fstat

    qF = stats.f.ppf(1 - p_value, r, df)
    if qF.size == 1:
        qF = np.full(ngenes, qF)

    # Initialize the result matrix with zeros
    result = np.zeros((ngenes, ntests))

    # If dimension names (row/column names) need to be preserved
    if tstat.index is not None and tstat.columns is not None:
        result = pd.DataFrame(result, index=tstat.index, columns=tstat.columns)

    for i in range(ngenes):  # Loop over genes
        x = tstat.iloc[i, :].values  # Extract t-statistics for this gene
        if np.any(np.isnan(x)):  # If any value is NaN, flag the gene as NaN
            result.iloc[i, :] = np.nan
        else:
            if np.dot(np.dot(Q, x), np.dot(Q, x)) > qF[i]:  # Sum of squares of transformed t > critical qF: globally significant
                ord = np.argsort(np.abs(x))[::-1]  # Sort by |t| descending
                result.iloc[i, ord[0]] = np.sign(x[ord[0]])  # Flag the most significant test with its sign (+/-1)

                # Step-down checks for the next-most-significant tests
                for j in range(1, ntests):
                    bigger = ord[:j]  # Already-flagged tests
                    x[bigger] = np.sign(x[bigger]) * np.abs(x[ord[j]])  # Set already-flagged t-values to the current test's |t| (preserving sign)
                    if np.dot(np.dot(Q, x), np.dot(Q, x)) > qF[i]:  # Still significant: flag the current test with its sign
                        result.iloc[i, ord[j]] = np.sign(x[ord[j]])
                    else:
                        break  # Not significant: stop flagging additional tests

    return result
