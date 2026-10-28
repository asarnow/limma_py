import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


def contrasts_fit(
        fit: dict,
        contrasts: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        coefficients: Optional[Union[list, np.ndarray]] = None
) -> dict:
    """
    Replicates the R function limma::contrasts.fit(), extracting results of specified group contrasts from linear model fitting results.

    Parameter Description:
    ----------
    fit : dict
        Dictionary of linear model fitting results, must contain the following keys (corresponding to core fields of limma::lmFit() output):
        - "coefficients": Gene×coefficient expression coefficient matrix (np.ndarray or pd.DataFrame)
        - "stdev_unscaled": Gene×coefficient unscaled standard deviation matrix (np.ndarray or pd.DataFrame)
        - "cov_coefficients": Covariance matrix of coefficients (np.ndarray or pd.DataFrame, optional)
        - "pivot": Column indices of estimable coefficients (np.ndarray, required only when design matrix is singular)
        - Other fields (e.g., "t", "p.value", etc.) will be automatically removed (to avoid interference from eBayes results)

    contrasts : np.ndarray or pd.DataFrame, optional
        Contrast matrix (number of rows = number of coefficients in fit, number of columns = number of contrasts), to be passed mutually exclusively with coefficients

    coefficients : list or np.ndarray, optional
        Column indices/names of coefficients to retain, to be passed mutually exclusively with contrasts (for simple coefficient filtering, not group contrasts)

    Return Value:
    ----------
    fit : dict
        Updated fitting result dictionary with core fields adjusted according to the contrast matrix:
        - "coefficients": Contrasted coefficient matrix (gene×contrast)
        - "stdev_unscaled": Contrasted unscaled standard deviation matrix
        - "cov_coefficients": Contrasted coefficient covariance matrix
        - "contrasts": Passed contrast matrix (recording contrast rules)
        - Other irrelevant fields (e.g., original "t", "p.value") have been removed
    """
    # --------------------------
    # 1. Parameter validity check (core: exactly one of contrasts or coefficients must be specified)
    # --------------------------
    if (contrasts is None) == (coefficients is None):
        raise ValueError("Must specify exactly one of 'contrasts' or 'coefficients'")

    # Remove possible old statistics from fit (to avoid interference from eBayes results)
    for key in ["t", "p.value", "lods", "F", "F.p.value"]:
        if key in fit:
            del fit[key]

    # Extract core matrices from fit (unify to np.ndarray, retain dimension names for subsequent recovery)
    coef = fit["coefficients"].values if isinstance(fit["coefficients"], pd.DataFrame) else fit["coefficients"]
    stdev_unscaled = fit["stdev_unscaled"].values if isinstance(fit["stdev_unscaled"], pd.DataFrame) else fit[
        "stdev_unscaled"]
    coef_names = fit["coefficients"].columns.tolist() if isinstance(fit["coefficients"], pd.DataFrame) else None  # Coefficient column names
    gene_names = fit["coefficients"].index.tolist() if isinstance(fit["coefficients"], pd.DataFrame) else None  # Gene names

    n_genes, n_coef = coef.shape  # Number of genes, original number of coefficients

    # --------------------------
    # 2. Branch 1: Coefficient filtering only (coefficients parameter, no group contrasts)
    # --------------------------
    if coefficients is not None:
        # Process coefficient indices/names (supports column names or integer indices)
        if isinstance(coefficients, list) and isinstance(coef_names, list):
            coef_idx = [coef_names.index(name) for name in coefficients if name in coef_names]
        else:
            coef_idx = np.array(coefficients, dtype=int)

        # Filter coefficient matrix and standard deviation matrix
        fit["coefficients"] = coef[:, coef_idx]
        fit["stdev_unscaled"] = stdev_unscaled[:, coef_idx]

        # If covariance matrix exists, filter synchronously
        if "cov_coefficients" in fit:
            cov_coef = fit["cov_coefficients"].values if isinstance(fit["cov_coefficients"], pd.DataFrame) else fit[
                "cov_coefficients"]
            fit["cov_coefficients"] = cov_coef[np.ix_(coef_idx, coef_idx)]

        # Restore DataFrame format (if original format was DataFrame)
        if coef_names is not None:
            fit["coefficients"] = pd.DataFrame(fit["coefficients"], index=gene_names,
                                               columns=[coef_names[i] for i in coef_idx])
            fit["stdev_unscaled"] = pd.DataFrame(fit["stdev_unscaled"], index=gene_names,
                                                 columns=[coef_names[i] for i in coef_idx])
            if "cov_coefficients" in fit:
                fit["cov_coefficients"] = pd.DataFrame(fit["cov_coefficients"], index=[coef_names[i] for i in coef_idx],
                                                       columns=[coef_names[i] for i in coef_idx])

        return fit

    # --------------------------
    # 3. Branch 2: Group contrasts (contrasts parameter, core logic)
    # --------------------------
    # 3.1 Contrast matrix preprocessing (unify to np.ndarray, check dimensions)
    if isinstance(contrasts, pd.DataFrame):
        contrasts = contrasts.values
    contrasts = np.asmatrix(contrasts, dtype=np.float64)

    if contrasts.shape[0] != n_coef:
        raise ValueError(
            f"Number of rows of contrast matrix ({contrasts.shape[0]}) must match number of coefficients in fit ({n_coef})")

    # Check if contrast matrix row names match coefficient column names (if names exist)
    if coef_names is not None and hasattr(contrasts, "rownames") and contrasts.rownames is not None:
        contrast_rownames = contrasts.rownames
        # Unify intercept names (may be "(Intercept)" in R, unified to "Intercept" in Python)
        if contrast_rownames[0] == "(Intercept)":
            contrast_rownames[0] = "Intercept"
        if coef_names[0] == "(Intercept)":
            coef_names[0] = "Intercept"
        # Warn if names do not match
        if not np.array_equal(contrast_rownames, coef_names):
            print(f"Warning: row names of contrasts don't match col names of coefficients")

    # Record contrast matrix in fit
    fit["contrasts"] = contrasts

    # 3.2 Handle singular design matrix (only when covariance matrix rank < number of coefficients)
    if "cov_coefficients" in fit:
        cov_coef = fit["cov_coefficients"].values if isinstance(fit["cov_coefficients"], pd.DataFrame) else fit[
            "cov_coefficients"]
        cov_coef = np.asmatrix(cov_coef, dtype=np.float64)
        r = np.linalg.matrix_rank(cov_coef)  # Rank of covariance matrix (number of estimable coefficients)
    else:
        r = n_coef  # Default all coefficients are estimable when no covariance matrix exists

    if r < n_coef:
        if "pivot" not in fit:
            raise ValueError("cov_coefficients not full rank but 'pivot' not found in fit")
        # Extract indices of estimable coefficients
        estimable_idx = fit["pivot"][:r]
        # Check if contrast matrix involves non-estimable coefficients
        if np.any(contrasts[~np.isin(np.arange(n_coef), estimable_idx), :] != 0):
            raise ValueError("trying to take contrast of non-estimable coefficient")
        # Retain only parts of estimable coefficients
        contrasts = contrasts[estimable_idx, :]
        coef = coef[:, estimable_idx]
        stdev_unscaled = stdev_unscaled[:, estimable_idx]
        if "cov_coefficients" in fit:
            cov_coef = cov_coef[np.ix_(estimable_idx, estimable_idx)]
        n_coef = r  # Update number of coefficients to estimable count

    # 3.3 Remove coefficients with no contribution in contrasts (rows with sum 0, optimize computation speed)
    zero_contrast_rows = np.where(np.sum(np.abs(contrasts), axis=1) == 0)[0]
    if len(zero_contrast_rows) > 0:
        # Retain coefficients with non-zero rows
        keep_idx = np.delete(np.arange(n_coef), zero_contrast_rows)
        contrasts = contrasts[keep_idx, :]
        coef = coef[:, keep_idx]
        stdev_unscaled = stdev_unscaled[:, keep_idx]
        if "cov_coefficients" in fit:
            cov_coef = cov_coef[np.ix_(keep_idx, keep_idx)]
        n_coef = len(keep_idx)  # Update number of coefficients

    # 3.4 Handle NA coefficients (mask with large standard deviation to avoid contrast calculation errors)
    na_mask = np.isnan(coef)
    if np.any(na_mask):
        coef[na_mask] = 0  # Set NA coefficients to 0
        stdev_unscaled[na_mask] = 1e30  # Set standard deviation corresponding to NA to a large value (restore NA later)

    # --------------------------
    # 4. Core calculation: Update coefficients and standard deviations based on contrast matrix
    # --------------------------
    # 4.1 Calculate contrasted coefficient matrix (gene×contrast)
    fit_coef = coef @ contrasts  # Matrix multiplication: gene coefficients × contrast rules

    # 4.2 Calculate contrasted covariance matrix
    if "cov_coefficients" in fit:
        # Covariance matrix update: contrasts^T × cov_coef × contrasts
        fit_cov_coef = contrasts.T @ cov_coef @ contrasts
    else:
        # Assume coefficients are orthogonal when no covariance matrix exists (warning)
        print("Warning: cov_coefficients not found in fit - assuming coefficients are orthogonal")
        var_coef = np.mean(stdev_unscaled **2, axis=0)  # Variance of each coefficient (average unscaled variance)
        fit_cov_coef = np.diag(np.diag(var_coef @ np.square(contrasts)))  # Covariance matrix under orthogonal assumption
    fit["cov_coefficients"] = fit_cov_coef

    # 4.3 Calculate contrasted unscaled standard deviation
    # As in R, orthogonality is judged on the correlation of the original coefficients
    # (not of the contrasts), and the non-orthogonal path uses its Cholesky factor.
    if "cov_coefficients" in fit and n_coef >= 2:
        sd = np.sqrt(np.diag(np.asarray(cov_coef)))
        cormatrix = np.asarray(cov_coef) / np.outer(sd, sd)
        is_orthog = np.sum(np.abs(cormatrix[np.tril_indices_from(cormatrix, k=-1)])) < 1e-12
    else:
        is_orthog = True

    contrasts_arr = np.asarray(contrasts)
    if is_orthog:
        # Orthogonal design: standard deviation = sqrt(sum of (coefficient standard deviation² × contrast²))
        fit_stdev_unscaled = np.sqrt(stdev_unscaled ** 2 @ np.square(contrasts_arr))
    else:
        R = np.linalg.cholesky(cormatrix).T  # upper triangular, as R's chol()
        n_contrasts = contrasts_arr.shape[1]
        fit_stdev_unscaled = np.zeros((n_genes, n_contrasts))
        for i in range(n_genes):
            ruc = R @ (stdev_unscaled[i, :][:, None] * contrasts_arr)
            fit_stdev_unscaled[i, :] = np.sqrt(np.sum(ruc ** 2, axis=0))

    # --------------------------
    # 5. Restore NA coefficients (restore coefficients corresponding to standard deviations set to large values to NA)
    # --------------------------
    if np.any(na_mask):
        # Find positions needing NA restoration (standard deviation > 1e20)
        na_restore_mask = fit_stdev_unscaled > 1e20
        fit_coef[na_restore_mask] = np.nan
        fit_stdev_unscaled[na_restore_mask] = np.nan

    # --------------------------
    # 6. Result formatting (restore DataFrame format, retain names)
    # --------------------------
    contrast_names = contrasts.columns.tolist() if hasattr(contrasts, "columns") else [f"Contrast_{i + 1}" for i in
                                                                                       range(contrasts.shape[1])]
    if gene_names is not None and coef_names is not None:
        fit["coefficients"] = pd.DataFrame(fit_coef, index=gene_names, columns=contrast_names)
        fit["stdev_unscaled"] = pd.DataFrame(fit_stdev_unscaled, index=gene_names, columns=contrast_names)
        fit["cov_coefficients"] = pd.DataFrame(fit_cov_coef, index=contrast_names, columns=contrast_names)
    else:
        fit["coefficients"] = fit_coef
        fit["stdev_unscaled"] = fit_stdev_unscaled

    return fit
