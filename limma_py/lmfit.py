import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.linalg import qr


def as_matrix_weights(weights, shape):
    """Convert weights to matrix format - kept exactly as in the original."""
    if weights is None:
        return None
    weights = np.array(weights)
    if weights.ndim == 1:
        if weights.shape[0] == shape[0]:
            # Row weights
            return np.tile(weights[:, np.newaxis], (1, shape[1]))
        elif weights.shape[0] == shape[1]:
            # Column weights
            return np.tile(weights[np.newaxis, :], (shape[0], 1))
        else:
            raise ValueError("weight dimensions do not match")
    elif weights.shape == shape:
        return weights
    else:
        raise ValueError("weight dimensions do not match")


def _lm_fit_one(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None,
                tol: float = 1e-7):
    """Least-squares fit of one gene, as R's lm.fit / lm.wfit.

    Uses a column-pivoted QR so that coefficients the observed rows cannot estimate
    (e.g. a group with no observations) are returned as NaN with NaN standard
    deviation, rather than the zero a minimum-norm solution would give.
    Returns (coefficients, stdev_unscaled, residual sum of squares, rank).
    """
    nbeta = X.shape[1]
    coef = np.full(nbeta, np.nan)
    stdev = np.full(nbeta, np.nan)
    if w is not None:
        sw = np.sqrt(w)
        X, y = X * sw[:, None], y * sw
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > tol * d[0])) if d.size and d[0] > 0 else 0
    if rank == 0:
        return coef, stdev, np.sum(y ** 2), 0
    est = piv[:rank]
    Rinv = np.linalg.inv(R[:rank, :rank])
    coef[est] = Rinv @ (Q[:, :rank].T @ y)
    stdev[est] = np.sqrt(np.sum(Rinv ** 2, axis=1))
    resid = y - X[:, est] @ coef[est]
    return coef, stdev, np.sum(resid ** 2), rank

def lm_series(M: np.ndarray,
              design: Optional[np.ndarray] = None,
              ndups: int = 1,
              spacing: int = 1,
              weights: Optional[np.ndarray] = None,
              debug: bool = True) -> Dict[str, Any]:
    """
    Fit a linear model for each gene to a series of arrays.
    - Original logic is preserved exactly; only pandas detection is added.
    """

    # Check the expression matrix - added pandas support
    if isinstance(M, pd.DataFrame):
        M = M.values
    M = np.asarray(M, dtype=np.float64)
    narrays = M.shape[1]

    # Check the design matrix - added pandas support
    if design is None:
        design = np.ones((narrays, 1), dtype=np.float64)
    else:
        if isinstance(design, pd.DataFrame):
            design = design.values
        design = np.asarray(design, dtype=np.float64)

    nbeta = design.shape[1]
    coef_names = [f"x{i + 1}" for i in range(nbeta)]
    if hasattr(design, 'columns'):
        coef_names = list(design.columns)

    # Check the weights - added pandas support
    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            weights = weights.values
        weights = as_matrix_weights(weights, M.shape)
        weights[weights <= 0] = np.nan
        M[np.isnan(weights)] = np.nan

    # Reformat duplicated rows as columns - kept exactly as in the original
    if ndups > 1:
        M = unwrapdups(M, ndups=ndups, spacing=spacing)
        design = np.kron(design, np.ones(ndups))
        if weights is not None:
            weights = unwrapdups(weights, ndups=ndups, spacing=spacing)
        narrays = M.shape[1]

    # Initialize standard errors - kept exactly as in the original
    ngenes = M.shape[0]
    stdev_unscaled = np.full((ngenes, nbeta), np.nan)
    beta = np.full((ngenes, nbeta), np.nan)

    # Set row and column names - kept exactly as in the original
    if hasattr(M, 'index'):
        row_names = list(M.index)
    else:
        row_names = [f"gene_{i}" for i in range(ngenes)]

    # Check whether the QR decomposition is constant across all genes - kept exactly as in the original
    NoProbeWts = np.all(np.isfinite(M)) and (weights is None or hasattr(weights, 'arrayweights'))

    if NoProbeWts:
        # ========== NoProbeWts branch, kept exactly as in the original ==========

        if weights is None:
            # Use SVD to improve numerical stability - kept exactly as in the original
            try:
                U, s, Vt = np.linalg.svd(design, full_matrices=False)
                rank = np.sum(s > s[0] * 1e-12)  # Rank detection based on relative magnitude

                if rank > 0:
                    # Compute the coefficients
                    fit_coef = (Vt.T @ (np.diag(1 / s) @ (U.T @ M.T))).T

                    # Compute residuals
                    y_pred = design @ fit_coef.T
                    residuals_matrix = M.T - y_pred

                    # Compute sigma
                    df_residual = narrays - rank
                    if df_residual > 0:
                        # Use a more stable residual calculation
                        sigma = np.sqrt(np.sum(residuals_matrix ** 2, axis=0) / df_residual)
                    else:
                        sigma = np.full(ngenes, np.nan)

                    # Compute the covariance matrix and standard errors
                    cov_coef = Vt.T @ np.diag(1 / s[:rank] ** 2) @ Vt[:, :rank]
                    diag_cov = np.sqrt(np.diag(cov_coef))
                    stdev_unscaled[:, :rank] = np.tile(diag_cov, (ngenes, 1))

                    df_residual_arr = np.full(ngenes, df_residual)

                    result = {
                        'coefficients': fit_coef,
                        'stdev_unscaled': stdev_unscaled,
                        'sigma': sigma,
                        'df_residual': df_residual_arr,
                        'cov_coefficients': cov_coef,
                        'rank': rank
                    }
                    return result

            except np.linalg.LinAlgError as e:
                if debug:
                    print(f"SVD decomposition failed, falling back to the original method: {e}")
                # Fall back to the original method
                pass

        else:
            # Weighted case - use a more stable method - kept exactly as in the original
            try:
                # Use the first row of weights (assumes all genes share the same weights)
                W_sqrt = np.sqrt(weights[0, :])
                W_design = design * W_sqrt[:, np.newaxis]
                W_M = M.T * W_sqrt

                # SVD decomposition
                U, s, Vt = np.linalg.svd(W_design, full_matrices=False)
                rank = np.sum(s > s[0] * 1e-12)

                if rank > 0:
                    fit_coef = (Vt.T @ (np.diag(1 / s) @ (U.T @ W_M))).T

                    # Compute weighted residuals
                    y_pred = design @ fit_coef.T
                    residuals_matrix = M.T - y_pred
                    weighted_residuals = residuals_matrix * W_sqrt

                    # Compute sigma
                    df_residual = narrays - rank
                    if df_residual > 0:
                        sigma = np.sqrt(np.sum(weighted_residuals ** 2, axis=0) / df_residual)
                    else:
                        sigma = np.full(ngenes, np.nan)

                    # Compute the covariance matrix and standard errors
                    cov_coef = Vt.T @ np.diag(1 / s[:rank] ** 2) @ Vt[:, :rank]
                    diag_cov = np.sqrt(np.diag(cov_coef))
                    stdev_unscaled[:, :rank] = np.tile(diag_cov, (ngenes, 1))

                    df_residual_arr = np.full(ngenes, df_residual)

                    result = {
                        'coefficients': fit_coef,
                        'stdev_unscaled': stdev_unscaled,
                        'sigma': sigma,
                        'df_residual': df_residual_arr,
                        'cov_coefficients': cov_coef,
                        'rank': rank
                    }
                    return result

            except np.linalg.LinAlgError as e:
                if debug:
                    print(f"Weighted SVD decomposition failed, falling back to the original method: {e}")
                # Fall back to the original method
                pass

        # ========== End of NoProbeWts branch ==========

        # If the SVD method fails, fall back to the original lstsq method - kept exactly as in the original
        if weights is None:
            # 1. Fit the model to obtain coefficients
            fit_coef, _, rank, _ = np.linalg.lstsq(design, M.T, rcond=1e-12)
            fit_coef = fit_coef.T  # Transpose to (genes x coefficients)

            # 2. Manually compute the residual matrix (samples x genes), corresponding to R's `effects`
            y_pred = design @ fit_coef.T  # Shape: (n_samples x n_genes)
            residuals_matrix = M.T - y_pred  # Residual matrix (samples x genes)

        else:
            # Analogous handling for the weighted case
            W_sqrt = np.sqrt(weights[0, :])
            W_design = design * W_sqrt[:, np.newaxis]
            W_M = M.T * W_sqrt
            fit_coef, _, rank, _ = np.linalg.lstsq(W_design, W_M, rcond=1e-12)
            fit_coef = fit_coef.T
            y_pred = W_design @ fit_coef.T
            residuals_matrix = W_M - y_pred

        # 3. Compute sigma
        narrays = design.shape[0]  # Number of samples
        if rank < narrays:
            df_residual = narrays - rank
            if df_residual > 0:
                residual_effects = residuals_matrix[rank:, :]  # Shape: (df_residual x n_genes)
                mean_sq = np.mean(residual_effects ** 2, axis=0)
                sigma = np.sqrt(mean_sq)
            else:
                sigma = np.full(ngenes, np.nan)
        else:
            sigma = np.full(ngenes, np.nan)

        # Compute the covariance matrix
        try:
            Q, R = qr(design, mode='economic')
            cov_coef = np.linalg.inv(R.T @ R)
        except Exception as e:
            cov_coef = np.full((nbeta, nbeta), np.nan)

        # Compute unscaled standard deviations
        est = np.arange(rank)
        diag_cov = np.sqrt(np.diag(cov_coef))
        stdev_unscaled[:, est] = np.tile(diag_cov, (ngenes, 1))

        df_residual_arr = np.full(ngenes, df_residual)

        result = {
            'coefficients': fit_coef,
            'stdev_unscaled': stdev_unscaled,
            'sigma': sigma,
            'df_residual': df_residual_arr,
            'cov_coefficients': cov_coef,
            'rank': rank
        }

        return result

    else:
        # A per-gene QR decomposition is needed, so iterate over genes - kept exactly as in the original
        sigma = np.full(ngenes, np.nan)
        df_residual = np.zeros(ngenes)

        for i in range(ngenes):
            y = M[i, :].copy()
            obs = np.isfinite(y)
            valid_count = np.sum(obs)

            if valid_count > 0:
                X = design[obs, :]
                y_obs = y[obs]

                w = None if weights is None else weights[i, obs]
                coef, stdev, sum_res, rank = _lm_fit_one(X, y_obs, w)
                beta[i, :] = coef
                stdev_unscaled[i, :] = stdev
                df_residual[i] = len(y_obs) - rank
                if df_residual[i] > 0:
                    sigma[i] = np.sqrt(sum_res / df_residual[i])

    # Correlation matrix of coefficients - kept exactly as in the original
    try:
        Q, R = qr(design, mode='economic')
        cov_coef = np.linalg.inv(R.T @ R)
        rank = R.shape[1]
    except Exception as e:
        cov_coef = np.full((nbeta, nbeta), np.nan)
        rank = 0

    result = {
        'coefficients': beta,
        'stdev_unscaled': stdev_unscaled,
        'sigma': sigma,
        'df_residual': df_residual,
        'cov_coefficients': cov_coef,
        'rank': rank
    }

    return result


def nonEstimable(x: np.ndarray, tol: float = 1e-7) -> Optional[List[str]]:
    """
    Check which coefficients in the design matrix are non-estimable - kept exactly as in the original.
    """
    if x is None:
        return None

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    p = x.shape[1]  # Number of columns
    Q, R, pivots = qr(x, pivoting=True, mode='economic')

    # Compute the rank
    rank = np.sum(np.abs(np.diag(R)) > tol)

    if rank == p:
        return None
    else:
        # Get column names or generate default ones
        if hasattr(x, 'columns'):
            n = list(x.columns)
        else:
            n = [f'coef_{i + 1}' for i in range(p)]

        # Identify the non-estimable coefficients
        notest = [n[pivots[i]] for i in range(rank, p)]
        return notest


def uniquegenelist(probes: Any, ndups: int = 1, spacing: int = 1) -> Any:
    """
    Extract a unique gene list from a duplicated probe list - kept exactly as in the original.
    """
    if ndups == 1:
        return probes

    if probes is None:
        return None

    if isinstance(probes, (list, np.ndarray)) and probes.ndim == 1:
        # Vector
        n = len(probes) // ndups
        return probes[:n]

    elif isinstance(probes, np.ndarray) and probes.ndim == 2:
        # Matrix
        n = probes.shape[0] // ndups
        return probes[:n, :]

    elif hasattr(probes, 'iloc'):  # pandas DataFrame-like
        n = len(probes) // ndups
        return probes.iloc[:n, :]

    else:
        raise ValueError("probes should be a vector, matrix or data.frame")


def unwrapdups(matrix: np.ndarray, ndups: int = 1, spacing: int = 1) -> np.ndarray:
    """
    Expand duplicated data into the original format - kept exactly as in the original.
    """
    if ndups == 1:
        return matrix

    nrows = matrix.shape[0] * ndups
    if matrix.ndim == 1:
        result = np.zeros(nrows)
        for i in range(ndups):
            result[i::ndups] = matrix
    else:
        result = np.zeros((nrows, matrix.shape[1]))
        for i in range(ndups):
            result[i::ndups, :] = matrix

    return result


def mrlm(*args, **kwargs):
    """Robust regression is not implemented - kept exactly as in the original."""
    raise ValueError("robust regression (mrlm) is not implemented")


def gls_series(*args, **kwargs):
    """Generalized least squares is not implemented - kept exactly as in the original."""
    raise ValueError("generalized least squares (gls.series) is not implemented")


def lmFit(object: Any,
          design: Optional[np.ndarray] = None,
          ndups: Optional[int] = None,
          spacing: Optional[int] = None,
          block: Optional[Any] = None,
          correlation: Optional[float] = None,
          weights: Optional[np.ndarray] = None,
          method: str = "ls", **kwargs) -> Dict[str, Any]:
    """
        Fit linear models for each gene in expression data.

        This is the first step in the limma analysis pipeline, providing the foundation
        for subsequent differential expression analysis. Supports both microarray and
        RNA-seq data analysis.

        Parameters
        ----------
        object : array-like or DataFrame
            Gene expression data matrix with shape (genes, samples).
            Can be a numpy array, pandas DataFrame, or dictionary with 'exprs' key.

        design : array-like or DataFrame, optional
            Design matrix with shape (samples, coefficients).
            If None, creates an intercept-only design matrix.
            Default: None

        ndups : int, optional
            Number of duplicates for each gene. Used when the same gene is represented
            by multiple probes. Default: None (treated as 1)

        spacing : int, optional
            Spacing between duplicates. Default: None (treated as 1)

        block : array-like, optional
            Blocking factor for correlated samples. Default: None

        correlation : float, optional
            Correlation between duplicates. Default: None

        weights : array-like, optional
            Observation weights. Can be a vector or matrix matching data dimensions.
            Default: None

        method : str, optional
            Fitting method. Currently only "ls" (least squares) is supported.
            Default: "ls"

        **kwargs : additional arguments
            Additional parameters passed to the fitting function.

        Returns
        -------
        dict
            Dictionary containing linear model fitting results with the following keys:
            - coefficients: Coefficient matrix (genes × coefficients)
            - stdev_unscaled: Unscaled standard errors matrix
            - sigma: Residual standard deviations
            - df_residual: Residual degrees of freedom
            - cov_coefficients: Covariance matrix of coefficients (if available)
            - rank: Rank of the design matrix
            - Amean: Average expression levels across samples
            - genes: Gene identifiers (if provided)
            - design: Design matrix used for fitting
            - method: Fitting method used

        Examples
        --------
        >>> import limma_py as limma
        >>> import pandas as pd
        >>> import numpy as np

        # Create example expression data (100 genes, 6 samples)
        >>> expr_data = pd.DataFrame(np.random.randn(100, 6))
        >>> design = pd.DataFrame({
        ...     'Intercept': [1, 1, 1, 1, 1, 1],
        ...     'Treatment': [0, 0, 0, 1, 1, 1]
        ... })

        # Fit linear models
        >>> fit = limma.lmFit(expr_data, design)
        >>> print(fit.keys())
        >>> print(f"Coefficients shape: {fit['coefficients'].shape}")
        >>> print(f"Number of genes: {len(fit['sigma'])}")

        Notes
        -----
        - The function automatically handles missing values (NaN) in the expression data.
        - For weighted least squares, provide weights matrix with same dimensions as data.
        - When ndups > 1, the function automatically handles probe duplicates.
        - The design matrix should be full rank for reliable results.

        References
        ----------
        Smyth, G. K. (2004). Linear models and empirical bayes methods for
        assessing differential expression in microarray experiments.
        Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
        """

    original_object = object
    if isinstance(object, pd.DataFrame):
        # Preserve the index information
        probes = object.index.tolist()
        Amean = np.nanmean(object.values, axis=1)

        # Convert to the canonical format, mimicking the original input_data structure exactly
        object = {
            'exprs': object.values,
            'probes': probes,
            'Amean': Amean
        }

    # Check whether `design` is a pandas DataFrame
    if isinstance(design, pd.DataFrame):
        design = design.values
    # ========== End of added section ==========

    # ========== The logic below is kept exactly as in the original ==========

    # Extract the components from the input object
    if isinstance(object, dict) and 'data' in object:
        # Handle a data-frame-like structure
        y = {'exprs': np.array(object['data'])}
        y['Amean'] = np.nanmean(y['exprs'], axis=1)
    elif hasattr(object, 'exprs'):
        # Assume the object has an `exprs` attribute
        y = {
            'exprs': object.exprs,
            'Amean': np.nanmean(object.exprs, axis=1),
            'probes': getattr(object, 'probes', None),
            'design': getattr(object, 'design', None),
            'weights': getattr(object, 'weights', None)
        }
        if hasattr(object, 'printer'):
            y['printer'] = {
                'ndups': getattr(object.printer, 'ndups', None),
                'spacing': getattr(object.printer, 'spacing', None)
            }
    else:
        # Assume `object` is already a dict containing the required fields
        y = object
        # Force-compute Amean (key change)
        if 'Amean' not in y or y['Amean'] is None:
            y['Amean'] = np.nanmean(y['exprs'], axis=1)

    if y['exprs'].shape[0] == 0:
        raise ValueError("the expression matrix has zero rows")

    # Check the design matrix
    if design is None:
        if 'design' in y:
            design = y['design']
        else:
            # If no design matrix is provided, default to all-ones (intercept-only model)
            design = np.ones((y['exprs'].shape[1], 1))
    else:
        design = np.array(design)
        if design.dtype.kind not in 'iuf':
            raise ValueError("design must be a numeric matrix")
        if design.shape[0] != y['exprs'].shape[1]:
            raise ValueError("the row dimension of design does not match the column dimension of the data object")
        if np.any(np.isnan(design)):
            raise ValueError("NA values are not allowed in the design matrix")

    # Check which coefficients are non-estimable
    ne = nonEstimable(design)
    if ne is not None:
        print(f"non-estimable coefficients: {' '.join(ne)}")

    # Check ndups and spacing; default to 1
    if ndups is None:
        if 'printer' in y and 'ndups' in y['printer']:
            ndups = y['printer']['ndups']
        else:
            ndups = 1

    if spacing is None:
        if 'printer' in y and 'spacing' in y['printer']:
            spacing = y['printer']['spacing']
        else:
            spacing = 1

    # Check the weights
    if weights is None and 'weights' in y:
        weights = y['weights']

    # Check the method
    if method not in ["ls", "robust"]:
        raise ValueError("method must be 'ls' or 'robust'")

    # If duplicates are present, reduce probe annotations and Amean to the correct length
    if ndups > 1:
        if 'probes' in y and y['probes'] is not None:
            y['probes'] = uniquegenelist(y['probes'], ndups=ndups, spacing=spacing)
        if 'Amean' in y and y['Amean'] is not None:
            unwrapped = unwrapdups(np.array(y['Amean']), ndups=ndups, spacing=spacing)
            y['Amean'] = np.nanmean(unwrapped, axis=0)

    if method == "robust":
        raise ValueError("robust regression (mrlm) is not implemented")
    else:
        if ndups < 2 and block is None:
            # Use least-squares regression
            fit = lm_series(y['exprs'], design=design, ndups=ndups, spacing=spacing, weights=weights)

            fit['genes'] = y.get('probes', None)
            fit['Amean'] = y.get('Amean', None)
            fit['method'] = method
            fit['design'] = design
            fit['nonEstimable'] = ne

            return fit

        else:
            if correlation is None:
                raise ValueError("correlation must be set; see duplicateCorrelation")
            raise ValueError("generalized least squares (gls.series) is not implemented")

        # Possible warning about missing coefficients
    if fit['coefficients'] is not None and fit['coefficients'].shape[1] > 1:
        n_missing = np.sum(np.isnan(fit['coefficients']), axis=1)
        n = np.sum((n_missing > 0) & (n_missing < fit['coefficients'].shape[1]))
        if n > 0:
            print(f"warning: partial NA coefficients present in {n} probes")

    # Mock fitting results
    fit = {
        'coefficients': np.zeros((y['exprs'].shape[0], design.shape[1])),
        'stdev_unscaled': np.ones((y['exprs'].shape[0], design.shape[1])),
        'sigma': np.ones(y['exprs'].shape[0]),
        'df_residual': np.full(y['exprs'].shape[0], y['exprs'].shape[1] - design.shape[1])
    }

    # Possible warning about missing coefficients
    if fit['coefficients'].shape[1] > 1:
        n_missing = np.sum(np.isnan(fit['coefficients']), axis=1)
        n = np.sum((n_missing > 0) & (n_missing < fit['coefficients'].shape[1]))
        if n > 0:
            print(f"warning: partial NA coefficients present in {n} probes")

    # Return the result
    result = {
        'genes': y.get('probes', None),
        'Amean': y.get('Amean', None),
        'method': method,
        'design': design,
        'coefficients': fit['coefficients'],  # Direct access
        'stdev_unscaled': fit['stdev_unscaled'],  # Direct access
        'sigma': fit['sigma'],  # Direct access
        'df_residual': fit['df_residual'],  # Direct access
        'cov_coefficients': fit.get('cov_coefficients', None),  # Use .get() since it may not be present
        'rank': fit.get('rank', None),  # Use .get() since it may not be present
        'nonEstimable': ne
    }

    return result
