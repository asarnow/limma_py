import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import statsmodels.stats.multitest as smm
from statsmodels.stats.multitest import multipletests


def toptable(fit,
             coef=[0],
             number=10,
             genelist=None,
             adjust_method="BH",
             sort_by="B",
             resort_by=None,
             p_value=1.0,
             fc=None,
             lfc=None,
             confint=False):
    """
        Extract a table of the top-ranked genes from linear model fit.

        Generates a ranked list of genes based on differential expression statistics,
        with options for multiple testing correction and various filtering criteria.

        Parameters
        ----------
        fit : dict
            Dictionary containing linear model fitting results, typically from eBayes.
            Required keys: coefficients, t, p_value, lods

        coef : int or list, optional
            Coefficient(s) to display. Can be column index or list of indices.
            Default: 1 (second coefficient)

        number : int, optional
            Number of top genes to display. Default: 10

        genelist : array-like, optional
            Gene identifiers to include in the output. Default: None

        adjust_method : str, optional
            Method for multiple testing correction. Options:
            "BH" (Benjamini-Hochberg), "BY", "bonferroni", "holm", "none"
            Default: "BH"

        sort_by : str, optional
            Column to sort results by. Options:
            "B" (log-odds), "logFC", "AveExpr", "P" (p-value), "t", "none"
            Default: "B"

        resort_by : str, optional
            Secondary sort column. Default: None

        p_value : float, optional
            P-value cutoff for filtering. Default: 1.0 (no filtering)

        fc : float, optional
            Fold-change cutoff (linear scale). Default: None

        lfc : float, optional
            Log-fold-change cutoff (log2 scale). Default: None

        confint : bool or float, optional
            Whether to compute confidence intervals. If float, specifies confidence level.
            Default: False

        Returns
        -------
        DataFrame
            Pandas DataFrame containing the top-ranked genes with columns:
            - logFC: Log2 fold change
            - AveExpr: Average expression level
            - t: Moderated t-statistic
            - P.Value: P-value
            - adj.P.Val: Adjusted p-value (FDR)
            - B: Log-odds of differential expression
            - CI.L: Lower confidence interval (if confint=True)
            - CI.R: Upper confidence interval (if confint=True)

        Examples
        --------
        >>> import limma_py as limma
        >>> import numpy as np
        >>> import pandas as pd

        # Create example data and perform analysis
        >>> expr_data = pd.DataFrame(np.random.randn(100, 6))
        >>> design = pd.DataFrame({'Group1': [1,1,1,0,0,0], 'Group2': [0,0,0,1,1,1]})
        >>> fit = limma.lmFit(expr_data, design)
        >>> eb_fit = limma.eBayes(fit)

        # Extract top 20 differentially expressed genes
        >>> top_genes = limma.toptable(eb_fit, number=20, sort_by="P")
        >>> print(top_genes.head())

        # Extract results with fold-change and p-value filtering
        >>> filtered_results = limma.toptable(eb_fit, number=50, p_value=0.05, lfc=1.0)
        >>> print(f"Number of significant genes: {len(filtered_results)}")

        Notes
        -----
        - When coef is a list with multiple coefficients, F-statistics are computed.
        - The "B" statistic (log-odds) provides Bayesian posterior probabilities.
        - Multiple testing correction is recommended for large-scale genomic studies.
        - Confidence intervals are based on the moderated t-distribution.

        See Also
        --------
        eBayes : Empirical Bayes moderation of standard errors
        lmFit : Linear model fitting for gene expression data
        """

    if genelist is None:
        if 'probes' in fit and fit['probes'] is not None:
            genelist = fit['probes']
        elif 'genes' in fit and fit['genes'] is not None:
            genelist = fit['genes']
    # Automatically convert numpy arrays in `fit` to DataFrames
    if isinstance(fit, dict):
        # Ensure `coefficients` is a DataFrame
        if "coefficients" in fit and isinstance(fit["coefficients"], np.ndarray):
            # Use default column names if none are provided
            n_cols = fit["coefficients"].shape[1]
            col_names = [f'Coef{i}' for i in range(n_cols)]
            index_names = range(fit["coefficients"].shape[0])
            fit["coefficients"] = pd.DataFrame(fit["coefficients"], columns=col_names, index=index_names)

        # Convert the t-statistic
        if "t" in fit and isinstance(fit["t"], np.ndarray):
            # Reuse the column names and index from `coefficients`
            col_names = fit["coefficients"].columns
            index_names = fit["coefficients"].index
            fit["t"] = pd.DataFrame(fit["t"], columns=col_names, index=index_names)

        # Convert p_value
        if "p_value" in fit and isinstance(fit["p_value"], np.ndarray):
            col_names = fit["coefficients"].columns
            index_names = fit["coefficients"].index
            fit["p_value"] = pd.DataFrame(fit["p_value"], columns=col_names, index=index_names)

        # Convert lods
        if "lods" in fit and isinstance(fit["lods"], np.ndarray):
            col_names = fit["coefficients"].columns
            index_names = fit["coefficients"].index
            fit["lods"] = pd.DataFrame(fit["lods"], columns=col_names, index=index_names)

        # Convert other possibly-array components such as F and F_p_value
        if "F" in fit and isinstance(fit["F"], np.ndarray):
            index_names = fit["coefficients"].index
            fit["F"] = pd.Series(fit["F"], index=index_names)

        if "F_p_value" in fit and isinstance(fit["F_p_value"], np.ndarray):
            index_names = fit["coefficients"].index
            fit["F_p_value"] = pd.Series(fit["F_p_value"], index=index_names)

        # Convert stdev_unscaled
        if "stdev_unscaled" in fit and isinstance(fit["stdev_unscaled"], np.ndarray):
            col_names = fit["coefficients"].columns
            index_names = fit["coefficients"].index
            fit["stdev_unscaled"] = pd.DataFrame(fit["stdev_unscaled"], columns=col_names, index=index_names)

        # Convert Amean
        if "Amean" in fit and isinstance(fit["Amean"], np.ndarray):
            index_names = fit["coefficients"].index
            fit["Amean"] = pd.Series(fit["Amean"], index=index_names)

    if genelist is None:
        genelist = pd.DataFrame(index=fit["coefficients"].index)

    if not isinstance(fit, dict):
        raise TypeError("fit must be a dict-like object")

    if "coefficients" not in fit or fit["coefficients"] is None:
        raise ValueError("coefficients not found in fit object")

    t_exists = "t" in fit and fit["t"] is not None
    F_exists = "F" in fit and fit["F"] is not None
    if not (t_exists or F_exists):
        raise ValueError("Need to run eBayes or treat first")

    if confint:
        if "stdev.unscaled" not in fit or fit["stdev.unscaled"] is None:
            raise ValueError("stdev.unscaled not found in fit object")

    if coef is None:
        if fit.get("treat_lfc") is None:
            coef = list(range(fit["coefficients"].shape[1]))
            cn = fit["design"].columns.tolist()
            if cn is not None:
                try:
                    i = cn.index("(Intercept)")
                    coef.pop(i)
                    print("Removing intercept from test coefficients")
                except ValueError:
                    pass
        else:
            coef = fit["coefficients"].shape[1] - 1

    if adjust_method is None:
        adjust_method = "BH"

    if fc is None:
        if lfc is None:
            lfc = 0
    else:
        if fc < 1:
            raise ValueError("fc must be greater than or equal to 1")
        lfc = np.log2(fc)

    if len(coef) > 1:
        if fit.get("treat.lfc") is not None:
            raise ValueError("Treat p-values can only be displayed for single coefficients")

        coef = list(dict.fromkeys(coef))  # Unique values, preserving order

        # If `fit` has more coefficient columns than we want to keep, slice by `coef`
        if fit["coefficients"].shape[1] > len(coef):
            fit = {k: v.iloc[:, coef] if k in ("coefficients", "t", "p_value", "lods", "stdev_unscaled")
            else v for k, v in fit.items()}

        if sort_by == "B":
            sort_by = "F"

        # Call the Python version of .topTableF
        return _topTableF(fit,
                          number=number,
                          genelist=genelist,
                          adjust_method=adjust_method,
                          sort_by=sort_by,
                          p_value=p_value,
                          lfc=lfc)
    # 1. Cast `fit` to a plain dict/DataFrame (equivalent to R's unclass)
    fit = dict(fit)  # If `fit` was a custom object, convert it to a plain dict

    # 2. Build ebcols
    ebcols = ["t", "p_value", "lods"]
    if confint:
        ebcols = ["s2_post", "df_total"] + ebcols

    if isinstance(coef, list) and len(coef) == 1:
        coef_int = coef[0]
    else:
        coef_int = coef

    eb_dict = {}
    for k in ebcols:
        if k in fit:
            # Slice columns of t, p_value, and lods; leave the other statistics untouched
            if k in ("t", "p_value", "lods") and hasattr(fit[k], 'iloc'):
                eb_dict[k] = fit[k].iloc[:, coef]
            else:
                eb_dict[k] = fit[k]

    # 3. Assemble the arguments passed to _top_table_t
    return _topTableT(
        fit={
            "coefficients": fit["coefficients"].iloc[:, coef],  # Keep only the requested coefficient columns
            "stdev_unscaled": fit["stdev_unscaled"].iloc[:, coef]
        },
        coef=coef_int,
        number=number,
        genelist=genelist,
        A=fit.get("Amean"),  # Amean if it is stored in `fit`
        eb=eb_dict,
        adjust_method=adjust_method,
        sort_by=sort_by,
        resort_by=resort_by,
        p_value=p_value,
        lfc=lfc,
        confint=confint
    )


def _topTableF(fit,
               number=10,
               genelist=None,
               adjust_method="BH",
               sort_by="F",
               p_value=1,
               lfc=0):
    if genelist is None:
        if 'probes' in fit and fit['probes'] is not None:
            genelist = fit['probes']

    if fit.get("coefficients") is None:
        raise ValueError("Coefficients not found in fit")

    M = np.asarray(fit["coefficients"])
    rn = fit["coefficients"].index.tolist()

    if rn is None or len(rn) == 0:
        rn = [f"Gene{i}" for i in range(M.shape[0])]

    if fit["coefficients"].columns.name is None or len(fit["coefficients"].columns) == 0:
        col_names = [f"Coef{i + 1}" for i in range(M.shape[1])]
    else:
        col_names = fit["coefficients"].columns.tolist()

    Amean = fit.get("Amean")
    Fstat = fit.get("F")
    Fp = fit.get("F_p_value")

    if Fstat is None:
        raise ValueError("F-statistics not found in fit")

    if genelist is not None and (not isinstance(genelist, pd.DataFrame) or genelist.ndim == 1):
        genelist = pd.DataFrame({"ProbeID": genelist})

    # Handle row names
    if rn is None or len(rn) == 0:
        rn = list(range(M.shape[0]))
    else:
        # Check for duplicate row names
        if len(rn) != len(set(rn)):
            if genelist is None:
                genelist = pd.DataFrame({"ID": rn})
            else:
                if "ID" in genelist.columns:
                    genelist = genelist.assign(ID0=rn)
                else:
                    genelist = genelist.assign(ID=rn)
            rn = list(range(M.shape[0]))  # Replace with positional indices

    valid = ("F", "none")
    if sort_by not in valid:
        raise ValueError(f"sort.by must be one of {valid}")

    _method_map = {"BH": "fdr_bh", "BY": "fdr_by",
                   "bonferroni": "bonferroni", "holm": "holm", "none": "none"}
    adjust_method = _method_map.get(adjust_method, adjust_method)
    adj_p_value = smm.multipletests(Fp, method=adjust_method)[1]

    if lfc > 0 or p_value < 1:
        # Fold-change filtering
        if lfc > 0:
            big = np.any(np.abs(M) > lfc, axis=1)  # At least one coefficient per row must pass
        else:
            big = np.ones(M.shape[0], dtype=bool)

        # P-value filtering
        if p_value < 1:
            sig = adj_p_value <= p_value
            sig = np.where(np.isnan(sig), False, sig)
        else:
            sig = np.ones(M.shape[0], dtype=bool)

        keep = big & sig

        # Slice by index uniformly - key fix: use iloc for pandas objects
        if not keep.all():
            idx = np.where(keep)[0]
            M = M[idx, :]
            rn = [rn[i] for i in idx]

            # Use iloc for positional indexing on pandas Series
            if isinstance(Amean, pd.Series):
                Amean = Amean.iloc[idx]
            else:
                Amean = Amean[idx]

            if isinstance(Fstat, pd.Series):
                Fstat = Fstat.iloc[idx]
            else:
                Fstat = Fstat[idx]

            if isinstance(Fp, pd.Series):
                Fp = Fp.iloc[idx]
            else:
                Fp = Fp[idx]

            genelist = genelist.iloc[idx] if genelist is not None else None
            adj_p_value = adj_p_value[idx]

    # Ensure `number` does not exceed the amount of available data
    if M.shape[0] < number:
        number = M.shape[0]
    if number < 1:
        return pd.DataFrame()

    # Sort indices
    if sort_by == "F":
        o = np.argsort(Fp)[:number]  # Ascending (smallest p-values first)
    else:
        o = np.arange(number)

    # Build the result table
    if genelist is None:
        tab = pd.DataFrame(M[o, :], columns=col_names)
    else:
        # Ensure genelist is sliced correctly
        tab = genelist.iloc[o, :].copy()
        m_df = pd.DataFrame(M[o, :], columns=col_names, index=tab.index)  # Keep indices aligned
        tab = pd.concat([tab, m_df], axis=1)

    # Append statistic columns - ensure all arrays have matching lengths
    tab["AveExpr"] = Amean.iloc[o].to_numpy() if isinstance(Amean, pd.Series) else Amean[o]
    tab["F"] = Fstat.iloc[o].to_numpy() if isinstance(Fstat, pd.Series) else Fstat[o]
    tab["P.Value"] = Fp.iloc[o].to_numpy() if isinstance(Fp, pd.Series) else Fp[o]
    tab["adj.P.Val"] = adj_p_value[o]

    # Set the index
    tab.index = [rn[i] for i in o]
    tab = tab.reset_index(drop=True)
    return tab


def _topTableT(fit,
               coef=1,
               number=10,
               genelist=None,
               A=None,
               eb=None,
               adjust_method="BH",
               sort_by="B",
               resort_by=None,
               p_value=1,
               lfc=0,
               confint=False):

    _method_map = {"BH": "fdr_bh", "BY": "fdr_by",
                   "bonferroni": "bonferroni", "holm": "holm", "none": "none"}
    adjust_method = _method_map.get(adjust_method, adjust_method)

    # Key change: pull gene names from the `fit` object
    if genelist is None:
        if 'probes' in fit and fit['probes'] is not None:
            genelist = fit['probes']
        elif hasattr(fit.get('coefficients'), 'index'):
            genelist = fit['coefficients'].index.tolist()

    # Key change: ensure the row names are obtained correctly
    if hasattr(fit["coefficients"], 'index'):
        rn = fit["coefficients"].index.tolist()
    else:
        rn = list(range(fit["coefficients"].shape[0]))

    fit["coefficients"] = np.asarray(fit["coefficients"])

    if isinstance(coef, list):
        col_index = coef[0]
    else:
        col_index = coef

    if col_index >= fit["coefficients"].shape[1]:
        col_index = 0

    if isinstance(coef, (list, tuple)) and len(coef) > 1:
        import warnings
        warnings.warn("Treat is for single coefficients: only first value of coef being used")

    # Key change: handle genelist correctly
    if genelist is not None:
        if isinstance(genelist, list):
            genelist = pd.DataFrame({"ID": genelist})
        elif not isinstance(genelist, pd.DataFrame):
            genelist = pd.DataFrame({"ID": genelist})

    # Check rownames
    if rn is None or len(rn) == 0:
        rn = list(range(fit["coefficients"].shape[0]))
    else:
        # Detect duplicates in Python
        if len(rn) != len(set(rn)):
            if genelist is None:
                genelist = pd.DataFrame({"ID": rn})
            else:
                if "ID" in genelist.columns:
                    genelist["ID0"] = rn
                else:
                    genelist["ID"] = rn
            rn = list(range(fit["coefficients"].shape[0]))

    valid = ["logFC", "M", "A", "Amean", "AveExpr", "P", "p", "T", "t", "B", "none"]
    if sort_by not in valid:
        raise ValueError(f"sort.by must be one of {valid}")

    # Alias normalization
    alias_map = {
        "M": "logFC",
        "A": "AveExpr",
        "Amean": "AveExpr",
        "T": "t",
        "p": "P"
    }
    sort_by = alias_map.get(sort_by, sort_by)

    if resort_by is not None:
        valid_res = ["logFC", "M", "A", "Amean", "AveExpr", "P", "p", "T", "t", "B"]
        if resort_by not in valid_res:
            raise ValueError(f"resort.by must be one of {valid_res}")

        alias_res = {"M": "logFC", "A": "AveExpr", "Amean": "AveExpr",
                     "p": "P", "T": "t"}
        resort_by = alias_res.get(resort_by, resort_by)

    if A is None:
        if sort_by == "AveExpr":
            raise ValueError("Cannot sort by A-values as these have not been given")
    else:
        A = np.asarray(A)
        if A.ndim > 1:
            A = np.nanmean(A, axis=1)

    if eb is None or eb.get("lods") is None:
        if sort_by == "B":
            raise ValueError("Trying to sort.by B, but B-statistic (lods) not found")
        if resort_by is not None and resort_by == "B":
            raise ValueError("Trying to resort.by B, but B-statistic (lods) not found")
        include_B = False
    else:
        include_B = True

    # Extract the requested column
    M = fit["coefficients"][:, col_index]
    tstat = np.asarray(eb["t"])[:, col_index]
    P_Value = np.asarray(eb["p_value"])[:, col_index]
    if include_B:
        B = np.asarray(eb["lods"])[:, col_index]
    else:
        B = None

    adj_P_Value = smm.multipletests(P_Value, method=adjust_method)[1]

    if p_value < 1 or lfc > 0:
        mask = (adj_P_Value <= p_value) & (np.abs(M) >= lfc)
        mask = np.where(np.isnan(mask), False, mask)

        if not mask.any():
            return pd.DataFrame()

        idx = np.where(mask)[0]

        genelist = genelist.iloc[idx] if genelist is not None else None
        M = M[idx]
        A = A[idx] if A is not None else None
        tstat = tstat[idx]
        P_Value = P_Value[idx]
        adj_P_Value = adj_P_Value[idx]
        if include_B:
            B = B[idx]
        rn = [rn[i] for i in idx]
    else:
        idx = np.arange(len(M))
        rn = rn

    if len(M) < number:
        number = len(M)
    if number < 1:
        return pd.DataFrame()

    # Build the sort indices
    if sort_by == "logFC":
        order = np.argsort(-np.abs(M))
    elif sort_by == "AveExpr":
        order = np.argsort(-A)
    elif sort_by == "P":
        order = np.argsort(P_Value)
    elif sort_by == "t":
        order = np.argsort(-np.abs(tstat))
    elif sort_by == "B":
        order = np.argsort(-B)
    else:
        order = np.arange(len(M))

    top = order[:number]

    # Key change: build the result table and set gene names as the index
    if genelist is None:
        tab = pd.DataFrame({"logFC": M[top]})
        # Set the gene-name index directly
        tab.index = [rn[i] for i in top]
    else:
        tab = genelist.iloc[top].copy()
        tab["logFC"] = M[top]
        # Set the gene-name index
        tab.index = [rn[i] for i in top]

    # Confidence intervals and other statistics
    if confint:
        alpha = 0.975 if not isinstance(confint, (int, float)) else (1 + float(confint)) / 2
        df_t = eb["df_total"][top]
        se = np.sqrt(eb["s2_post"][top]) * fit["stdev_unscaled"][top, col_index]
        margin = se * t_dist.ppf(alpha, df_t)
        tab["CI.L"] = M[top] - margin
        tab["CI.R"] = M[top] + margin

    if A is not None:
        tab["AveExpr"] = A[top]

    tab = pd.concat([
        tab,
        pd.DataFrame({
            "t": tstat[top],
            "P.Value": P_Value[top],
            "adj.P.Val": adj_P_Value[top]
        }, index=tab.index)
    ], axis=1)

    if include_B:
        tab["B"] = B[top]
    tab = tab.reset_index(drop=True)
    return tab