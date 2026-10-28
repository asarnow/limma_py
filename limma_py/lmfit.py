import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.linalg import qr


def as_matrix_weights(weights, shape):
    """将权重转换为矩阵格式 - 完全保持原样"""
    if weights is None:
        return None
    weights = np.array(weights)
    if weights.ndim == 1:
        if weights.shape[0] == shape[0]:
            # 行权重
            return np.tile(weights[:, np.newaxis], (1, shape[1]))
        elif weights.shape[0] == shape[1]:
            # 列权重
            return np.tile(weights[np.newaxis, :], (shape[0], 1))
        else:
            raise ValueError("权重维度不匹配")
    elif weights.shape == shape:
        return weights
    else:
        raise ValueError("权重维度不匹配")


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
    为每个基因拟合线性模型到一系列数组
    - 完全保持原有逻辑，只添加pandas检测
    """

    # 检查表达矩阵 - 添加pandas支持
    if isinstance(M, pd.DataFrame):
        M = M.values
    M = np.asarray(M, dtype=np.float64)
    narrays = M.shape[1]

    # 检查设计矩阵 - 添加pandas支持
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

    # 检查权重 - 添加pandas支持
    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            weights = weights.values
        weights = as_matrix_weights(weights, M.shape)
        weights[weights <= 0] = np.nan
        M[np.isnan(weights)] = np.nan

    # 将重复的行重新格式化为列 - 完全保持原样
    if ndups > 1:
        M = unwrapdups(M, ndups=ndups, spacing=spacing)
        design = np.kron(design, np.ones(ndups))
        if weights is not None:
            weights = unwrapdups(weights, ndups=ndups, spacing=spacing)
        narrays = M.shape[1]

    # 初始化标准误 - 完全保持原样
    ngenes = M.shape[0]
    stdev_unscaled = np.full((ngenes, nbeta), np.nan)
    beta = np.full((ngenes, nbeta), np.nan)

    # 设置行名和列名 - 完全保持原样
    if hasattr(M, 'index'):
        row_names = list(M.index)
    else:
        row_names = [f"gene_{i}" for i in range(ngenes)]

    # 检查QR分解是否对所有基因都是常数 - 完全保持原样
    NoProbeWts = np.all(np.isfinite(M)) and (weights is None or hasattr(weights, 'arrayweights'))

    if NoProbeWts:
        # ========== 完全保持原有的 NoProbeWts 情况 ==========

        if weights is None:
            # 使用 SVD 提高数值稳定性 - 完全保持原样
            try:
                U, s, Vt = np.linalg.svd(design, full_matrices=False)
                rank = np.sum(s > s[0] * 1e-12)  # 基于相对大小的秩检测

                if rank > 0:
                    # 计算系数
                    fit_coef = (Vt.T @ (np.diag(1 / s) @ (U.T @ M.T))).T

                    # 计算残差
                    y_pred = design @ fit_coef.T
                    residuals_matrix = M.T - y_pred

                    # 计算 sigma
                    df_residual = narrays - rank
                    if df_residual > 0:
                        # 使用更稳定的残差计算
                        sigma = np.sqrt(np.sum(residuals_matrix ** 2, axis=0) / df_residual)
                    else:
                        sigma = np.full(ngenes, np.nan)

                    # 计算协方差矩阵和标准误
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
                    print(f"SVD 分解失败，回退到原方法: {e}")
                # 回退到原方法
                pass

        else:
            # 加权情况 - 使用更稳定的方法 - 完全保持原样
            try:
                # 使用第一行权重（假设所有基因权重相同）
                W_sqrt = np.sqrt(weights[0, :])
                W_design = design * W_sqrt[:, np.newaxis]
                W_M = M.T * W_sqrt

                # SVD 分解
                U, s, Vt = np.linalg.svd(W_design, full_matrices=False)
                rank = np.sum(s > s[0] * 1e-12)

                if rank > 0:
                    fit_coef = (Vt.T @ (np.diag(1 / s) @ (U.T @ W_M))).T

                    # 计算加权残差
                    y_pred = design @ fit_coef.T
                    residuals_matrix = M.T - y_pred
                    weighted_residuals = residuals_matrix * W_sqrt

                    # 计算 sigma
                    df_residual = narrays - rank
                    if df_residual > 0:
                        sigma = np.sqrt(np.sum(weighted_residuals ** 2, axis=0) / df_residual)
                    else:
                        sigma = np.full(ngenes, np.nan)

                    # 计算协方差矩阵和标准误
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
                    print(f"加权 SVD 分解失败，回退到原方法: {e}")
                # 回退到原方法
                pass

        # ========== NoProbeWts 情况结束 ==========

        # 如果 SVD 方法失败，回退到原来的 lstsq 方法 - 完全保持原样
        if weights is None:
            # 1. 拟合模型获取系数
            fit_coef, _, rank, _ = np.linalg.lstsq(design, M.T, rcond=1e-12)
            fit_coef = fit_coef.T  # 转为（基因×系数）

            # 2. 手动计算残差矩阵（样本×基因），对应R的effects
            y_pred = design @ fit_coef.T  # 形状：（样本数 × 基因数）
            residuals_matrix = M.T - y_pred  # 残差矩阵（样本×基因）

        else:
            # 加权情况的类似处理
            W_sqrt = np.sqrt(weights[0, :])
            W_design = design * W_sqrt[:, np.newaxis]
            W_M = M.T * W_sqrt
            fit_coef, _, rank, _ = np.linalg.lstsq(W_design, W_M, rcond=1e-12)
            fit_coef = fit_coef.T
            y_pred = W_design @ fit_coef.T
            residuals_matrix = W_M - y_pred

        # 3. 计算sigma
        narrays = design.shape[0]  # 样本数
        if rank < narrays:
            df_residual = narrays - rank
            if df_residual > 0:
                residual_effects = residuals_matrix[rank:, :]  # 形状：（df_residual × 基因数）
                mean_sq = np.mean(residual_effects ** 2, axis=0)
                sigma = np.sqrt(mean_sq)
            else:
                sigma = np.full(ngenes, np.nan)
        else:
            sigma = np.full(ngenes, np.nan)

        # 计算协方差矩阵
        try:
            Q, R = qr(design, mode='economic')
            cov_coef = np.linalg.inv(R.T @ R)
        except Exception as e:
            cov_coef = np.full((nbeta, nbeta), np.nan)

        # 计算未缩放的标准偏差
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
        # 需要逐基因QR分解，因此遍历基因 - 完全保持原样不变
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

    # 系数的相关矩阵 - 完全保持原样
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
    检查设计矩阵中哪些系数是不可估计的 - 完全保持原样
    """
    if x is None:
        return None

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    p = x.shape[1]  # 列数
    Q, R, pivots = qr(x, pivoting=True, mode='economic')

    # 计算秩
    rank = np.sum(np.abs(np.diag(R)) > tol)

    if rank == p:
        return None
    else:
        # 获取列名或生成默认列名
        if hasattr(x, 'columns'):
            n = list(x.columns)
        else:
            n = [f'coef_{i + 1}' for i in range(p)]

        # 找出不可估计的系数
        notest = [n[pivots[i]] for i in range(rank, p)]
        return notest


def uniquegenelist(probes: Any, ndups: int = 1, spacing: int = 1) -> Any:
    """
    从重复的探针列表中提取唯一的基因列表 - 完全保持原样
    """
    if ndups == 1:
        return probes

    if probes is None:
        return None

    if isinstance(probes, (list, np.ndarray)) and probes.ndim == 1:
        # 向量
        n = len(probes) // ndups
        return probes[:n]

    elif isinstance(probes, np.ndarray) and probes.ndim == 2:
        # 矩阵
        n = probes.shape[0] // ndups
        return probes[:n, :]

    elif hasattr(probes, 'iloc'):  # 类似pandas DataFrame
        n = len(probes) // ndups
        return probes.iloc[:n, :]

    else:
        raise ValueError("probes should be a vector, matrix or data.frame")


def unwrapdups(matrix: np.ndarray, ndups: int = 1, spacing: int = 1) -> np.ndarray:
    """
    将重复的数据展开为原始格式 - 完全保持原样
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
    """稳健回归功能未实现 - 完全保持原样"""
    raise ValueError("稳健回归(mrlm)功能未实现")


def gls_series(*args, **kwargs):
    """广义最小二乘功能未实现 - 完全保持原样"""
    raise ValueError("广义最小二乘(gls.series)功能未实现")


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
        # 保存索引信息
        probes = object.index.tolist()
        Amean = np.nanmean(object.values, axis=1)

        # 转换为标准格式，完全模拟原来的input_data结构
        object = {
            'exprs': object.values,
            'probes': probes,
            'Amean': Amean
        }

    # 检查 design 是否是 pandas DataFrame
    if isinstance(design, pd.DataFrame):
        design = design.values
    # ========== 新增结束 ==========

    # ========== 以下是完全保持原样的逻辑 ==========

    # 从输入对象中提取组件
    if isinstance(object, dict) and 'data' in object:
        # 处理数据框结构
        y = {'exprs': np.array(object['data'])}
        y['Amean'] = np.nanmean(y['exprs'], axis=1)
    elif hasattr(object, 'exprs'):
        # 假设对象有exprs属性
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
        # 假设object已经是包含必要字段的字典
        y = object
        # 强制计算Amean（核心修改）
        if 'Amean' not in y or y['Amean'] is None:
            y['Amean'] = np.nanmean(y['exprs'], axis=1)

    if y['exprs'].shape[0] == 0:
        raise ValueError("表达矩阵有零行")

    # 检查设计矩阵
    if design is None:
        if 'design' in y:
            design = y['design']
        else:
            # 如果没有提供设计矩阵，默认为全1矩阵（仅截距模型）
            design = np.ones((y['exprs'].shape[1], 1))
    else:
        design = np.array(design)
        if design.dtype.kind not in 'iuf':
            raise ValueError("design必须是数值矩阵")
        if design.shape[0] != y['exprs'].shape[1]:
            raise ValueError("design的行维度与数据对象的列维度不匹配")
        if np.any(np.isnan(design)):
            raise ValueError("设计矩阵中不允许有NA值")

    # 检查哪些系数不可估计
    ne = nonEstimable(design)
    if ne is not None:
        print(f"不可估计的系数: {' '.join(ne)}")

    # 检查ndups和spacing，默认为1
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

    # 检查权重
    if weights is None and 'weights' in y:
        weights = y['weights']

    # 检查方法
    if method not in ["ls", "robust"]:
        raise ValueError("method必须是'ls'或'robust'")

    # 如果存在重复，将探针注释和Amean减少到正确的长度
    if ndups > 1:
        if 'probes' in y and y['probes'] is not None:
            y['probes'] = uniquegenelist(y['probes'], ndups=ndups, spacing=spacing)
        if 'Amean' in y and y['Amean'] is not None:
            unwrapped = unwrapdups(np.array(y['Amean']), ndups=ndups, spacing=spacing)
            y['Amean'] = np.nanmean(unwrapped, axis=0)

    if method == "robust":
        raise ValueError("稳健回归(mrlm)功能未实现")
    else:
        if ndups < 2 and block is None:
            # 使用最小二乘回归
            fit = lm_series(y['exprs'], design=design, ndups=ndups, spacing=spacing, weights=weights)

            fit['genes'] = y.get('probes', None)
            fit['Amean'] = y.get('Amean', None)
            fit['method'] = method
            fit['design'] = design
            fit['nonEstimable'] = ne

            return fit

        else:
            if correlation is None:
                raise ValueError("必须设置相关性，请参见duplicateCorrelation")
            raise ValueError("广义最小二乘(gls.series)功能未实现")

        # 关于缺失系数的可能警告
    if fit['coefficients'] is not None and fit['coefficients'].shape[1] > 1:
        n_missing = np.sum(np.isnan(fit['coefficients']), axis=1)
        n = np.sum((n_missing > 0) & (n_missing < fit['coefficients'].shape[1]))
        if n > 0:
            print(f"警告: 部分NA系数存在于{n}个探针中")

    # 模拟拟合结果
    fit = {
        'coefficients': np.zeros((y['exprs'].shape[0], design.shape[1])),
        'stdev_unscaled': np.ones((y['exprs'].shape[0], design.shape[1])),
        'sigma': np.ones(y['exprs'].shape[0]),
        'df_residual': np.full(y['exprs'].shape[0], y['exprs'].shape[1] - design.shape[1])
    }

    # 关于缺失系数的可能警告
    if fit['coefficients'].shape[1] > 1:
        n_missing = np.sum(np.isnan(fit['coefficients']), axis=1)
        n = np.sum((n_missing > 0) & (n_missing < fit['coefficients'].shape[1]))
        if n > 0:
            print(f"警告: 部分NA系数存在于{n}个探针中")

    # 输出结果
    result = {
        'genes': y.get('probes', None),
        'Amean': y.get('Amean', None),
        'method': method,
        'design': design,
        'coefficients': fit['coefficients'],  # 直接访问
        'stdev_unscaled': fit['stdev_unscaled'],  # 直接访问
        'sigma': fit['sigma'],  # 直接访问
        'df_residual': fit['df_residual'],  # 直接访问
        'cov_coefficients': fit.get('cov_coefficients', None),  # 这个可以用get，因为可能不存在
        'rank': fit.get('rank', None),  # 这个可以用get，因为可能不存在
        'nonEstimable': ne
    }

    return result
