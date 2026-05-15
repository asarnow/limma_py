import numpy as np
import pandas as pd
import warnings
from scipy.linalg import qr
from scipy.interpolate import BSpline


def spline_design(knots, x, ord=4, derivs=0):
    """
    Faithful reproduction of R's splineDesign function.
    """
    if not isinstance(derivs, (list, tuple, np.ndarray)):
        derivs = [derivs]

    n = len(knots) - ord
    result = np.zeros((len(x), n))

    for i in range(n):
        # Create the B-spline basis function
        coefs = np.zeros(n)
        coefs[i] = 1.0
        spl = BSpline(knots, coefs, ord - 1, extrapolate=False)

        for j, deriv in enumerate(derivs):
            if deriv == 0:
                result[:, i] = spl(x)
            else:
                # Evaluate the derivative
                spl_deriv = spl.derivative(deriv)
                result[:, i] = spl_deriv(x)

    return result


def ns(x, df=None, knots=None, intercept=False, Boundary_knots=None):
    """
    Faithful reproduction of R's splines::ns function.
    """
    # Preserve the input name
    nx = None
    if hasattr(x, 'name'):
        nx = x.name
    elif hasattr(x, 'index'):
        nx = x.index

    x = np.asarray(x).flatten()
    nax = np.isnan(x)
    nas = nax.any()
    if nas:
        x = x[~nax]

    # Handle the boundary knots - follows R's logic exactly
    if Boundary_knots is None:
        Boundary_knots = np.array([np.min(x), np.max(x)])
    else:
        Boundary_knots = np.sort(Boundary_knots)

    # Check for points outside the boundaries
    outside = (x < Boundary_knots[0]) | (x > Boundary_knots[1])

    # Determine the interior knots - follows R's logic exactly
    mk_knots = (df is not None and knots is None)
    if mk_knots:
        nIknots = df - 1 - intercept
        if nIknots < 0:
            nIknots = 0
            warnings.warn(f"'df' was too small; have used {1 + intercept}")

        if nIknots > 0:
            # Compute quantile positions exactly as R does
            knots_pos = np.linspace(0, 1, nIknots + 2)[1:-1]
            # Use R-style quantile interpolation
            x_filtered = x[~outside]
            knots = np.quantile(x_filtered, knots_pos,
                                method='linear')  # R uses type=7 by default
        else:
            knots = None
    else:
        if knots is not None and not np.all(np.isfinite(knots)):
            raise ValueError("non-finite knots")
        nIknots = len(knots) if knots is not None else 0

    # Adjust interior knots that coincide with boundary knots - follows R's logic exactly
    if mk_knots and knots is not None and len(knots) > 0:
        lr_eq = [np.min(knots) == Boundary_knots[0], np.max(knots) == Boundary_knots[1]]

        if lr_eq[0]:
            piv = Boundary_knots[0]
            i = knots == piv
            if np.all(i):
                raise ValueError("all interior knots match left boundary knot")
            knots[i] = knots[i] + (np.min(knots[knots > piv]) - piv) / 8

        if lr_eq[1]:
            piv = Boundary_knots[1]
            i = knots == piv
            if np.all(i):
                raise ValueError("all interior knots match right boundary knot")
            knots[i] = knots[i] - (piv - np.max(knots[knots < piv])) / 8

        if any(lr_eq):
            warnings.warn("shoving 'interior' knots matching boundary knots to inside")

    # Build the extended knot sequence - follows R's logic exactly
    if knots is None:
        Aknots = np.repeat(Boundary_knots, 4)
    else:
        Aknots = np.sort(np.concatenate([np.repeat(Boundary_knots, 4), knots]))

    # Evaluate the spline basis - follows R's logic
    if np.any(outside):
        basis = np.zeros((len(x), len(Aknots) - 4))

        # Handle points to the left of the boundary
        if np.any(ol := (x < Boundary_knots[0])):
            k_pivot = Boundary_knots[0]
            xl = x[ol]

            # Compute the function value and first derivative at the boundary
            tt0 = spline_design(Aknots, [k_pivot], ord=4, derivs=0)
            tt1 = spline_design(Aknots, [k_pivot], ord=4, derivs=1)
            tt = np.vstack([tt0, tt1])

            # Linear extrapolation - follows R's logic exactly
            xl_mat = np.column_stack([np.ones(len(xl)), xl - k_pivot])
            basis[ol, :] = xl_mat @ tt

        # Handle points to the right of the boundary
        if np.any(or_ := (x > Boundary_knots[1])):
            k_pivot = Boundary_knots[1]
            xr = x[or_]

            # Compute the function value and first derivative at the boundary
            tt0 = spline_design(Aknots, [k_pivot], ord=4, derivs=0)
            tt1 = spline_design(Aknots, [k_pivot], ord=4, derivs=1)
            tt = np.vstack([tt0, tt1])

            # Linear extrapolation - follows R's logic exactly
            xr_mat = np.column_stack([np.ones(len(xr)), xr - k_pivot])
            basis[or_, :] = xr_mat @ tt

        # Handle interior points
        if np.any(inside := ~outside):
            basis[inside, :] = spline_design(Aknots, x[inside], ord=4)
    else:
        basis = spline_design(Aknots, x, ord=4)

    # Build the constraint matrix (second derivatives at the boundary knots) - follows R's logic exactly
    const = spline_design(Aknots, Boundary_knots, ord=4, derivs=2)

    # Drop the intercept column if needed - follows R's logic exactly
    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]

    # QR decomposition and orthogonal projection - follows R's logic exactly
    const_t = const.T
    Q, R = qr(const_t, mode='full')

    # Equivalent to R's qr.qty(qr.const, t(basis))
    basis_t = basis.T
    basis_transformed = Q.T @ basis_t

    # Transpose back and drop the first two columns - follows R's logic exactly
    basis_result = basis_transformed.T
    if basis_result.shape[1] > 2:
        basis_result = basis_result[:, 2:]
    else:
        basis_result = np.zeros((basis_result.shape[0], 0))

    # Restore missing values
    if nas:
        nmat = np.full((len(nax), basis_result.shape[1]), np.nan)
        nmat[~nax, :] = basis_result
        basis_result = nmat

    # Build the result DataFrame
    n_col = basis_result.shape[1]
    columns = [str(i + 1) for i in range(n_col)]

    result = pd.DataFrame(basis_result, columns=columns)

    # Attach metadata (matching R's behavior)
    result._metadata = ['degree', 'knots', 'Boundary_knots', 'intercept']
    result.degree = 3
    result.knots = np.array([]) if knots is None else knots
    result.Boundary_knots = Boundary_knots
    result.intercept = intercept

    return result