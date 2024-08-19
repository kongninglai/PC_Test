
import numpy as np


from scipy import special
from pgmpy.myscipy._special import xlogy, prod
from pgmpy.myscipy._utils import namedtuple



_power_div_lambda_names = {
    "pearson": 1,
    "log-likelihood": 0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1,
    "neyman": -2,
    "cressie-read": 2/3,
}



Power_divergenceResult = namedtuple('Power_divergenceResult',
                                    ('statistic', 'pvalue'))

def _broadcast_shapes(shapes, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes
    AxisError: type[Exception]
    # input validation
    if axis is not None:
        axis = np.atleast_1d(axis)
        axis_int = axis.astype(int)
        if not np.array_equal(axis_int, axis):
            raise AxisError('`axis` must be an integer, a '
                            'tuple of integers, or `None`.')
        axis = axis_int

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row)-len(shape):] = shape  # can't use negative indices (-0:)

    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = (f"`axis` is out of bounds "
                       f"for array of dimension {n_dims}")
            raise AxisError(message)

        if len(np.unique(axis)) != len(axis):
            raise AxisError("`axis` must contain only distinct elements")

        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)

    # If arrays are broadcastable, shape elements that are 1 may be replaced
    # with a corresponding non-1 shape element. Assuming arrays are
    # broadcastable, that final shape element can be found with:
    new_shape = np.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= new_shapes.all(axis=0)

    # Among all arrays, there can only be one unique non-1 shape element.
    # Therefore, if any non-1 shape element does not match what we found
    # above, the arrays must not be broadcastable after all.
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        new_axis = axis - np.arange(len(axis))
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape))
                      for removed_shape in removed_shapes]
        return new_shapes
    else:
        return tuple(new_shape)

def _m_broadcast_to(a, shape):
    if np.ma.isMaskedArray(a):
        return np.ma.masked_array(np.broadcast_to(a, shape),
                                  mask=np.broadcast_to(a.mask, shape))
    return np.broadcast_to(a, shape)

def _m_sum(a, *, axis, preserve_mask):
    if np.ma.isMaskedArray(a):
        sum = a.sum(axis)
        return sum if preserve_mask else np.asarray(sum)
    return np.sum(a, axis=axis)

def _m_mean(a, *, axis, keepdims):
    if np.ma.isMaskedArray(a):
        return np.asarray(a.mean(axis=axis, keepdims=keepdims))
    return np.mean(a, axis=axis, keepdims=keepdims)

def xp_size(x):
    """
    Return the total number of elements of x.

    This is equivalent to `x.size` according to the `standard
    <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html>`__.
    This helper is included because PyTorch defines `size` in an
    :external+torch:meth:`incompatible way <torch.Tensor.size>`.

    """
    if None in x.shape:
        return None
    return prod(x.shape)

def _m_count(a, *, axis):
    """Count the number of non-masked elements of an array.

    This function behaves like `np.ma.count`, but is much faster
    for ndarrays.
    """
    if hasattr(a, 'count'):
        num = a.count(axis=axis)
        if isinstance(num, np.ndarray) and num.ndim == 0:
            # In some cases, the `count` method returns a scalar array (e.g.
            # np.array(3)), but we want a plain integer.
            num = int(num)
    else:
        if axis is None:
            num = xp_size(a)
        else:
            num = a.shape[axis]
    return num

class _SimpleChi2:
    # A very simple, array-API compatible chi-squared distribution for use in
    # hypothesis tests. May be replaced by new infrastructure chi-squared
    # distribution in due time.
    def __init__(self, df):
        self.df = df

    def sf(self, x):
        return special.chdtrc(self.df, x)
    
def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    """Cressie-Read power divergence statistic and goodness of fit test.

    This function tests the null hypothesis that the categorical data
    has the given frequencies, using the Cressie-Read power divergence
    statistic.

    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.

        .. deprecated:: 1.14.0
            Support for masked array input was deprecated in
            SciPy 1.14.0 and will be removed in version 1.16.0.

    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.

        .. deprecated:: 1.14.0
            Support for masked array input was deprecated in
            SciPy 1.14.0 and will be removed in version 1.16.0.

    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.
    lambda_ : float or str, optional
        The power in the Cressie-Read power divergence statistic.  The default
        is 1.  For convenience, `lambda_` may be assigned one of the following
        strings, in which case the corresponding numerical value is used:

        * ``"pearson"`` (value 1)
            Pearson's chi-squared statistic. In this case, the function is
            equivalent to `chisquare`.
        * ``"log-likelihood"`` (value 0)
            Log-likelihood ratio. Also known as the G-test [3]_.
        * ``"freeman-tukey"`` (value -1/2)
            Freeman-Tukey statistic.
        * ``"mod-log-likelihood"`` (value -1)
            Modified log-likelihood ratio.
        * ``"neyman"`` (value -2)
            Neyman's statistic.
        * ``"cressie-read"`` (value 2/3)
            The power recommended in [5]_.

    Returns
    -------
    res: Power_divergenceResult
        An object containing attributes:

        statistic : float or ndarray
            The Cressie-Read power divergence test statistic.  The value is
            a float if `axis` is None or if` `f_obs` and `f_exp` are 1-D.
        pvalue : float or ndarray
            The p-value of the test.  The value is a float if `ddof` and the
            return value `stat` are scalars.

    See Also
    --------
    chisquare

    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5.

    Also, the sum of the observed and expected frequencies must be the same
    for the test to be valid; `power_divergence` raises an error if the sums
    do not agree within a relative tolerance of ``eps**0.5``, where ``eps``
    is the precision of the input dtype.

    When `lambda_` is less than zero, the formula for the statistic involves
    dividing by `f_obs`, so a warning or error may be generated if any value
    in `f_obs` is 0.

    Similarly, a warning or error may be generated if any value in `f_exp` is
    zero when `lambda_` >= 0.

    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not a chisquare, in which case this
    test is not appropriate.

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171015035606/http://faculty.vassar.edu/lowry/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] "G-test", https://en.wikipedia.org/wiki/G-test
    .. [4] Sokal, R. R. and Rohlf, F. J. "Biometry: the principles and
           practice of statistics in biological research", New York: Freeman
           (1981)
    .. [5] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.

    Examples
    --------
    (See `chisquare` for more examples.)

    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.  Here we
    perform a G-test (i.e. use the log-likelihood ratio statistic):

    >>> import numpy as np
    >>> from scipy.stats import power_divergence
    >>> power_divergence([16, 18, 16, 14, 12, 12], lambda_='log-likelihood')
    (2.006573162632538, 0.84823476779463769)

    The expected frequencies can be given with the `f_exp` argument:

    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[16, 16, 16, 16, 16, 8],
    ...                  lambda_='log-likelihood')
    (3.3281031458963746, 0.6495419288047497)

    When `f_obs` is 2-D, by default the test is applied to each column.

    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> power_divergence(obs, lambda_="log-likelihood")
    (array([ 2.00657316,  6.77634498]), array([ 0.84823477,  0.23781225]))

    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.

    >>> power_divergence(obs, axis=None)
    (23.31034482758621, 0.015975692534127565)
    >>> power_divergence(obs.ravel())
    (23.31034482758621, 0.015975692534127565)

    `ddof` is the change to make to the default degrees of freedom.

    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=1)
    (2.0, 0.73575888234288467)

    The calculation of the p-values is done by broadcasting the
    test statistic with `ddof`.

    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=[0,1,2])
    (2.0, array([ 0.84914504,  0.73575888,  0.5724067 ]))

    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we must use ``axis=1``:

    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[[16, 16, 16, 16, 16, 8],
    ...                         [8, 20, 20, 16, 12, 12]],
    ...                  axis=1)
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))

    """

    default_float = np.asarray(1.).dtype

    # Convert the input argument `lambda_` to a numerical value.
    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError(f"invalid string for lambda_: {lambda_!r}. "
                             f"Valid strings are {names}")
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1

    def warn_masked(arg):
        pass
        

    warn_masked(f_obs)
    f_obs = f_obs if np.ma.isMaskedArray(f_obs) else np.asarray(f_obs)
    # dtype = default_float if np.isdtype(f_obs.dtype, 'integral') else f_obs.dtype
    dtype = default_float
    f_obs = (f_obs.astype(dtype) if np.ma.isMaskedArray(f_obs)
             else np.asarray(f_obs, dtype=dtype))
    f_obs_float = (f_obs.astype(np.float64) if hasattr(f_obs, 'mask')
                   else np.asarray(f_obs, dtype=np.float64))

    if f_exp is not None:
        warn_masked(f_exp)
        f_exp = f_exp if np.ma.isMaskedArray(f_obs) else np.asarray(f_exp)
        # dtype = default_float if np.isdtype(f_exp.dtype, 'integral') else f_exp.dtype
        dtype = default_float
        f_exp = (f_exp.astype(dtype) if np.ma.isMaskedArray(f_exp)
                 else np.asarray(f_exp, dtype=dtype))

        bshape = _broadcast_shapes((f_obs_float.shape, f_exp.shape))
        f_obs_float = _m_broadcast_to(f_obs_float, bshape)
        f_exp = _m_broadcast_to(f_exp, bshape)
        dtype_res = np.result_type(f_obs.dtype, f_exp.dtype)
        rtol = np.finfo(dtype_res).eps**0.5  # to pass existing tests
        with np.errstate(invalid='ignore'):
            f_obs_sum = _m_sum(f_obs_float, axis=axis, preserve_mask=False)
            f_exp_sum = _m_sum(f_exp, axis=axis, preserve_mask=False)
            relative_diff = (np.abs(f_obs_sum - f_exp_sum) /
                             np.minimum(f_obs_sum, f_exp_sum))
            diff_gt_tol = np.any(relative_diff > rtol, axis=None)
        if diff_gt_tol:
            msg = (f"For each axis slice, the sum of the observed "
                   f"frequencies must agree with the sum of the "
                   f"expected frequencies to a relative tolerance "
                   f"of {rtol}, but the percent differences are:\n"
                   f"{relative_diff}")
            raise ValueError(msg)

    else:
        # Ignore 'invalid' errors so the edge case of a data set with length 0
        # is handled without spurious warnings.
        with np.errstate(invalid='ignore'):
            f_exp = _m_mean(f_obs, axis=axis, keepdims=True)

    # `terms` is the array of terms that are summed along `axis` to create
    # the test statistic.  We use some specialized code for a few special
    # cases of lambda_.
    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs - f_exp)**2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)
        terms = 2.0 * xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        terms = 2.0 * xlogy(f_exp, f_exp / f_obs)
    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp)**lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    stat = _m_sum(terms, axis=axis, preserve_mask=True)

    num_obs = _m_count(terms, axis=axis)
    ddof = np.asarray(ddof)

    df = np.asarray(num_obs - 1 - ddof)
    chi2 = _SimpleChi2(df)
    # pvalue = _get_pvalue(stat, chi2 , alternative='greater', symmetric=False, xp=xp)
    pvalue = chi2.sf(stat)
    stat = stat[()] if stat.ndim == 0 else stat
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue

    return Power_divergenceResult(stat, pvalue)
