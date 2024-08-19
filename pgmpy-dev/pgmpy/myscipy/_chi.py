import numpy as np
from functools import reduce
import sys as _sys

from pgmpy.myscipy._power import power_divergence
from pgmpy.myscipy._utils import _validate_names

# from scipy.stats import power_divergence
def _make_tuple_bunch(typename, field_names, extra_field_names=None,
                      module=None):
    """
    Create a namedtuple-like class with additional attributes.

    This function creates a subclass of tuple that acts like a namedtuple
    and that has additional attributes.

    The additional attributes are listed in `extra_field_names`.  The
    values assigned to these attributes are not part of the tuple.

    The reason this function exists is to allow functions in SciPy
    that currently return a tuple or a namedtuple to returned objects
    that have additional attributes, while maintaining backwards
    compatibility.

    This should only be used to enhance *existing* functions in SciPy.
    New functions are free to create objects as return values without
    having to maintain backwards compatibility with an old tuple or
    namedtuple return value.

    Parameters
    ----------
    typename : str
        The name of the type.
    field_names : list of str
        List of names of the values to be stored in the tuple. These names
        will also be attributes of instances, so the values in the tuple
        can be accessed by indexing or as attributes.  At least one name
        is required.  See the Notes for additional restrictions.
    extra_field_names : list of str, optional
        List of names of values that will be stored as attributes of the
        object.  See the notes for additional restrictions.

    Returns
    -------
    cls : type
        The new class.

    Notes
    -----
    There are restrictions on the names that may be used in `field_names`
    and `extra_field_names`:

    * The names must be unique--no duplicates allowed.
    * The names must be valid Python identifiers, and must not begin with
      an underscore.
    * The names must not be Python keywords (e.g. 'def', 'and', etc., are
      not allowed).

    Examples
    --------
    >>> from scipy._lib._bunch import _make_tuple_bunch

    Create a class that acts like a namedtuple with length 2 (with field
    names `x` and `y`) that will also have the attributes `w` and `beta`:

    >>> Result = _make_tuple_bunch('Result', ['x', 'y'], ['w', 'beta'])

    `Result` is the new class.  We call it with keyword arguments to create
    a new instance with given values.

    >>> result1 = Result(x=1, y=2, w=99, beta=0.5)
    >>> result1
    Result(x=1, y=2, w=99, beta=0.5)

    `result1` acts like a tuple of length 2:

    >>> len(result1)
    2
    >>> result1[:]
    (1, 2)

    The values assigned when the instance was created are available as
    attributes:

    >>> result1.y
    2
    >>> result1.beta
    0.5
    """
    if len(field_names) == 0:
        raise ValueError('field_names must contain at least one name')

    if extra_field_names is None:
        extra_field_names = []
    _validate_names(typename, field_names, extra_field_names)

    typename = _sys.intern(str(typename))
    field_names = tuple(map(_sys.intern, field_names))
    extra_field_names = tuple(map(_sys.intern, extra_field_names))

    all_names = field_names + extra_field_names
    arg_list = ', '.join(field_names)
    full_list = ', '.join(all_names)
    repr_fmt = ''.join(('(',
                        ', '.join(f'{name}=%({name})r' for name in all_names),
                        ')'))
    tuple_new = tuple.__new__
    _dict, _tuple, _zip = dict, tuple, zip

    # Create all the named tuple methods to be added to the class namespace

    s = f"""\
def __new__(_cls, {arg_list}, **extra_fields):
    return _tuple_new(_cls, ({arg_list},))

def __init__(self, {arg_list}, **extra_fields):
    for key in self._extra_fields:
        if key not in extra_fields:
            raise TypeError("missing keyword argument '%s'" % (key,))
    for key, val in extra_fields.items():
        if key not in self._extra_fields:
            raise TypeError("unexpected keyword argument '%s'" % (key,))
        self.__dict__[key] = val

def __setattr__(self, key, val):
    if key in {repr(field_names)}:
        raise AttributeError("can't set attribute %r of class %r"
                             % (key, self.__class__.__name__))
    else:
        self.__dict__[key] = val
"""
    del arg_list
    namespace = {'_tuple_new': tuple_new,
                 '__builtins__': dict(TypeError=TypeError,
                                      AttributeError=AttributeError),
                 '__name__': f'namedtuple_{typename}'}
    exec(s, namespace)
    __new__ = namespace['__new__']
    __new__.__doc__ = f'Create new instance of {typename}({full_list})'
    __init__ = namespace['__init__']
    __init__.__doc__ = f'Instantiate instance of {typename}({full_list})'
    __setattr__ = namespace['__setattr__']

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + repr_fmt % self._asdict()

    def _asdict(self):
        'Return a new dict which maps field names to their values.'
        out = _dict(_zip(self._fields, self))
        out.update(self.__dict__)
        return out

    def __getnewargs_ex__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return _tuple(self), self.__dict__

    # Modify function metadata to help with introspection and debugging
    for method in (__new__, __repr__, _asdict, __getnewargs_ex__):
        method.__qualname__ = f'{typename}.{method.__name__}'

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        '__doc__': f'{typename}({full_list})',
        '_fields': field_names,
        '__new__': __new__,
        '__init__': __init__,
        '__repr__': __repr__,
        '__setattr__': __setattr__,
        '_asdict': _asdict,
        '_extra_fields': extra_field_names,
        '__getnewargs_ex__': __getnewargs_ex__,
    }
    for index, name in enumerate(field_names):

        def _get(self, index=index):
            return self[index]
        class_namespace[name] = property(_get)
    for name in extra_field_names:

        def _get(self, name=name):
            return self.__dict__[name]
        class_namespace[name] = property(_get)

    result = type(typename, (tuple,), class_namespace)

    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the named tuple is created.  Bypass this step in environments
    # where sys._getframe is not defined (Jython for example) or sys._getframe
    # is not defined for arguments greater than 0 (IronPython), or where the
    # user has specified a particular module.
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module
        __new__.__module__ = module

    return result

def margins(a):
    """Return a list of the marginal sums of the array `a`.

    Parameters
    ----------
    a : ndarray
        The array for which to compute the marginal sums.

    Returns
    -------
    margsums : list of ndarrays
        A list of length `a.ndim`.  `margsums[k]` is the result
        of summing `a` over all axes except `k`; it has the same
        number of dimensions as `a`, but the length of each axis
        except axis `k` will be 1.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import margins

    >>> a = np.arange(12).reshape(2, 6)
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])
    >>> m0, m1 = margins(a)
    >>> m0
    array([[15],
           [51]])
    >>> m1
    array([[ 6,  8, 10, 12, 14, 16]])

    >>> b = np.arange(24).reshape(2,3,4)
    >>> m0, m1, m2 = margins(b)
    >>> m0
    array([[[ 66]],
           [[210]]])
    >>> m1
    array([[[ 60],
            [ 92],
            [124]]])
    >>> m2
    array([[[60, 66, 72, 78]]])
    """
    margsums = []
    ranged = list(range(a.ndim))
    for k in ranged:
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums

def expected_freq(observed):
    """
    Compute the expected frequencies from a contingency table.

    Given an n-dimensional contingency table of observed frequencies,
    compute the expected frequencies for the table based on the marginal
    sums under the assumption that the groups associated with each
    dimension are independent.

    Parameters
    ----------
    observed : array_like
        The table of observed frequencies.  (While this function can handle
        a 1-D array, that case is trivial.  Generally `observed` is at
        least 2-D.)

    Returns
    -------
    expected : ndarray of float64
        The expected frequencies, based on the marginal sums of the table.
        Same shape as `observed`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import expected_freq
    >>> observed = np.array([[10, 10, 20],[20, 20, 20]])
    >>> expected_freq(observed)
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    """
    # Typically `observed` is an integer array. If `observed` has a large
    # number of dimensions or holds large values, some of the following
    # computations may overflow, so we first switch to floating point.
    observed = np.asarray(observed, dtype=np.float64)

    # Create a list of the marginal sums.
    margsums = margins(observed)

    # Create the array of expected frequencies.  The shapes of the
    # marginal sums returned by apply_over_axes() are just what we
    # need for broadcasting in the following product.
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected

Chi2ContingencyResult = _make_tuple_bunch(
    'Chi2ContingencyResult',
    ['statistic', 'pvalue', 'dof', 'expected_freq'], []
)

def chi2_contingency(observed, correction=True, lambda_=None):
    """Chi-square test of independence of variables in a contingency table.

    This function computes the chi-square statistic and p-value for the
    hypothesis test of independence of the observed frequencies in the
    contingency table [1]_ `observed`.  The expected frequencies are computed
    based on the marginal sums under the assumption of independence; see
    `scipy.stats.contingency.expected_freq`.  The number of degrees of
    freedom is (expressed using numpy functions and attributes)::

        dof = observed.size - sum(observed.shape) + observed.ndim - 1


    Parameters
    ----------
    observed : array_like
        The contingency table. The table contains the observed frequencies
        (i.e. number of occurrences) in each category.  In the two-dimensional
        case, the table is often described as an "R x C table".
    correction : bool, optional
        If True, *and* the degrees of freedom is 1, apply Yates' correction
        for continuity.  The effect of the correction is to adjust each
        observed value by 0.5 towards the corresponding expected value.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic [2]_.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family [3]_ to be used instead.  See
        `scipy.stats.power_divergence` for details.

    Returns
    -------
    res : Chi2ContingencyResult
        An object containing attributes:

        statistic : float
            The test statistic.
        pvalue : float
            The p-value of the test.
        dof : int
            The degrees of freedom.
        expected_freq : ndarray, same shape as `observed`
            The expected frequencies, based on the marginal sums of the table.

    See Also
    --------
    scipy.stats.contingency.expected_freq
    scipy.stats.fisher_exact
    scipy.stats.chisquare
    scipy.stats.power_divergence
    scipy.stats.barnard_exact
    scipy.stats.boschloo_exact

    Notes
    -----
    An often quoted guideline for the validity of this calculation is that
    the test should be used only if the observed and expected frequencies
    in each cell are at least 5.

    This is a test for the independence of different categories of a
    population. The test is only meaningful when the dimension of
    `observed` is two or more.  Applying the test to a one-dimensional
    table will always result in `expected` equal to `observed` and a
    chi-square statistic equal to 0.

    This function does not handle masked arrays, because the calculation
    does not make sense with missing values.

    Like `scipy.stats.chisquare`, this function computes a chi-square
    statistic; the convenience this function provides is to figure out the
    expected frequencies and degrees of freedom from the given contingency
    table. If these were already known, and if the Yates' correction was not
    required, one could use `scipy.stats.chisquare`.  That is, if one calls::

        res = chi2_contingency(obs, correction=False)

    then the following is true::

        (res.statistic, res.pvalue) == stats.chisquare(obs.ravel(),
                                                       f_exp=ex.ravel(),
                                                       ddof=obs.size - 1 - dof)

    The `lambda_` argument was added in version 0.13.0 of scipy.

    References
    ----------
    .. [1] "Contingency table",
           https://en.wikipedia.org/wiki/Contingency_table
    .. [2] "Pearson's chi-squared test",
           https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In [4]_, the use of aspirin to prevent cardiovascular events in women
    and men was investigated. The study notably concluded:

        ...aspirin therapy reduced the risk of a composite of
        cardiovascular events due to its effect on reducing the risk of
        ischemic stroke in women [...]

    The article lists studies of various cardiovascular events. Let's
    focus on the ischemic stoke in women.

    The following table summarizes the results of the experiment in which
    participants took aspirin or a placebo on a regular basis for several
    years. Cases of ischemic stroke were recorded::

                          Aspirin   Control/Placebo
        Ischemic stroke     176           230
        No stroke         21035         21018

    Is there evidence that the aspirin reduces the risk of ischemic stroke?
    We begin by formulating a null hypothesis :math:`H_0`:

        The effect of aspirin is equivalent to that of placebo.

    Let's assess the plausibility of this hypothesis with
    a chi-square test.

    >>> import numpy as np
    >>> from scipy.stats import chi2_contingency
    >>> table = np.array([[176, 230], [21035, 21018]])
    >>> res = chi2_contingency(table)
    >>> res.statistic
    6.892569132546561
    >>> res.pvalue
    0.008655478161175739

    Using a significance level of 5%, we would reject the null hypothesis in
    favor of the alternative hypothesis: "the effect of aspirin
    is not equivalent to the effect of placebo".
    Because `scipy.stats.contingency.chi2_contingency` performs a two-sided
    test, the alternative hypothesis does not indicate the direction of the
    effect. We can use `stats.contingency.odds_ratio` to support the
    conclusion that aspirin *reduces* the risk of ischemic stroke.

    Below are further examples showing how larger contingency tables can be
    tested.

    A two-way example (2 x 3):

    >>> obs = np.array([[10, 10, 20], [20, 20, 20]])
    >>> res = chi2_contingency(obs)
    >>> res.statistic
    2.7777777777777777
    >>> res.pvalue
    0.24935220877729619
    >>> res.dof
    2
    >>> res.expected_freq
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    Perform the test using the log-likelihood ratio (i.e. the "G-test")
    instead of Pearson's chi-squared statistic.

    >>> res = chi2_contingency(obs, lambda_="log-likelihood")
    >>> res.statistic
    2.7688587616781319
    >>> res.pvalue
    0.25046668010954165

    A four-way example (2 x 2 x 2 x 2):

    >>> obs = np.array(
    ...     [[[[12, 17],
    ...        [11, 16]],
    ...       [[11, 12],
    ...        [15, 16]]],
    ...      [[[23, 15],
    ...        [30, 22]],
    ...       [[14, 17],
    ...        [15, 16]]]])
    >>> res = chi2_contingency(obs)
    >>> res.statistic
    8.7584514426741897
    >>> res.pvalue
    0.64417725029295503
    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    if np.any(expected == 0):
        # Include one of the positions where expected is zero in
        # the exception message.
        zeropos = list(zip(*np.nonzero(expected == 0)))[0]
        raise ValueError("The internally computed table of expected "
                         f"frequencies has a zero element at {zeropos}.")

    # The degrees of freedom
    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 0:
        # Degenerate case; this occurs when `observed` is 1D (or, more
        # generally, when it has only one nontrivial dimension).  In this
        # case, we also have observed == expected, so chi2 is 0.
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            # Adjust `observed` according to Yates' correction for continuity.
            # Magnitude of correction no bigger than difference; see gh-13875
            diff = expected - observed
            direction = np.sign(diff)
            magnitude = np.minimum(0.5, np.abs(diff))
            observed = observed + magnitude * direction

        chi2, p = power_divergence(observed, expected,
                                   ddof=observed.size - 1 - dof, axis=None,
                                   lambda_=lambda_)

    return Chi2ContingencyResult(chi2, p, dof, expected)