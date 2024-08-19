import sys as _sys
from keyword import iskeyword as _iskeyword
from operator import itemgetter as _itemgetter
try:
    from _collections import _tuplegetter
except ImportError:
    _tuplegetter = lambda index, doc: property(_itemgetter(index), doc=doc)

def _validate_names(typename, field_names, extra_field_names):
    """
    Ensure that all the given names are valid Python identifiers that
    do not start with '_'.  Also check that there are no duplicates
    among field_names + extra_field_names.
    """
    for name in [typename] + field_names + extra_field_names:
        if not isinstance(name, str):
            raise TypeError('typename and all field names must be strings')
        if not name.isidentifier():
            raise ValueError('typename and all field names must be valid '
                             f'identifiers: {name!r}')
        if _iskeyword(name):
            raise ValueError('typename and all field names cannot be a '
                             f'keyword: {name!r}')

    seen = set()
    for name in field_names + extra_field_names:
        if name.startswith('_'):
            raise ValueError('Field names cannot start with an underscore: '
                             f'{name!r}')
        if name in seen:
            raise ValueError(f'Duplicate field name: {name!r}')
        seen.add(name)

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

def namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    """

    # Validate the field names.  At the user's option, either generate an error
    # message or automatically replace the field name with a valid name.
    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = list(map(str, field_names))
    typename = _sys.intern(str(typename))

    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            if (not name.isidentifier()
                or _iskeyword(name)
                or name.startswith('_')
                or name in seen):
                field_names[index] = f'_{index}'
            seen.add(name)

    for name in [typename] + field_names:
        if type(name) is not str:
            raise TypeError('Type names and field names must be strings')
        if not name.isidentifier():
            raise ValueError('Type names and field names must be valid '
                             f'identifiers: {name!r}')
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a '
                             f'keyword: {name!r}')

    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: '
                             f'{name!r}')
        if name in seen:
            raise ValueError(f'Encountered duplicate field name: {name!r}')
        seen.add(name)

    field_defaults = {}
    if defaults is not None:
        defaults = tuple(defaults)
        if len(defaults) > len(field_names):
            raise TypeError('Got more default values than field names')
        field_defaults = dict(reversed(list(zip(reversed(field_names),
                                                reversed(defaults)))))

    # Variables used in the methods and docstrings
    field_names = tuple(map(_sys.intern, field_names))
    num_fields = len(field_names)
    arg_list = ', '.join(field_names)
    if num_fields == 1:
        arg_list += ','
    repr_fmt = '(' + ', '.join(f'{name}=%r' for name in field_names) + ')'
    tuple_new = tuple.__new__
    _dict, _tuple, _len, _map, _zip = dict, tuple, len, map, zip

    # Create all the named tuple methods to be added to the class namespace

    namespace = {
        '_tuple_new': tuple_new,
        '__builtins__': {},
        '__name__': f'namedtuple_{typename}',
    }
    code = f'lambda _cls, {arg_list}: _tuple_new(_cls, ({arg_list}))'
    __new__ = eval(code, namespace)
    __new__.__name__ = '__new__'
    __new__.__doc__ = f'Create new instance of {typename}({arg_list})'
    if defaults is not None:
        __new__.__defaults__ = defaults

    @classmethod
    def _make(cls, iterable):
        result = tuple_new(cls, iterable)
        if _len(result) != num_fields:
            raise TypeError(f'Expected {num_fields} arguments, got {len(result)}')
        return result

    _make.__func__.__doc__ = (f'Make a new {typename} object from a sequence '
                              'or iterable')

    def _replace(self, /, **kwds):
        result = self._make(_map(kwds.pop, field_names, self))
        if kwds:
            raise ValueError(f'Got unexpected field names: {list(kwds)!r}')
        return result

    _replace.__doc__ = (f'Return a new {typename} object replacing specified '
                        'fields with new values')

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + repr_fmt % self

    def _asdict(self):
        'Return a new dict which maps field names to their values.'
        return _dict(_zip(self._fields, self))

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return _tuple(self)

    # Modify function metadata to help with introspection and debugging
    for method in (
        __new__,
        _make.__func__,
        _replace,
        __repr__,
        _asdict,
        __getnewargs__,
    ):
        method.__qualname__ = f'{typename}.{method.__name__}'

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '_fields': field_names,
        '_field_defaults': field_defaults,
        '__new__': __new__,
        '_make': _make,
        '_replace': _replace,
        '__repr__': __repr__,
        '_asdict': _asdict,
        '__getnewargs__': __getnewargs__,
        '__match_args__': field_names,
    }
    for index, name in enumerate(field_names):
        doc = _sys.intern(f'Alias for field number {index}')
        class_namespace[name] = _tuplegetter(index, doc)

    result = type(typename, (tuple,), class_namespace)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython), or where the user has
    # specified a particular module.
    if module is None:
        try:
            module = _sys._getframemodulename(1) or '__main__'
        except AttributeError:
            try:
                module = _sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                pass
    if module is not None:
        result.__module__ = module

    return result

