"""This module registers various YAML constructors and representers, notably
those for :py:class:`~paramspace.paramspace.ParamSpace` and
:py:class:`~paramspace.paramdim.ParamDim`.

Furthermore, it defines a shared ``ruamel.yaml.YAML`` object that can be
imported and used for loading and storing YAML files using the representers and
constructors.
"""
import copy as _copy
import operator as _operator
from functools import partial as _partial
from functools import reduce as _reduce

import numpy as np
from ruamel.yaml import YAML

from .paramdim import CoupledParamDim, Masked, ParamDim
from .paramspace import ParamSpace
from .tools import recursive_update as _recursive_update
from .yaml_constructors import (
    _func_constructor,
    _list_constructor,
    _range_constructor,
    _slice_constructor,
    coupled_pdim,
    coupled_pdim_default,
    pdim,
    pdim_default,
    pspace,
    pspace_unsorted,
)
from .yaml_representers import _range_representer, _slice_representer

# -----------------------------------------------------------------------------
# Define a safe and an unsafe ruamel.yaml YAML object
yaml_safe = YAML(typ="safe")
yaml_unsafe = YAML(typ="unsafe")

# Define the safe one as default
yaml = yaml_safe


# Attach representers .........................................................
# ... to all YAML objects by registering the classes or by adding the custom
# representer functions

for _yaml in (yaml_safe, yaml_unsafe):
    _yaml.register_class(Masked)
    _yaml.register_class(ParamDim)
    _yaml.register_class(CoupledParamDim)
    _yaml.register_class(ParamSpace)

    _yaml.representer.add_representer(slice, _slice_representer)
    _yaml.representer.add_representer(range, _range_representer)

# NOTE It is important that this happens _before_ the custom constructors are
#      added below, because otherwise it is tried to construct the classes
#      using the (inherited) default constructor (which might not work)


# Attach constructors .........................................................
# Define list of (tag, constructor function) pairs
_constructors = [
    ("!pspace", pspace),  # ***
    ("!pspace-unsorted", pspace_unsorted),
    ("!pdim", pdim),  # ***
    ("!pdim-default", pdim_default),
    ("!coupled-pdim", coupled_pdim),  # ***
    ("!coupled-pdim-default", coupled_pdim_default),
    #
    # additional constructors for Python objects
    ("!slice", _slice_constructor),
    ("!range", _range_constructor),
    ("!listgen", _list_constructor),
]
# NOTE entries marked with '***' overwrite a default constructor. Thus, they
#      need to be defined down here, after the classes and their tags were
#      registered with the YAML objects.

# Programmatically define and add a bunch of utility constructors, which
# evaluate nodes directly during construction. Distinguish between those where
# sequence or mapping arguments are NOT to be unpacked and those where
# unpacking them as positional and/or keyword arguments makes sense.
_util_constructors_no_unpack = [
    # built-ins operating on iterables
    ("!any", any),
    ("!all", all),
    ("!min", min),
    ("!max", max),
    ("!sum", sum),
    ("!prod", lambda a: _reduce(_operator.mul, a, 1)),
    ("!sorted", lambda a: list(sorted(a))),
    ("!isorted", lambda a: list(sorted(a, reverse=True))),
    #
    # built-ins operating on scalars
    ("!abs", lambda v: abs(float(v))),
    ("!int", lambda v: int(float(v))),
    ("!round", lambda v: round(float(v))),
    #
    # misc
    ("!deepcopy", _copy.deepcopy),
]
_util_constructors_unpack = [
    # from operators module
    ("!add", _operator.add),
    ("!sub", _operator.sub),
    ("!mul", _operator.mul),
    ("!truediv", _operator.truediv),
    ("!floordiv", _operator.floordiv),
    ("!mod", _operator.mod),
    ("!pow", lambda x, y, z=None: pow(x, y, z)),
    ("!not", _operator.not_),
    ("!and", _operator.and_),
    ("!or", _operator.or_),
    ("!xor", _operator.xor),
    ("!lt", _operator.lt),
    ("!le", _operator.le),
    ("!eq", _operator.eq),
    ("!ne", _operator.ne),
    ("!ge", _operator.ge),
    ("!gt", _operator.gt),
    ("!negate", _operator.neg),
    ("!invert", _operator.invert),
    ("!contains", _operator.contains),
    ("!concat", lambda *l: _reduce(_operator.concat, l, [])),
    ("!format", lambda fstr, *a, **k: fstr.format(*a, **k)),
    #
    # numpy
    ("!arange", lambda *a: [float(f) for f in np.arange(*a)]),
    ("!linspace", lambda *a: [float(f) for f in np.linspace(*a)]),
    ("!logspace", lambda *a: [float(f) for f in np.logspace(*a)]),
    #
    # misc
    (
        "!rec-update",
        lambda d, u: _recursive_update(_copy.deepcopy(d), _copy.deepcopy(u)),
    ),
]

# Register them
_constructors += [
    (tag, _partial(_func_constructor, func=func, unpack=False))
    for tag, func in _util_constructors_no_unpack
]
_constructors += [
    (tag, _partial(_func_constructor, func=func, unpack=True))
    for tag, func in _util_constructors_unpack
]

# Now, add all the above constructors to all YAML objects
for tag, constr_func in _constructors:
    yaml_safe.constructor.add_constructor(tag, constr_func)
    yaml_unsafe.constructor.add_constructor(tag, constr_func)
