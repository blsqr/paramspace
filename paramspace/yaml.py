"""This module registers various YAML constructors and representers, notably
those for :py:class:`~paramspace.paramspace.ParamSpace` and
:py:class:`~paramspace.paramdim.ParamDim`.

Furthermore, it defines a shared ``ruamel.yaml.YAML`` object that can be
imported and used for loading and storing YAML files using the representers and
constructors.
"""
from ruamel.yaml import YAML

from .paramdim import CoupledParamDim, Masked, ParamDim
from .paramspace import ParamSpace
from .yaml_constructors import (
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

# Add the constructors to all YAML objects
for tag, constr_func in _constructors:
    yaml_safe.constructor.add_constructor(tag, constr_func)
    yaml_unsafe.constructor.add_constructor(tag, constr_func)
