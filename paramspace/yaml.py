"""This module adds yaml constructors for ParamSpace and ParamDim generation"""

from ruamel.yaml import YAML

from .paramdim import ParamDim, CoupledParamDim, Masked
from .paramspace import ParamSpace

from .yaml_constructors import pspace, pspace_unsorted
from .yaml_constructors import pdim, pdim_default
from .yaml_constructors import coupled_pdim, coupled_pdim_default
from .yaml_constructors import _slice_constructor, _range_constructor
from .yaml_constructors import _list_constructor

from .yaml_representers import _slice_representer, _range_representer

# -----------------------------------------------------------------------------
# Define a safe and an unsafe ruamel.yaml YAML object
yaml_safe = YAML(typ='safe')
yaml_unsafe = YAML(typ='unsafe')

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
    (u'!pspace',                pspace),        # ***
    (u'!pspace-unsorted',       pspace_unsorted),
    (u'!pdim',                  pdim),          # ***
    (u'!pdim-default',          pdim_default),
    (u'!coupled-pdim',          coupled_pdim),  # ***
    (u'!coupled-pdim-default',  coupled_pdim_default),
    #
    # additional constructors for Python objects
    (u'!slice',                 _slice_constructor),
    (u'!range',                 _range_constructor),
    (u'!listgen',               _list_constructor)

]
# NOTE entries marked with '***' overwrite a default constructor. Thus, they
#      need to be defined down here, after the classes and their tags were
#      registered with the YAML objects.

# Add the constructors to all YAML objects
for tag, constr_func in _constructors:
    yaml_safe.constructor.add_constructor(tag, constr_func)
    yaml_unsafe.constructor.add_constructor(tag, constr_func)
