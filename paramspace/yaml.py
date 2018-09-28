"""This module adds yaml constructors for ParamSpace and ParamDim generation"""

from ruamel.yaml import YAML

from .paramdim import ParamDim, CoupledParamDim
from .paramspace import ParamSpace

from .yaml_constructors import pspace, pspace_unsorted
from .yaml_constructors import pdim, pdim_default
from .yaml_constructors import coupled_pdim, coupled_pdim_default

# -----------------------------------------------------------------------------
# Define numerous ruamel.yaml YAML objects, safe and unsafe
yaml_rt = YAML(typ='rt')
yaml_safe = YAML(typ='safe')
yaml_unsafe = YAML(typ='unsafe')

# Define the default YAML object
yaml = yaml_unsafe


# Attach representers .........................................................
# ... to all YAML objects by registering the classes.

for _yaml in (yaml_rt, yaml_safe, yaml_unsafe):
    _yaml.register_class(ParamDim)
    _yaml.register_class(CoupledParamDim)
    _yaml.register_class(ParamSpace)

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
    (u'!coupled-pdim-default',  coupled_pdim_default)
]
# NOTE entries marked with '***' overwrite a default constructor. Thus, they
#      need to be defined down here, after the classes and their tags were
#      registered with the YAML objects.

# Add the constructors to all YAML objects
for tag, constr_func in _constructors:
    yaml_rt.constructor.add_constructor(tag, constr_func)
    yaml_safe.constructor.add_constructor(tag, constr_func)
    yaml_unsafe.constructor.add_constructor(tag, constr_func)
