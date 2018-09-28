"""This module adds yaml constructors for ParamSpace and ParamDim generation"""

from ruamel.yaml import YAML

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

# Attach constructors .........................................................
# Define (tag, constructor) pairs
_constructors = [
    (u'!pspace',                pspace),
    (u'!pspace-unsorted',       pspace_unsorted),
    (u'!pdim',                  pdim),
    (u'!pdim-default',          pdim_default),
    (u'!coupled-pdim',          coupled_pdim),
    (u'!coupled-pdim-default',  coupled_pdim_default)
]

# Iterate over the list to add the constructors to all the YAML objects
for tag, constr_func in _constructors:
    yaml.constructor.add_constructor(tag, constr_func)
    yaml_rt.constructor.add_constructor(tag, constr_func)
    yaml_safe.constructor.add_constructor(tag, constr_func)
    yaml_unsafe.constructor.add_constructor(tag, constr_func)


# Attach representers .........................................................
