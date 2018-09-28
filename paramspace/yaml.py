"""This module adds yaml constructors for ParamSpace and ParamDim generation"""

from ruamel.yaml import YAML

from .yaml_constructors import pspace, pspace_unsorted
from .yaml_constructors import pdim, pdim_default
from .yaml_constructors import coupled_pdim, coupled_pdim_default

# -----------------------------------------------------------------------------
# Define the ruamel.yaml YAML object which is globally used
yaml = YAML(typ='safe')
yaml.default_flow_style = True

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

# Iterate over the list to add the constructors to the YAML object
for tag, constr_func in _constructors:
    yaml.constructor.add_constructor(tag, constr_func)
