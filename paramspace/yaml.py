"""This module registers various YAML constructors and representers, notably
those for :py:class:`~paramspace.paramspace.ParamSpace` and
:py:class:`~paramspace.paramdim.ParamDim`.

Furthermore, it defines a shared ``ruamel.yaml.YAML`` object that can be
imported and used for loading and storing YAML files using the representers and
constructors.
"""

from ruamel.yaml import YAML
from yayaml import yaml, yaml_safe, yaml_unsafe

from .paramdim import CoupledParamDim, Masked, ParamDim
from .paramspace import ParamSpace
from .yaml_constructors import *
from .yaml_representers import *

# -- Register classes ---------------------------------------------------------
# ... to all YAML objects by registering the classes or by adding the custom
# representer functions

for _yaml in (yaml_safe, yaml_unsafe):
    _yaml.register_class(Masked)
    _yaml.register_class(ParamDim)
    _yaml.register_class(CoupledParamDim)
    _yaml.register_class(ParamSpace)
