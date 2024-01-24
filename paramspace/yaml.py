"""This module makes YAML-related tools and functionality available.

The functionality itself is implemented in :py:mod:`yayaml`, this module only
adds additional constructors and representers for this package.
"""

from ruamel.yaml import YAML
from yayaml import yaml, yaml_safe

from .paramdim import CoupledParamDim, Masked, ParamDim
from .paramspace import ParamSpace
from .yaml_constructors import *
from .yaml_representers import *

# -- Register classes ---------------------------------------------------------
# ... to all YAML objects by registering the classes or by adding the custom
# representer functions

assert yaml is yaml_safe

yaml.register_class(Masked)
yaml.register_class(ParamDim)
yaml.register_class(CoupledParamDim)
yaml.register_class(ParamSpace)
