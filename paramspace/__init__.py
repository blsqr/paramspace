"""The paramspace package provides classes to conveniently create parameter
spaces and iterate over them.

To that end, any dict-like object can be filled with `ParamDim` objects to
create parameter dimensions. When passing this dict-like object to
`ParamSpace`, it is possible to iterate over the points in parameter space ...
"""

__version__ = "2.4.0a0"

from paramspace.paramdim import CoupledParamDim, ParamDim
from paramspace.paramspace import ParamSpace
from paramspace.yaml import yaml, yaml_safe, yaml_unsafe
