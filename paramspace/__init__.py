"""This package provides classes to conveniently define hierarchically
structured parameter spaces and iterate over them.

To that end, any dict-like object can be populated with
:py:class:`~paramspace.paramdim.ParamDim` objects to create a parameter
dimension at that key. When creating a
:py:class:`~paramspace.paramspace.ParamSpace` from this dict, it becomes
possible to iterate over all points in the space created by the parameter
dimensions, i.e. the *parameter space*.

Furthermore, the :py:mod:`paramspace.yaml` module provides possibilities to
define the parameter space fully from YAML configuration files, using custom
YAML tags.
"""

__version__ = "2.6.2"

from .paramdim import CoupledParamDim, ParamDim
from .paramspace import ParamSpace
from .yaml import yaml, yaml_safe, yaml_unsafe
