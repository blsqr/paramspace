"""Tests the yaml constructors"""

import yaml

import pytest
from paramspace import ParamSpace, ParamDim, yaml_constructors

# Add the constructors
yaml.add_constructor(u'!pspace', yaml_constructors.pspace)
yaml.add_constructor(u'!pspace-sorted', yaml_constructors.pspace)

yaml.add_constructor(u'!pdim', yaml_constructors.pdim)
yaml.add_constructor(u'!pdim-if-enabled', yaml_constructors.pdim_enabled_only)
yaml.add_constructor(u'!pdim-disabled', yaml_constructors.pdim_get_default)
yaml.add_constructor(u'!pdim-default', yaml_constructors.pdim_always_disable)
