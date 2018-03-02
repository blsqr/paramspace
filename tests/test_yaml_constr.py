"""Tests the yaml constructors"""

import yaml

import pytest
from paramspace import ParamSpace, ParamDim, yaml_constructors

# Add the constructors
yaml.add_constructor(u'!pspace', yaml_constructors.pspace)
yaml.add_constructor(u'!pspace-sorted', yaml_constructors.pspace_sorted)

yaml.add_constructor(u'!pdim', yaml_constructors.pdim)
yaml.add_constructor(u'!pdim-if-enabled', yaml_constructors.pdim_enabled_only)
yaml.add_constructor(u'!pdim-disabled', yaml_constructors.pdim_get_default)
yaml.add_constructor(u'!pdim-default', yaml_constructors.pdim_always_disable)


@pytest.fixture(scope='module')
def yamlstrs(request) -> dict:
    """Prepares a list of yaml strings to test agains"""
    strs = {}

    strs['pspace_only'] = """
sequence: !pspace
  - 1
  - 2
  - 3
mapping: !pspace 
  a: 1
  b: 2
  c: 3
sequence_sorted: !pspace-sorted  # should not actually be sorted!
  - 1
  - 2
  - 3
  - foo:
    bar: 1
    baz: 2
mapping_sorted: !pspace-sorted
  a: 1
  c: 3
  b: 2
  foo:
    bar: 1
    baz: 2
    """

    strs['pdims_only'] = """
pdims:
 - !pdim
   default: 0
   values: [1,2,3]
 - !pdim
   default: 0
   range: [10]
 - !pdim
   default: 0
   linspace: [1,2,3]
 - !pdim
   default: 0
   logspace: [1,2,3]
 - !pdim-if-enabled
   default: 0
   values: [1,2,3]
 - !pdim-if-enabled
   default: 0
   values: [1,2,3]
   enabled: False
 - !pdim-disabled
   default: 0
   values: [1,2,3]
 - !pdim-default
   default: 0
   values: [1,2,3]
    """

    strs['fail_pspace'] = """not_a_mapping_or_sequence: !pspace 1 """

    strs['fail_pdim1'] = """not_a_mapping: !pdim 1 """
    strs['fail_pdim2'] = """not_a_mapping: !pdim [1,2,3] """
    strs['fail_pdim3'] = """wrong_args: !pdim {foo: bar} """
   
    return strs

def test_loading(yamlstrs):
    """Tests whether the constructors loading works."""
    # Test plain loading
    for name, ystr in yamlstrs.items():
        if name[:5] == "fail_":
            continue
        print("Name of yamlstr: ", name)
        yaml.load(ystr)

    # Test some where it should be failing
    with pytest.raises(TypeError):
        yaml.load(yamlstrs['fail_pspace'])
    with pytest.raises(TypeError):
        yaml.load(yamlstrs['fail_pdim1'])
    with pytest.raises(TypeError):
        yaml.load(yamlstrs['fail_pdim3'])

@pytest.mark.skip("To Do!")
def test_correctness(yamlstrs):
    """Tests the correctness"""
    pass