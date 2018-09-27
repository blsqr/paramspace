"""Tests the yaml constructors"""

import pytest

import numpy as np
import yaml

# Add the constructors using the paramspace package methods
from paramspace import ParamSpace, ParamDim, yaml_constructors

yaml.add_constructor(u'!pspace', yaml_constructors.pspace)
yaml.add_constructor(u'!pspace-sorted', yaml_constructors.pspace_sorted)

yaml.add_constructor(u'!pdim',
                     yaml_constructors.pdim)
yaml.add_constructor(u'!pdim-default',
                     yaml_constructors.pdim_get_default)

yaml.add_constructor(u'!coupled-pdim',
                     yaml_constructors.coupled_pdim)
yaml.add_constructor(u'!coupled-pdim-default',
                     yaml_constructors.coupled_pdim_get_default)


# Fixtures --------------------------------------------------------------------

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
sequence_sorted: !pspace-sorted  # is a sequence, should not have an effect
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
 - !pdim-default
   default: 0
   values: [1,2,3]
    """

    strs['cpdims_only'] = """
pdims:
 - !coupled-pdim
   target_name: [foo, bar]
 - !coupled-pdim
   target_name: [foo, bar]
 - !coupled-pdim
   target_name: [foo, bar]
 - !coupled-pdim
   target_name: [foo, bar]
 - !coupled-pdim-default
   target_name: [foo, bar]
   default: 0
    """

    strs[('pspace', TypeError)] = """not_a_mapping_or_sequence: !pspace 1 """

    strs[('_pdim1', TypeError)]  = """not_a_mapping: !pdim 1 """
    strs[('_pdim2', TypeError)]  = """not_a_mapping: !pdim [1,2,3] """
    strs[('_pdim3', TypeError)]  = """wrong_args: !pdim {foo: bar} """

    strs[('cpdim1', TypeError)] = """not_a_mapping: !coupled-pdim 1 """
    strs[('cpdim2', TypeError)] = """not_a_mapping: !coupled-pdim [1,2,3] """
    strs[('cpdim3', TypeError)] = """wrong_args: !coupled-pdim {foo: bar} """
    strs[('cpdim4', None, UserWarning)] = """
too_many_args: !coupled-pdim
  target_name: [foo, bar]
  default: 0
  use_coupled_default: True 
    """
    strs[('cpdim5', None, UserWarning)] = """
too_many_args: !coupled-pdim
  target_name: [foo, bar]
  values: [1,2,3]
  use_coupled_values: True 
    """
   
    return strs

def test_loading(yamlstrs):
    """Tests whether the constructors loading works."""
    # Test plain loading
    for name, ystr in yamlstrs.items():
        print("Name of yamlstr that will be loaded: ", name)
        if isinstance(name, tuple):
            if len(name) == 2:
                name, exc = name
                warn = None
            elif len(name) == 3:
                name, exc, warn = name

            # Distinguish three cases
            if warn and exc:
                with pytest.raises(exc):
                    with pytest.warns(warn):
                        yaml.load(ystr)

            elif warn and not exc:
                with pytest.warns(warn):
                    yaml.load(ystr)

            elif exc and not warn:
                with pytest.raises(exc):
                    yaml.load(ystr)
        else:
            yaml.load(ystr)

def test_correctness(yamlstrs):
    """Tests the correctness of the constructors"""
    res = {}

    # Load the resolved yaml strings
    for name, ystr in yamlstrs.items():
        print("Name of yamlstr that will be loaded: ", name)
        if isinstance(name, tuple):
            # Will fail, don't use
            continue
        res[name] = yaml.load(ystr)

    # Test the ParamDim objects
    pdims = res['pdims_only']['pdims']

    assert pdims[0].default == 0
    assert pdims[0].values == (1,2,3)

    assert pdims[1].default == 0
    assert pdims[1].values == tuple(range(10))

    assert pdims[2].default == 0
    assert pdims[2].values == tuple(np.linspace(1,2,3))

    assert pdims[3].default == 0
    assert pdims[3].values == tuple(np.logspace(1,2,3))

    assert pdims[4] == 0

    # Test the ParamSpace's
    for psp in res['pspace_only'].values():
        assert isinstance(psp, ParamSpace)
