"""Tests the yaml constructors"""

import io

import pytest
import numpy as np

# Add the constructors using the paramspace package methods
from paramspace import ParamSpace, ParamDim
from paramspace.yaml import *

# Fixtures --------------------------------------------------------------------

@pytest.fixture()
def yamlstrs() -> dict:
    """Prepares a list of yaml strings to test agains"""
    strs = {}

    strs['pspace_only'] = """
mapping: !pspace 
  a: 1
  b: 2
  c: 3
mapping_sorted: !pspace
  a: 1
  c: 3
  b: 2
  foo:
    bar: 1
    baz: 2
mapping_unsorted: !pspace-unsorted
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

    strs['slice'] = """
slices:
 - !slice 5
 - !slice [5]
 - !slice [0, ~]
 - !slice [~, 0]
 - !slice [0, 10, 2]
 - !slice [0, 10, None]
 - !slice [2, None, 2]
    """
    
    strs['range'] = """
ranges:
 - !range 10
 - !range [10]
 - !range [5, 10]
 - !range [5, 10, 2]
    """
    
    strs['listgen'] = """
lists:
 - !listgen [10]
 - !listgen [0, 10, 2]
 - !listgen
   from_range: [0, 10, 3]
   unique: true
   append: [100]
   remove: [0]
   sort: true
 - !listgen [5, 10, 2]
    """

    # Failing or warning cases
    strs[('_pspace_scalar', TypeError)] = "scalar_node: !pspace 1"

    strs[('_pdim1', TypeError)] = "not_a_mapping: !pdim 1"
    strs[('_pdim2', TypeError)] = "not_a_mapping: !pdim [1,2,3]"
    strs[('_pdim3', TypeError)] = "wrong_args: !pdim {foo: bar}"

    strs[('cpdim1', TypeError)] = "not_a_mapping: !coupled-pdim 1"
    strs[('cpdim2', TypeError)] = "not_a_mapping: !coupled-pdim [1,2,3]"
    strs[('cpdim3', TypeError)] = "wrong_args: !coupled-pdim {foo: bar}"

    strs[('cpdim4', None, DeprecationWarning)] = """
too_many_args: !coupled-pdim
  target_name: [foo, bar]
  default: 0
  use_coupled_default: True 
    """
    strs[('cpdim5', None, DeprecationWarning)] = """
too_many_args: !coupled-pdim
  target_name: [foo, bar]
  values: [1,2,3]
  use_coupled_values: True 
    """

    strs[("_listgen_scalar", TypeError)] = "scalar_node: !listgen foo"
   
    return strs

# -----------------------------------------------------------------------------
# Tests

def test_load_and_safe(yamlstrs):
    """Tests whether the constructor and representers work"""
    # Test plain loading
    for name, ystr in yamlstrs.items():
        print("\n\nName of yamlstr that will be loaded: ", name)

        if isinstance(name, tuple):
            # Expected to warn or raise
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

            continue

        # else: Expected to load correctly
        obj = yaml.load(ystr)

        # Test the representer runs through
        stream = io.StringIO("")
        yaml.dump(obj, stream=stream)
        output = "\n".join(stream.readlines())

        # TODO Test output

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
