"""Tests the yaml constructors"""
import numpy as np
import pytest
from ruamel.yaml.constructor import ConstructorError
from yayaml import yaml_dumps

from paramspace import ParamDim, ParamSpace
from paramspace.yaml import yaml

# Fixtures --------------------------------------------------------------------


@pytest.fixture()
def yamlstrs() -> dict:
    """Prepares a list of yaml strings to test against"""
    # NOTE Leading indentation is ignored by yaml
    strs = {
        "pspace_only": """
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
        """,
        "pdims_only": """
            pdims:
             - !sweep
               default: 0
               values: [1,2,3]
             - !sweep
               default: 0
               range: [10]
             - !sweep
               default: 0
               linspace: [1,2,3]
             - !sweep
               default: 0
               logspace: [1,2,3]
             - !sweep-default
               default: 0
               values: [1,2,3]
        """,
        "cpdims_only": """
            pdims:
             - !coupled-sweep
               target_name: [foo, bar]
             - !coupled-sweep
               target_name: [foo, bar]
             - !coupled-sweep
               target_name: [foo, bar]
             - !coupled-sweep
               target_name: [foo, bar]
             - !coupled-sweep-default
               target_name: [foo, bar]
               default: 0
        """,
        "pspace_and_pdims": """
            pspace: !pspace
              foo: bar
              spam:
                fish: 123
              some:
                nested:
                  param: !sweep
                    default: 10
                    values: [1,2,3]
                    name: some_nested_param
              another:
                nesting:
                  - !sweep
                    default: 0
                    range: [0, 20, 2]
                  - !coupled-sweep
                    target_name: some_nested_param
                  - !coupled-sweep-default
                    target_name: some_nested_param
                    default: 123
        """,
        #
        # Failing or warning cases
        ("_pspace_scalar", TypeError): "scalar_node: !pspace 1",
        ("_pdim1", TypeError): "not_a_mapping: !sweep 1",
        ("_pdim2", TypeError): "not_a_mapping: !sweep [1,2,3]",
        ("_pdim3", TypeError): "wrong_args: !sweep {foo: bar}",
        ("cpdim1", TypeError): "not_a_mapping: !coupled-sweep 1",
        ("cpdim2", TypeError): "not_a_mapping: !coupled-sweep [1,2,3]",
        ("cpdim3", TypeError): "wrong_args: !coupled-sweep {foo: bar}",
        (
            "cpdim4",
            None,
            DeprecationWarning,
        ): """
            too_many_args: !coupled-sweep
              target_name: [foo, bar]
              default: 0
              use_coupled_default: True
        """,
        (
            "cpdim5",
            None,
            DeprecationWarning,
        ): """
            too_many_args: !coupled-sweep
              target_name: [foo, bar]
              values: [1,2,3]
              use_coupled_values: True
        """,
    }

    return strs


# -- Tests --------------------------------------------------------------------


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
        output = yaml_dumps(obj)

        # And that it uses the expected YAML tags
        if name == "pspace_only":
            assert output.count("!pspace") == 3

        elif name == "pdims_only":
            assert output.count("!sweep") == 4
            assert output.count("!pdim") == 0

        elif name == "cpdims_only":
            assert output.count("!coupled-sweep") == 4
            assert output.count("!coupled-pdim") == 0


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
    pdims = res["pdims_only"]["pdims"]
    assert all(isinstance(pd, ParamDim) for pd in pdims[:4])

    assert pdims[0].default == 0
    assert pdims[0].values == (1, 2, 3)

    assert pdims[1].default == 0
    assert pdims[1].values == tuple(range(10))

    assert pdims[2].default == 0
    assert pdims[2].values == tuple(np.linspace(1, 2, 3))

    assert pdims[3].default == 0
    assert pdims[3].values == tuple(np.logspace(1, 2, 3))

    # ... defaulted
    assert pdims[4] == 0
    assert not isinstance(pdims[4], ParamDim)
    assert pdims[4] == 0

    # Test the ParamSpace's
    for psp in res["pspace_only"].values():
        assert isinstance(psp, ParamSpace)
