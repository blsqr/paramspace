"""Tests the yaml constructors"""
import io
import operator

import numpy as np
import pytest

from paramspace import ParamDim, ParamSpace
from paramspace.yaml import *

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
        """,
        "cpdims_only": """
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
        """,
        "slice": """
            slices:
             - !slice 5
             - !slice [5]
             - !slice [0, ~]
             - !slice [~, 0]
             - !slice [0, 10, 2]
             - !slice [0, 10, None]
             - !slice [2, None, 2]
        """,
        "range": """
            ranges:
             - !range 10
             - !range [10]
             - !range [5, 10]
             - !range [5, 10, 2]
        """,
        "listgen": """
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
        """,
        "copy": """
            copy:
              foo: !deepcopy &foo
                bar: baz
              foo2:
                <<: *foo
              seq: !deepcopy
               - 1
               - 2
              scalar: !deepcopy 123
        """,
        "utils": """
            utils:
              any:      !any [false, false, false, true]
              all:      !all [true, true, true]
              abs:      !abs -1
              int:      !int 1.23
              round:    !round 1.23
              min:      !min [1,2,3]
              max:      !max [1,2,3]
              sorted:   !sorted [2,1,3]
              isorted:  !isorted [2,1,3]
              sum:      !sum [1,2,3]
              prod:     !prod [1,2,3]
              add:      !add [1,2]
              sub:      !sub [2,1]
              mul:      !mul [3,4]
              truediv:  !truediv [3,2]
              floordiv: !floordiv [3,2]
              mod:      !mod [3,2]
              pow:      !pow [2,4]
              pow3:     !pow [2,4,3]
              pow3d:    !pow {x: 2, y: 4, z: 3}
              not:      !not [true]
              and:      !and [true, false]
              or:       !or [true, false]
              xor:      !xor [true, true]
              lt:       !lt [1, 2]
              le:       !le [2, 2]
              eq:       !eq [3, 3]
              ne:       !ne [3, 1]
              ge:       !ge [1, -1]
              gt:       !gt [1, -1]
              negate:   !negate [1]
              invert:   !invert [true]
              contains: !contains [[1,2,3], a]
              concat:   !concat [[1,2,3], [4,5,6], [7,8,9]]
        """,
        #
        # Failing or warning cases
        ("_pspace_scalar", TypeError): "scalar_node: !pspace 1",
        ("_pdim1", TypeError): "not_a_mapping: !pdim 1",
        ("_pdim2", TypeError): "not_a_mapping: !pdim [1,2,3]",
        ("_pdim3", TypeError): "wrong_args: !pdim {foo: bar}",
        ("cpdim1", TypeError): "not_a_mapping: !coupled-pdim 1",
        ("cpdim2", TypeError): "not_a_mapping: !coupled-pdim [1,2,3]",
        ("cpdim3", TypeError): "wrong_args: !coupled-pdim {foo: bar}",
        (
            "cpdim4",
            None,
            DeprecationWarning,
        ): """
            too_many_args: !coupled-pdim
              target_name: [foo, bar]
              default: 0
              use_coupled_default: True
        """,
        (
            "cpdim5",
            None,
            DeprecationWarning,
        ): """
            too_many_args: !coupled-pdim
              target_name: [foo, bar]
              values: [1,2,3]
              use_coupled_values: True
        """,
        ("_listgen_scalar", TypeError): "scalar_node: !listgen foo",
    }

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
    pdims = res["pdims_only"]["pdims"]

    assert pdims[0].default == 0
    assert pdims[0].values == (1, 2, 3)

    assert pdims[1].default == 0
    assert pdims[1].values == tuple(range(10))

    assert pdims[2].default == 0
    assert pdims[2].values == tuple(np.linspace(1, 2, 3))

    assert pdims[3].default == 0
    assert pdims[3].values == tuple(np.logspace(1, 2, 3))

    assert pdims[4] == 0

    # Test the ParamSpace's
    for psp in res["pspace_only"].values():
        assert isinstance(psp, ParamSpace)

    # Test the utility constructors
    utils = res["utils"]["utils"]
    assert utils["any"] == any([False, False, False, True])
    assert utils["all"] == all([True, True, True])
    assert utils["abs"] == abs(-1)
    assert utils["int"] == int(1.23)
    assert utils["round"] == round(1.23)
    assert utils["min"] == min([1, 2, 3])
    assert utils["max"] == max([1, 2, 3])
    assert utils["sorted"] == sorted([2, 1, 3])
    assert utils["isorted"] == sorted([2, 1, 3], reverse=True)
    assert utils["sum"] == sum([1, 2, 3])
    assert utils["prod"] == 1 * 2 * 3
    assert utils["add"] == operator.add(*[1, 2])
    assert utils["sub"] == operator.sub(*[2, 1])
    assert utils["mul"] == operator.mul(*[3, 4])
    assert utils["truediv"] == operator.truediv(*[3, 2])
    assert utils["floordiv"] == operator.floordiv(*[3, 2])
    assert utils["mod"] == operator.mod(*[3, 2])
    assert utils["pow"] == 2 ** 4
    assert utils["pow3"] == 2 ** 4 % 3 == pow(2, 4, 3)
    assert utils["pow3d"] == 2 ** 4 % 3 == pow(2, 4, 3)
    assert utils["not"] == operator.not_(*[True])
    assert utils["and"] == operator.and_(*[True, False])
    assert utils["or"] == operator.or_(*[True, False])
    assert utils["xor"] == operator.xor(*[True, True])
    assert utils["lt"] == operator.lt(*[1, 2])
    assert utils["le"] == operator.le(*[2, 2])
    assert utils["eq"] == operator.eq(*[3, 3])
    assert utils["ne"] == operator.ne(*[3, 1])
    assert utils["ge"] == operator.ge(*[1, -1])
    assert utils["gt"] == operator.gt(*[1, -1])
    assert utils["negate"] == operator.neg(*[1])
    assert utils["invert"] == operator.invert(*[True])
    assert utils["contains"] == operator.contains(*[[1, 2, 3], "a"])
    assert utils["concat"] == [1, 2, 3] + [4, 5, 6] + [7, 8, 9]
