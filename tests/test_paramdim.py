"""Tests for the ParamDim classes"""
import warnings

import numpy as np
import pytest

from paramspace import CoupledParamDim, ParamDim
from paramspace.paramdim import Masked, MaskedValueError
from paramspace.yaml import *

# Setup methods ---------------------------------------------------------------


@pytest.fixture()
def various_pdims():
    """Used to setup various pspan objects to be tested on."""
    pds = {}

    pds["one"] = ParamDim(default=0, values=[1, 2, 3])
    pds["two"] = ParamDim(default=0, values=[1.0, 2, "three", np.inf])
    pds["range"] = ParamDim(default=0, range=[1, 4, 1])
    pds["linspace"] = ParamDim(default=0, linspace=[1, 3, 3, True])
    pds["logspace"] = ParamDim(default=0, logspace=[-1, 1, 11])
    pds["typed"] = ParamDim(default=0, range=[3], as_type="float")
    pds["a_list"] = ParamDim(default=0, values=[[1, 2], [3, 4]])
    pds["as_tuple"] = ParamDim(
        default=0, values=[1, [2], [3, 4, [5]]], as_type="tuple"
    )
    pds["named"] = ParamDim(default=0, values=[1, 2, 3], name="named_span")
    pds["with_order"] = ParamDim(default=0, values=[1, 2, 3], order=42)

    return pds


# Tests -----------------------------------------------------------------------


def test_init():
    """Test whether all initialisation methods work."""
    # These should all work
    ParamDim(default=0, values=[1, 2, 3])
    ParamDim(default=0, values=[1.0, 2, "three", np.inf])
    ParamDim(default=0, range=[1, 4, 1])
    ParamDim(default=0, linspace=[1, 3, 3, True])
    ParamDim(default=0, logspace=[-1, 1, 11])
    ParamDim(default=0, range=[3], as_type="float")
    ParamDim(default=0, values=[[1, 2], [3, 4]])
    ParamDim(default=0, values=[1, [2], [3, 4, [5]]], as_type="tuple")
    ParamDim(default=0, values=[1, 2, 3], name="named_span")
    ParamDim(default=0, values=[1, 2, 3], order=42)

    # No default given
    with pytest.raises(TypeError):
        ParamDim()

    # No values given
    with pytest.raises(TypeError, match="Missing one of the following"):
        ParamDim(default=0)

    with pytest.raises(ValueError, match="need be a container of length >= 1"):
        ParamDim(default=0, values=[])

    with pytest.raises(TypeError, match="Received invalid keyword argument"):
        ParamDim(default=0, foo="bar")

    # Multiple values or kwargs given
    with pytest.raises(TypeError, match="Received too many keyword arguments"):
        ParamDim(default=0, values=[1, 2, 3], linspace=[10, 20, 30])

    with pytest.raises(TypeError, match="Received too many keyword arguments"):
        ParamDim(default=0, range=[1, 2], linspace=[10, 20, 30])


def test_values(various_pdims):
    """Assert the correct values are chosen"""
    vpd = various_pdims

    assert vpd["range"].values == tuple(range(1, 4, 1))
    assert all(vpd["linspace"].values == np.linspace(1, 3, 3, True))
    assert all(vpd["logspace"].values == np.logspace(-1, 1, 11))

    # Test the as_type argument
    assert isinstance(vpd["a_list"].values[0], list)
    assert isinstance(vpd["typed"].values[0], float)
    assert vpd["as_tuple"].values == (1, (2,), (3, 4, (5,)))

    with pytest.raises(KeyError, match="some_type"):
        ParamDim(default=0, range=[10], as_type="some_type")

    # Assert that values are unique
    with pytest.raises(ValueError, match="need to be unique, but there were"):
        ParamDim(default=0, values=[1, 1, 2])


def test_properties(various_pdims):
    """Test all properties and whether they are write-protected."""
    vpd = various_pdims

    # Whether the values are write-protected
    with pytest.raises(AttributeError, match="can't set attribute"):
        vpd["one"].values = 0

    with pytest.raises(AttributeError, match="can't set attribute"):
        vpd["two"].values = [1, 2, 3]

    # Assert immutability of values
    with pytest.raises(TypeError, match="does not support item assignment"):
        vpd["one"].values[0] = "foo"

    with pytest.raises(TypeError, match="does not support item assignment"):
        vpd["two"].values[1] = "bar"

    # Whether the state is restricted to the value bounds
    with pytest.raises(ValueError, match="needs to be >= 0, was -1"):
        vpd["one"].state = -1

    with pytest.raises(ValueError, match="needs to be <= 4, was 5"):
        vpd["two"].state = 5

    with pytest.raises(TypeError, match="can only be of type int"):
        vpd["two"].state = "foo"

    # current_value
    assert vpd["one"].current_value == vpd["one"].default
    assert vpd["two"].current_value == vpd["two"].default

    # number of values and length (same in unmasked case)
    assert vpd["one"].num_values == 3 == len(vpd["one"])
    assert vpd["two"].num_values == 4 == len(vpd["two"])

    # target_of
    for pd in vpd.values():
        if isinstance(pd, ParamDim):
            # Can be a target of a coupled ParamDim
            pd.target_of

    # Check that the default value is masked:
    assert isinstance(vpd["one"].default, Masked)


def test_iteration(various_pdims):
    """Tests whether the iteration over the span's state works."""
    pd = ParamDim(default=0, values=[1, 2, 3])

    # Should start in default state
    assert pd.state == 0

    # First iteration
    assert pd.__next__() == 1
    assert pd.__next__() == 2
    assert pd.__next__() == 3
    with pytest.raises(StopIteration):
        pd.__next__()

    # Should be able to iterate again
    assert pd.__next__() == 1
    assert pd.__next__() == 2
    assert pd.__next__() == 3
    with pytest.raises(StopIteration):
        pd.__next__()

    # State should be reset to 0 now
    assert pd.state == 0

    # And as a loop
    for _ in pd:
        continue


def test_str_methods(various_pdims):
    """Run through the string methods, just to call them..."""
    # Whether string representation works ok -- mainly for coverage here
    for pd in various_pdims.values():
        str(pd)
        repr(pd)


def test_np_methods_return_floats():
    """Assert that when using linspace or logspace, the values are floats and
    _not_ numpy scalar types.
    """
    pds = [
        ParamDim(default=0, linspace=[0, 10, 11]),
        ParamDim(default=0, logspace=[0, 10, 11]),
    ]

    for pd in pds:
        types = [type(v) for v in pd.values]
        print("Types: " + str(types))
        assert all([t is float for t in types])


def test_mask():
    """Test that masking works"""
    # Test initialization, property getter and setter, and type
    pd = ParamDim(default=0, values=[1, 2, 3, 4], mask=False)
    assert pd.mask is False
    # NOTE not trivial to test because the .mask getter _computes_ the value
    assert not any([isinstance(v, Masked) for v in pd.values])

    pd.mask = True
    assert pd.mask is True
    assert all([isinstance(v, Masked) for v in pd.values])

    # Check the string representation of masked values
    assert str(pd.values[0]).find("<1>") > -1
    assert str(pd).find("Masked object, value:") > -1

    # Now to a more complex mask
    pd.mask = (True, False, True, False)
    assert pd.mask == (True, False, True, False)

    # Test the properties that inform about the number of masked and unmasked
    # values
    assert len(pd) == 2
    assert pd.num_masked == 2

    # Setting one with a bad length should not work
    with pytest.raises(ValueError, match="container of same length as the"):
        pd.mask = (True, False)

    # Check that iteration starts at first unmasked state
    pd.enter_iteration()
    assert pd.state == 2
    assert pd.current_value == 2

    # Iterate one step, this should jump to index and value 3
    assert pd.__next__() == 4
    assert pd.state == 4

    # Setting the state manually to something masked should not work
    with pytest.raises(MaskedValueError, match="Value at index 1 is masked"):
        pd.state = 1

    # No further iteration should be possible for this one
    with pytest.raises(StopIteration):
        pd.iterate_state()
    assert pd.state == 0

    # Check iteration again
    assert list(pd) == [2, 4]

    # For fully masked, the default value should be returned. Eff. length: 1
    pd.mask = True
    assert len(pd) == 1
    assert pd.num_masked == pd.num_states - 1
    assert list(pd) == [0]

    # Try using a slice to set the mask
    pd.mask = slice(2)
    assert pd.mask == (True, True, False, False)

    pd.mask = slice(2, None)
    assert pd.mask == (False, False, True, True)

    pd.mask = slice(None, None, 2)
    assert pd.mask == (True, False, True, False)


# CoupledParamDim -------------------------------------------------------------


def test_cpd_init():
    """Test whether initialisation of CoupledParamDim works"""
    # These should work
    CoupledParamDim(target_name=("foo",))
    CoupledParamDim(target_name=("foo",), default=0)
    CoupledParamDim(target_name=("foo",), values=[1, 2, 3])
    CoupledParamDim(target_name=("foo",), range=[3])
    CoupledParamDim(target_name=("foo",), linspace=[0, 1, 3])
    CoupledParamDim(target_name=("foo",), logspace=[0, 2, 3])
    CoupledParamDim(target_name="foo")

    # These should fail due to wrong arguments given
    with pytest.raises(TypeError, match="Expected either argument"):
        # Neither target_pdim nor target_name given
        CoupledParamDim()

    with pytest.raises(ValueError, match="The coupling target has not been"):
        # Not coupled yet
        CoupledParamDim(target_name=("foo",)).default

    with pytest.raises(TypeError, match="Got both `target_pdim` and"):
        # Got both target_pdim and target_name
        CoupledParamDim(
            target_pdim=ParamDim(default=0, values=[1, 2, 3]),
            target_name=["foo", "bar"],
        )

    with pytest.raises(TypeError, match="should be a tuple or list"):
        # Bad target_name type
        CoupledParamDim(target_name=dict(foo="bar"))

    # Set target
    pd = ParamDim(default=0, values=[1, 2, 3])
    cpd = CoupledParamDim(target_pdim=pd)
    assert len(pd) == len(cpd)
    assert pd.values == cpd.values
    assert pd.default == cpd.default
    assert cpd.target_name is None

    # Test if the name behaviour is correct
    with pytest.raises(ValueError, match="name cannot be changed after"):
        cpd.target_name = ("bar",)

    # Accessing coupling target without it having been set should raise errors
    cpd = CoupledParamDim(target_name=("foo",))

    with pytest.raises(ValueError, match="name cannot be changed!"):
        cpd.target_name = ("foo",)

    with pytest.raises(ValueError, match="The coupling target has not been"):
        cpd.target_pdim

    with pytest.raises(TypeError, match="Target of CoupledParamDim needs to"):
        cpd.target_pdim = "foo"

    cpd.target_pdim = pd

    # Setting it again also works
    cpd.target_pdim = pd

    # Test lengths are matching
    with pytest.raises(ValueError, match="The lengths of the value sequences"):
        cpd = CoupledParamDim(target_pdim=pd, values=[1, 2, 3, 4])

    # Assure values cannot be changed
    cpd = CoupledParamDim(target_pdim=pd, values=[2, 3, 4])
    with pytest.raises(AttributeError, match="Values already set; cannot be"):
        cpd._set_values([1, 2, 3], assert_unique=False)

    # Test it has no state set
    cpd = CoupledParamDim(target_pdim=pd, values=[2, 3, 4])
    assert cpd.state == 0
    assert cpd.current_value == 0  # that of the coupled ParamDim!

    # Check that it can access the target's mask
    assert cpd.mask is cpd.target_pdim.mask


def test_cpd_iteration():
    """Tests iteration of CoupledParamDim"""
    # ParamDim to couple to for testing
    pd = ParamDim(default=0, values=[1, 2, 3])

    # Simplest case: cpd follows pd
    for pval, cpval in zip(pd, CoupledParamDim(target_pdim=pd)):
        assert pval == cpval

    # With custom cpd values
    for pval, cpval in zip(
        pd, CoupledParamDim(target_pdim=pd, values=[2, 3, 4])
    ):
        assert pval + 1 == cpval


@pytest.mark.skip("Not working yet")
def test_cpd_standalone_iteration():
    """CoupledParamDim should be iterable standalone, too!"""
    cpd = CoupledParamDim(target_pdim=pd, values=[2, 3, 4])
    assert list(cpd) == [2, 3, 4]  # FIXME infinite loop


def test_coupled_mask():
    """Test the masking features' effects on CoupledParamDim"""
    pd = ParamDim(default=0, values=[1, 2, 3, 4], mask=(0, 1, 0, 1))

    # It should not be possible to mask a CPD
    with pytest.raises(TypeError, match="Received invalid keyword-argument"):
        CoupledParamDim(target_pdim=pd, mask=True)

    # Test that coupled iteration is masked accordingly
    vals = []
    for pval, cpval in zip(pd, CoupledParamDim(target_pdim=pd)):
        assert pval == cpval
        vals.append(pval)
    assert vals == [1, 3]


# YAML Dumping ----------------------------------------------------------------


def test_yaml_unsafe_dump_and_load(tmpdir, various_pdims):
    """Tests yaml dumping and loading with the unsafe methods"""
    yaml = yaml_unsafe

    d_out = various_pdims
    path = tmpdir.join("out.yml")

    # Dump it
    with open(path, "x") as out_file:
        yaml.dump(d_out, stream=out_file)

    # Read what was written
    with open(path) as in_file:
        print("Content of written file:\n")
        print("".join(in_file.readlines()))
        print("--- end of file ---")

    # Now load it
    with open(path) as in_file:
        d_in = yaml.load(in_file)

    print("Loaded object:", type(d_in))
    print(d_in)

    # Check that the contents are equivalent
    for k_out, v_out in d_out.items():
        assert k_out in d_in
        assert v_out == d_in[k_out]


def test_yaml_safe_dump_and_load(tmpdir, various_pdims):
    """Tests that YAML dumping and reloading works with both default dump and
    load methods as well as with the safe versions.
    """

    def dump_load_assert_equal(d_out: dict, *, path, dump_func, load_func):
        """Helper method for dumping, loading, and asserting equality"""
        # Dump it
        with open(path, "x") as out_file:
            dump_func(d_out, stream=out_file)

        # Read it in again
        with open(path) as in_file:
            d_in = load_func(in_file)

        # Check that the contents are equivalent
        for k_out, v_out in d_out.items():
            assert k_out in d_in
            assert v_out == d_in[k_out]

    # Use the dict of ParamDim objects for testing
    d_out = various_pdims

    # Test all possible combinations of dump and load methods
    methods = [
        ("def-def", yaml.dump, yaml.load),
        ("def-safe", yaml.dump, yaml_safe.load),
        ("safe-def", yaml_safe.dump, yaml.load),
        ("safe-safe", yaml_safe.dump, yaml_safe.load),
    ]

    for prefix, dump_func, load_func in methods:
        # Generate file name and some output to know what went wrong ...
        fname = prefix + ".yml"
        path = tmpdir.join(fname)

        print(f"Now testing combination:  {prefix}  ... ", end="")

        # Call the test function
        dump_load_assert_equal(
            d_out, path=path, dump_func=dump_func, load_func=load_func
        )

        print("Works!")


# Tests still to write --------------------------------------------------------


@pytest.mark.skip("Too early to write test.")
def test_save_and_restore():
    """Test whether saving of the current ParamDim state and restoring it works."""
    pass


@pytest.mark.skip("To do: ensure that it is well-behaving!")
def test_coupled_disabled():
    """Test whether saving of the current ParamDim state and restoring it works."""
    pass
