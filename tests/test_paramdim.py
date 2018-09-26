"""Tests for the ParamDim classes"""

import warnings

import pytest
import numpy as np
import yaml

from paramspace import ParamDim, CoupledParamDim
from paramspace.paramdim import ParamDimBase, Masked

# Setup methods ---------------------------------------------------------------

@pytest.fixture(scope='module')
def various_pdims(request):
    """Used to setup various pspan objects to be tested on."""
    pds = {}

    pds['one']       = ParamDim(default=0, values=[1,2,3])
    pds['two']       = ParamDim(default=0, values=[1., 2, 'three', (1,0,0)])
    pds['range']     = ParamDim(default=0, range=[1, 4, 1])
    pds['linspace']  = ParamDim(default=0, linspace=[1, 3, 3, True])
    pds['logspace']  = ParamDim(default=0, logspace=[-1, 1, 11])
    pds['named']     = ParamDim(default=0, values=[1,2,3], name="named_span")
    pds['with_order']= ParamDim(default=0, values=[1,2,3], order=42)

    # coupled
    pds['coupled1']  = CoupledParamDim(target_pdim=pds['one'])
    pds['coupled2']  = CoupledParamDim(target_pdim=pds['two'],values=[1,2,3,4])
    pds['coupled3']  = CoupledParamDim(target_pdim=pds['range'], default=0)

    # base
    pds['base']      = ParamDimBase(default=0, values=[1,2,3])

    return pds


# Tests -----------------------------------------------------------------------

def test_init(various_pdims):
    """Test whether all initialisation methods work. Already got various_pdims from fixture, so these should work, so only explicitly tests the cases where initialisation should fail."""
    # No default given
    with pytest.raises(TypeError):
        ParamDim()

    # No values given
    with pytest.raises(ValueError):
        ParamDim(default=0)
    
    with pytest.raises(ValueError):
        ParamDim(default=0, values=[])

    with pytest.raises(ValueError):
        ParamDim(default=0, foo="bar")

    # Multiple values or kwargs given
    with pytest.warns(UserWarning):
        ParamDim(default=0, values=[1,2,3], linspace=[10,20,30])
    
    with pytest.warns(UserWarning):
        ParamDim(default=0, range=[1,2], linspace=[10,20,30])

    # Assert correct range, linspace, logspace creation
    vpd = various_pdims
    assert vpd['range'].values == tuple(range(1, 4, 1))
    assert all(vpd['linspace'].values == np.linspace(1, 3, 3, True))
    assert all(vpd['logspace'].values == np.logspace(-1, 1, 11))

def test_properties(various_pdims):
    """Test all properties and whether they are write-protected."""
    vpd = various_pdims

    # Whether the values are write-protected
    with pytest.raises(AttributeError, match="can't set attribute"):
        vpd['one'].values = 0
    
    with pytest.raises(AttributeError, match="can't set attribute"):
        vpd['two'].values = [1,2,3]

    with pytest.raises(AttributeError, match="can't set attribute"):
        vpd['base'].values = "baz"

    # Assert immutability of values
    with pytest.raises(TypeError, match="does not support item assignment"):
        vpd['one'].values[0] = "foo"
    
    with pytest.raises(TypeError, match="does not support item assignment"):
        vpd['two'].values[1] = "bar"

    # Whether the state is restricted to the value bounds
    with pytest.raises(ValueError, match="needs to be positive"):
        vpd['one'].state = -1
    
    with pytest.raises(ValueError, match="cannot exceed the highest index"):
        vpd['two'].state = 4

    with pytest.raises(TypeError, match="can only be of type int or None"):
        vpd['two'].state = "foo"

    # Misc
    for pd in vpd.values():
        if isinstance(pd, ParamDim):
            # Can be a target of a coupled ParamDim
            pd.target_of

def test_iteration(various_pdims):
    """Tests whether the iteration over the span's state works."""
    pd = ParamDim(default=0, values=[0,1,2])

    # First iteration
    assert pd.__next__() == 0
    assert pd.__next__() == 1
    assert pd.__next__() == 2
    with pytest.raises(StopIteration):
        pd.__next__()

    # Should be able to iterate again
    assert pd.__next__() == 0
    assert pd.__next__() == 1
    assert pd.__next__() == 2
    with pytest.raises(StopIteration):
        pd.__next__()

    # State should be None now
    assert pd.state is None

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
        ParamDim(default=0, logspace=[0, 10, 11])
    ]

    for pd in pds:
        types = [type(v) for v in pd.values]
        print("Types: " + str(types))
        assert all([t is float for t in types])

def test_mask():
    """Test that masking works"""
    # Test initialization, property getter and setter, and type
    pd = ParamDim(default=0, values=[0,1,2], mask=False)
    assert pd.mask is False
    # NOTE not trivial to test because the .mask getter _computes_ the value
    assert not any([isinstance(v, Masked) for v in pd.values])

    pd.mask = (True, False, True)
    assert pd.mask == (True, False, True)

    pd.mask = True
    assert pd.mask is True
    assert all([isinstance(v, Masked) for v in pd.values])

    # Check that length remains the same
    assert len(pd) == 3


# CoupledParamDim -------------------------------------------------------------

def test_coupled_init():
    """Test whether initialisation of CoupledParamDim works"""
    # These should work
    CoupledParamDim(target_name=("foo",))
    CoupledParamDim(target_name=("foo",), default=0)
    CoupledParamDim(target_name=("foo",), values=[1,2,3])
    CoupledParamDim(target_name="foo")

    # These should fail
    with pytest.raises(TypeError, match="missing 1 required"):
        # No default given
        CoupledParamDim(target_name=("foo",), use_coupled_default=False)

    with pytest.raises(ValueError, match="No argument `values` or other"):
        # No values given
        CoupledParamDim(target_name=("foo",), use_coupled_values=False)

    with pytest.raises(ValueError, match="The coupling target has not been"):
        # Not coupled yet
        CoupledParamDim(target_name=("foo",)).default

    with pytest.warns(UserWarning, match="Got both `target_pdim` and"):
        CoupledParamDim(target_pdim=ParamDim(default=0, values=[1,2,3]),
                        target_name=["foo", "bar"])

    # Set target
    pd = ParamDim(default=0, values=[1,2,3])
    cpd = CoupledParamDim(target_pdim=pd)
    assert len(pd) == len(cpd)
    assert pd.values == cpd.values
    assert pd.default == cpd.default
    assert cpd.target_name is None

    # Test if the name behaviour is correct
    with pytest.warns(UserWarning, match="A target ParamDim was already set;"):
        cpd.target_name = ("foo",)

    with pytest.raises(RuntimeError, match="Target name cannot be changed"):
        cpd.target_name = ("bar",)

    # Accessing coupling target without it having been set should raise errors
    cpd = CoupledParamDim(target_name=("foo",))

    with pytest.raises(ValueError, match="The coupling target has not been"):
        cpd.target_pdim

    with pytest.raises(TypeError, match="Target of CoupledParamDim needs to"):
        cpd.target_pdim = "foo"

    cpd.target_pdim = pd
    with pytest.raises(RuntimeError, match="Cannot change target of"):
        cpd.target_pdim = pd
    
    # Test lengths are matching
    with pytest.raises(ValueError, match="The lengths of the value sequences"):
        cpd = CoupledParamDim(target_pdim=pd, values=[1,2,3,4])

    # Assure values cannot be changed
    cpd = CoupledParamDim(target_pdim=pd, values=[2,3,4])
    with pytest.raises(AttributeError, match="Values already set; cannot be"):
        cpd._set_values([1,2,3])

    # Test disabled has no state set
    cpd = CoupledParamDim(target_pdim=pd, values=[2,3,4])
    assert cpd.state is None
    assert cpd.current_value is 0 # that of the coupled ParamDim!

def test_coupled_iteration():
    """Tests iteration of CoupledParamDim"""
    # ParamDim to couple to for testing
    pd = ParamDim(default=0, values=[1,2,3])

    # Simplest case: cpd follows pd
    for pval, cpval in zip(pd, CoupledParamDim(target_pdim=pd)):
        assert pval == cpval

    # With custom cpd values
    for pval, cpval in zip(pd, CoupledParamDim(target_pdim=pd,
                                               values=[2,3,4])):
        assert pval + 1 == cpval


# YAML Dumping ----------------------------------------------------------------

def test_yaml_unsafe_dump_and_load(various_pdims, tmpdir):
    """Tests yaml dumping and loading with the unsafe methods"""
    d_out = various_pdims
    path = tmpdir.join("out.yml")

    # Dump it    
    with open(path, "x") as out_file:
        yaml.dump(d_out, stream=out_file)

    # Read it in again
    with open(path, "r") as in_file:
        d_in = yaml.load(in_file)

    # Check that the contents are equivalent
    for k_out, v_out in d_out.items():
        assert k_out in d_in
        assert v_out == d_in[k_out]

@pytest.mark.skip("Not yet working!")
def test_yaml_safe_dump_and_load(various_pdims, tmpdir):
    """Tests that YAML dumping and reloading works with both default dump and
    load methods as well as with the safe versions.
    """
    def dump_load_assert_equal(d_out: dict, *, path, dump_func, load_func):
        """Helper method for dumping, loading, and asserting equality"""
        # Dump it
        with open(path, "x") as out_file:
            dump_func(d_out, stream=out_file)

        # Read it in again
        with open(path, "r") as in_file:
            d_in = load_func(in_file)

        # Check that the contents are equivalent
        for k_out, v_out in d_out.items():
            assert k_out in d_in
            assert v_out == d_in[k_out]

    # Use the dict of ParamDim objects for testing
    d_out = various_pdims

    # Test all possible combinations of dump and load methods
    methods = [(yaml.dump, yaml.load),
               (yaml.dump, yaml.safe_load),
               (yaml.safe_dump, yaml.load),
               (yaml.safe_dump, yaml.safe_load)]

    for dump_func, load_func in methods:
        # Generate file name and some output to know what went wrong ...
        fname = "{}--{}.yml".format(dump_func.__name__, load_func.__name__)
        path = tmpdir.join(fname)

        print("Now testing combination:  {} + {}  ... "
              "".format(dump_func.__name__, load_func.__name__), end="")

        # Call the test function
        dump_load_assert_equal(d_out, path=path,
                               dump_func=dump_func, load_func=load_func)

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
