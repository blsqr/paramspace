"""Tests for the ParamDim classes"""

import warnings
import pytest

import numpy as np

from paramspace import ParamDim, CoupledParamDim

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
    pds['disabled']  = ParamDim(default=0, values=[1,2,3], enabled=False)

    # coupled
    pds['coupled1']  = CoupledParamDim(target_pdim=pds['one'])
    pds['coupled2']  = CoupledParamDim(target_pdim=pds['one'],
                                       values=[1,2,3],
                                       use_coupled_values=False)

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
    with pytest.raises(AttributeError):
        vpd['one'].values = 0
    
    with pytest.raises(AttributeError):
        vpd['two'].values = [1,2,3]

    # Assert immutability of values
    with pytest.raises(TypeError):
        vpd['one'].values[0] = "foo"
    
    with pytest.raises(TypeError):
        vpd['two'].values[1] = "bar"

    # Whether the state is restricted to the value bounds
    with pytest.raises(ValueError):
        vpd['one'].state = -1
    
    with pytest.raises(ValueError):
        vpd['two'].state = 4

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

    # For disabled ParamDim
    assert various_pdims['disabled'].current_value == 0
    assert len(various_pdims['disabled']) == 1
    with pytest.raises(StopIteration):
        various_pdims['disabled'].__next__()

def test_str_methods(various_pdims):
    """Run through the string methods, just to call them..."""
    # Whether string representation works ok -- mainly for coverage here
    for pd in various_pdims.values():
        str(pd)
        repr(pd)

def test_coupled():
    """Test whether initialisation of CoupledParamDim works"""
    # These should work
    CoupledParamDim(target_name=("foo",))
    CoupledParamDim(target_name=("foo",), default=0)
    CoupledParamDim(target_name=("foo",), values=[1,2,3])

    # These should fail
    with pytest.raises(TypeError):
        # No default given
        CoupledParamDim(target_name=("foo",), use_coupled_default=False)

    with pytest.raises(ValueError):
        # No values given
        CoupledParamDim(target_name=("foo",), use_coupled_values=False)

    with pytest.raises(TypeError):
        # Wrong target name type
        CoupledParamDim(target_name="foo")

    # Set target
    pd = ParamDim(default=0, values=[1,2,3])
    cpd = CoupledParamDim(target_pdim=pd)
    assert len(pd) == len(cpd)
    assert pd.values == cpd.values
    assert pd.default == cpd.default
        

# Tests still to write --------------------------------------------------------

@pytest.mark.skip("Too early to write test.")
def test_save_and_restore():
    """Test whether saving of the current ParamDim state and restoring it works."""
    pass