"""Tests for the ParamDim classes"""

import pytest
from paramspace import ParamDim, CoupledParamDim

# Setup methods ---------------------------------------------------------------

@pytest.fixture(scope='module')
def various_pdims(request):
    """Used to setup various pspan objects to be tested on."""
    pds = {}

    pds['span1']     = ParamDim(default=0, values=[1,2,3])
    pds['span2']     = ParamDim(default=0, values=[1., 2, 'three', (1,0,0)])
    pds['range']     = ParamDim(default=0, range=[1, 4, 1])
    pds['linspace']  = ParamDim(default=0, linspace=[1, 3, 3, True])
    pds['logspace']  = ParamDim(default=0, logspace=[-1, 1, 11])
    pds['named']     = ParamDim(default=0, values=[1,2,3], name="named_span")
    pds['with_order']= ParamDim(default=0, values=[1,2,3], order=42)
    # pds['disabled']  = ParamDim(values=[1,2,3], enabled=False)

    return pds


# Tests -----------------------------------------------------------------------

def test_init(various_pdims):
    """Test whether all initialisation methods work."""
    # Already got various_pdims from fixture, so these should work.

    # Now explicitly test the cases where initialisation should fail
    pass

def test_properties(various_pdims):
    """Test all properties and whether they are write-protected."""
    pass

def test_iteration():
    """Tests whether the iteration over the span's state works."""
    pd = ParamDim(values=[0,1,2])

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

def test_disabled_without_default():
    """Tests whether an exception is raised when disabling is tried without a default value being present."""


# Tests still to write --------------------------------------------------------

@pytest.mark.skip("Too early to write test.")
def test_save_and_restore():
    """Test whether saving of the current ParamDim state and restoring it works."""
    pass