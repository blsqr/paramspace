"""Tests for the ParamDim classes"""

import pytest
from paramspace import ParamDim, CoupledParamDim

# Setup methods ---------------------------------------------------------------

@pytest.fixture(scope='module')
def various_pdims(request):
    """Used to setup various pspan objects to be tested on."""
    pds = {}

    pds['span1']     = ParamDim(values=[1,2,3], default=0)
    pds['span2']     = ParamDim(values=[1., 2, 'three', (0, 1, 0, 0)])
    pds['range']     = ParamDim(range=[1, 4, 1])
    pds['linspace']  = ParamDim(linspace=[1, 3, 3, True])
    pds['logspace']  = ParamDim(logspace=[-1, 1, 11])
    pds['named']     = ParamDim(values=[1,2,3], name="named_span")
    pds['with_order']= ParamDim(values=[1,2,3], order=42)
    pds['disabled']  = ParamDim(values=[1,2,3], enabled=False)

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

def test_iteration(various_pdims):
    """Tests whether the iteration over the span's state works."""
    pass


# Tests still to write --------------------------------------------------------

@pytest.mark.skip("Too early to write test.")
def test_save_and_restore():
    """Test whether saving of the current ParamDim state and restoring it works."""
    pass