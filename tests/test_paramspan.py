"""Tests for the ParamSpan classes"""

import pytest
from paramspace import ParamSpan, CoupledParamSpan

# Setup methods ---------------------------------------------------------------

@pytest.fixture(scope='module')
def various_pspans(request):
    """Used to setup various pspan objects to be tested on."""
    # Initialise a number of pspan objects
    psps = {}

    psps['span1']     = ParamSpan(span=[1,2,3], default=0)
    psps['span2']     = ParamSpan(span=[1., 2, 'three', (0, 1, 0, 0)])
    psps['range']     = ParamSpan(range=[1, 4, 1])
    psps['linspace']  = ParamSpan(linspace=[1, 3, 3, True])
    psps['logspace']  = ParamSpan(logspace=[-1, 1, 11])
    psps['named']     = ParamSpan(span=[1,2,3], name="named_span")
    psps['with_order']= ParamSpan(span=[1,2,3], order=42)
    psps['disabled']  = ParamSpan(span=[1,2,3], enabled=False)

    # Return the initialised object
    return psps


# Tests -----------------------------------------------------------------------

def test_init(various_pspans):
    """Test whether all initialisation methods work."""
    # Already got various_pspans from fixture, so these should work.

    # Now explicitly test the cases where initialisation should fail
    pass

def test_properties(various_pspans):
    """Test all properties and whether they are write-protected."""
    pass

def test_iteration(various_pspans):
    """Tests whether the iteration over the span's state works."""
    pass


# Tests still to write --------------------------------------------------------

@pytest.mark.skip("Too early to write test.")
def test_save_and_restore():
    """Test whether saving of the current ParamSpan state and restoring it works."""
    pass