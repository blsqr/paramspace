"""Tests for the ParamSpace class"""

import pytest
from paramspace import ParamSpace

# Setup methods ---------------------------------------------------------------

@pytest.fixture(scope='module')
def pspace(request):
    """Used to setup and tear down the pspace object to be tested on."""
    # Initialise pspace

    # Add a teardown function
    def teardown_pspace():
        print('tearing down pspace')
    request.addfinalizer(teardown_pspace)

    # Return the initialised 
    return ParamSpace(dict(a=1))


# Tests -----------------------------------------------------------------------

def test_init():
    """Test whether initialisation from a dictionary works."""
    pspace = ParamSpace(dict(a=1))