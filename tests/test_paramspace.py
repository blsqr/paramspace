"""Tests for the ParamSpace class"""

import pytest
from paramspace import ParamSpace, ParamDim

# Setup methods ---------------------------------------------------------------

@pytest.fixture(scope='module')
def basic_psp(request):
    """Used to setup a basic pspace object to be tested on."""
    d = dict(a=1, b=2,
             p1=ParamDim(default=0, values=[1,2,3]),
             p2=ParamDim(default=0, values=[1,2,3]),
             d=dict(aa=1, bb=2,
                    pp1=ParamDim(default=0, values=[1,2,3]),
                    pp2=ParamDim(default=0, values=[1,2,3]),
                    dd=dict(aaa=1, bbb=2,
                            ppp1=ParamDim(default=0, values=[1,2,3]),
                            ppp2=ParamDim(default=0, values=[1,2,3])
                            )
                    )
             )
   
    return ParamSpace(d)

@pytest.fixture(scope='module')
def adv_psp(request):
    """Used to setup a more elaborate pspace object to be tested on. Includes name clashes, manually set names, order, ..."""
    d = dict(a=1, b=2,
             p1=ParamDim(default=0, values=[1,2,3], order=0),
             p2=ParamDim(default=0, values=[1,2,3], order=1),
             d=dict(a=1, b=2,
                    p1=ParamDim(default=0, values=[1,2,3], order=-1),
                    p2=ParamDim(default=0, values=[1,2,3], order=0),
                    d=dict(a=1, b=2,
                           p1=ParamDim(default=0, values=[1,2,3], name='ppp1'),
                           p2=ParamDim(default=0, values=[1,2,3], name='ppp2')
                           )
                    )
             )
   
    return ParamSpace(d)

# Tests -----------------------------------------------------------------------

def test_init(basic_psp, adv_psp):
    """Test whether initialisation from a dictionary works."""
    pspace = ParamSpace(dict(a=1))