"""Tests for the ParamSpace class"""

from collections import OrderedDict

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
    """Test whether initialisation behaves as expected"""
    # These should work
    ParamSpace(dict(a=1))
    ParamSpace(OrderedDict(a=1))

    # These should also work, albeit not that practical
    ParamSpace(list(range(10)))

    # These should create a warning (not mutable)
    with pytest.warns(UserWarning):
        ParamSpace(tuple(range(10)))
    with pytest.warns(UserWarning):
        ParamSpace(set(range(10)))

    # These should warn and fail (not iterable)
    with pytest.raises(TypeError):
        with pytest.warns(UserWarning):
            ParamSpace(1)
    with pytest.raises(TypeError):
        with pytest.warns(UserWarning):
            ParamSpace(lambda x: None)

def test_default(basic_psp, adv_psp):
    """Tests whether the default values can be retrieved."""
    d1 = basic_psp.default
    assert d1['p1'] == 0
    assert d1['p2'] == 0
    assert d1['d']['pp1'] == 0
    assert d1['d']['pp2'] == 0
    assert d1['d']['dd']['ppp1'] == 0
    assert d1['d']['dd']['ppp2'] == 0

    d2 = adv_psp.default
    assert d2['p1'] == 0
    assert d2['p2'] == 0
    assert d2['d']['p1'] == 0
    assert d2['d']['p2'] == 0
    assert d2['d']['d']['p1'] == 0
    assert d2['d']['d']['p2'] == 0

def test_volume(basic_psp, adv_psp):
    """Asserts that the volume calculation is correct"""
    assert basic_psp.volume == 3**6
    assert basic_psp.volume == basic_psp.full_volume
    assert adv_psp.volume == 3**6
    assert adv_psp.volume == adv_psp.full_volume

    p = ParamSpace(dict(a=ParamDim(default=0, values=[1]), # 1
                        b=ParamDim(default=0, range=[0,10,2]), # 5
                        c=ParamDim(default=0, linspace=[1,2,20]), # 20
                        d=ParamDim(default=0, logspace=[1,2,12,1]), # 12
                        e=ParamDim(default=0, values=[1,2], enabled=False))) 
    assert p.volume == 1*5*20*12
    assert p.volume == p.full_volume

def test_shape(basic_psp, adv_psp):
    """Asserts that the returned shape is correct"""
    assert basic_psp.shape == (3,3,3,3,3,3)
    assert adv_psp.shape == (3,3,3,3,3,3)

    p = ParamSpace(dict(a=ParamDim(default=0, values=[1]), # 1
                        b=ParamDim(default=0, range=[0,10,2]), # 5
                        c=ParamDim(default=0, linspace=[1,2,20]), # 20
                        d=ParamDim(default=0, logspace=[1,2,12,1]), # 12
                        e=ParamDim(default=0, values=[1,2], enabled=False))) 
    assert p.shape == (1,5,20,12,1) # disabled dimensions still count here

    # Also test the number of dimensions
    assert basic_psp.num_dims == 6
    assert adv_psp.num_dims == 6
    assert p.num_dims == 5

def test_dim_order(basic_psp, adv_psp):
    """Tests whether the dimension order is correct."""
    basic_psp_names = (# alphabetically sorted
                       ('d', 'dd', 'ppp1',),
                       ('d', 'dd', 'ppp2',),
                       ('d', 'pp1',),
                       ('d', 'pp2',),
                       ('p1',),
                       ('p2',))
    for name_is, name_should in zip(basic_psp.dims, basic_psp_names):
        assert name_is == name_should

    adv_psp_names = (# sorted by order parameter
                    ('d', 'p1'),
                    ('d', 'p2'),
                    ('p1',),
                    ('p2',),
                    ('d', 'd', 'p1'),
                    ('d', 'd', 'p2'))
    for name_is, name_should in zip(adv_psp.dims, adv_psp_names):
        assert name_is == name_should

def test_iteration(basic_psp, adv_psp):
    """Tests whether the iteration goes through all points"""
    cntr = dict(basic=0, adv=0)

    for _, _ in zip(basic_psp.all_points(), adv_psp.all_points()):
        cntr['basic'] += 1
        cntr['adv'] += 1

    assert cntr['basic'] == basic_psp.volume
    assert cntr['adv'] == adv_psp.volume

@pytest.mark.skip("getting state number not implemented yet")
def test_inverse_mapping(basic_psp, adv_psp):
    """Test whether the state mapping is correct."""
    basic_psp.inverse_mapping()
    adv_psp.inverse_mapping()

@pytest.mark.skip("Not implemented yet.")
def test_subspace():
    """Test whether the subspace retrieval is correct."""
    pass