"""Tests for the ParamSpace class"""

from collections import OrderedDict

import pytest
import yaml
import numpy as np

from paramspace import ParamSpace, ParamDim, CoupledParamDim

# Setup methods ---------------------------------------------------------------

@pytest.fixture()
def basic_psp(request):
    """Used to setup a basic pspace object to be tested on."""
    d = dict(a=1, b=2, foo="bar", spam="eggs", mutable=[0, 0, 0],
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

@pytest.fixture()
def adv_psp(request):
    """Used to setup a more elaborate pspace object to be tested on. Includes name clashes, manually set names, order, ..."""
    d = dict(a=1, b=2, foo="bar", spam="eggs", mutable=[0, 0, 0],
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

@pytest.fixture()
def psp_with_coupled(request):
    """Used to setup a pspace object with coupled param dims"""
    d = dict(a=ParamDim(default=0, values=[1,2,3], order=0),
             c1=CoupledParamDim(target_name=('a',)),
             d=dict(aa=ParamDim(default=0, values=[1,2,3], order=-1),
                    cc1=CoupledParamDim(target_name=('d', 'aa')),
                    cc2=CoupledParamDim(target_name=('a',)),
                    cc3=CoupledParamDim(target_name='aa')),
             foo="bar", spam="eggs", mutable=[0, 0, 0],
             )
   
    return ParamSpace(d)

@pytest.fixture()
def psp_nested(request, basic_psp):
	"""Creates two ParamSpaces nested within another ParamSpace"""
	return ParamSpace(dict(foo="bar", basic=basic_psp,
	                       deeper=dict(basic=basic_psp)))


# Tests -----------------------------------------------------------------------

def test_init(basic_psp, adv_psp):
    """Test whether initialisation behaves as expected"""
    # These should work
    ParamSpace(dict(a=1))
    ParamSpace(OrderedDict(a=1))

    # These should also work, albeit not that practical
    ParamSpace(list(range(10)))

    # These should create a warning (not mutable)
    with pytest.warns(UserWarning, match="Got unusual type <class 'tuple'>"):
        ParamSpace(tuple(range(10)))

    with pytest.warns(UserWarning, match="Got unusual type <class 'set'>"):
        ParamSpace(set(range(10)))

    # These should warn and fail (not iterable)
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        with pytest.warns(UserWarning, match="Got unusual type"):
            ParamSpace(1)

    with pytest.raises(TypeError, match="'function' object is not iterable"):
        with pytest.warns(UserWarning, match="Got unusual type"):
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
                        d=ParamDim(default=0, logspace=[1,2,12,1]) # 12
                        ))
    assert p.volume == 1*5*20*12
    assert p.volume == p.full_volume

    # And of a paramspace without dimensions
    assert ParamSpace(dict(a=1)).volume == 0

def test_shape(basic_psp, adv_psp):
    """Asserts that the returned shape is correct"""
    assert basic_psp.shape == (3,3,3,3,3,3)
    assert adv_psp.shape == (3,3,3,3,3,3)

    p = ParamSpace(dict(a=ParamDim(default=0, values=[1]), # 1
                        b=ParamDim(default=0, range=[0,10,2]), # 5
                        c=ParamDim(default=0, linspace=[1,2,20]), # 20
                        d=ParamDim(default=0, logspace=[1,2,12,1]) # 12
                        ))
    assert p.shape == (1, 5, 20, 12)

    # Also test the number of dimensions
    assert basic_psp.num_dims == 6
    assert adv_psp.num_dims == 6
    assert p.num_dims == 4

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
    def check_counts(iters, counts):
        cntrs = {i:0 for i, _ in enumerate(counts)}

        for it_no, (it, count) in enumerate(zip(iters, counts)):
            for _ in it:
                cntrs[it_no] += 1
            assert cntrs[it_no] == count

    # For the explicit call
    check_counts((basic_psp.all_points(), adv_psp.all_points()),
                 (basic_psp.volume, adv_psp.volume))

    # For the call via __next__
    check_counts((basic_psp, adv_psp),
                 (basic_psp.volume, adv_psp.volume))

    # Also test all information tuples
    info = ("state_no", "state_vec", "progress")
    check_counts((basic_psp.all_points(with_info=info),
                  adv_psp.all_points(with_info=info)),
                 (basic_psp.volume, adv_psp.volume))

    # and whether invalid values lead to failure
    with pytest.raises(ValueError):
        info = ("state_no", "foo bar")
        check_counts((basic_psp.all_points(with_info=info),
                      adv_psp.all_points(with_info=info)),
                     (basic_psp.volume, adv_psp.volume))        

def test_state_no(basic_psp, adv_psp, psp_with_coupled):
    """Test that state number calculation is correct"""    
    def test_state_nos(psp):
        # Check that the state number is None outside an iteration
        assert psp.state_no is None
        
        # Get all points, then check them
        nos = [n for _, n in psp.all_points(with_info=("state_no",))]

        # Interval is correct
        assert nos[0] == 0
        assert len(nos) == psp.volume
        assert nos[-1] == max(nos) == psp.volume - 1
        
        # Increment is always 1
        d = np.diff(nos)
        assert max(d) == min(d) == 1

        # All ok
        return True

    # Call the test function on the given parameter spaces
    assert test_state_nos(basic_psp)
    assert test_state_nos(adv_psp)
    assert test_state_nos(psp_with_coupled)

def test_inverse_mapping(basic_psp, adv_psp):
    """Test whether the state mapping is correct."""
    basic_psp.inverse_mapping()
    adv_psp.inverse_mapping()

    # Test caching branch
    basic_psp.inverse_mapping()
    adv_psp.inverse_mapping()

def test_coupled(psp_with_coupled):
    """Test parameter spaces with CoupledParamDims in them"""
    psp = psp_with_coupled
    print("ParamSpace with CoupledParamDim:\n", psp)

    def assert_coupling(src: tuple, target: tuple):
        """Asserts that the CoupledParamDim at keyseq src is coupled to the target ParamDim at keyseq target."""
        assert psp.coupled_dims[src].target_pdim == psp.dims[target]

    # Assert correct coupling
    assert_coupling(('c1',), ('a',))
    assert_coupling(('d', 'cc1'), ('d', 'aa'))
    assert_coupling(('d', 'cc2'), ('a',))
    assert_coupling(('d', 'cc3'), ('d', 'aa'))

    # Check default is correct
    default = psp.default

    assert default['c1']       == default['a']
    assert default['d']['cc1'] == default['d']['aa']
    assert default['d']['cc2'] == default['a']
    assert default['d']['cc3'] == default['d']['aa']

    # Iterate over the paramspace and check correctness
    for pt in psp:
        print("Point: ", pt)
        
        assert pt['c1']       == pt['a']
        assert pt['d']['cc1'] == pt['d']['aa']
        assert pt['d']['cc2'] == pt['a']
        assert pt['d']['cc3'] == pt['d']['aa']
    
def test_nested(psp_nested, basic_psp):
	"""Tests whether nested ParamSpaces behave as desired"""
	default = psp_nested.default

	assert default['foo'] == "bar"
	assert default['basic'] == basic_psp
	assert default['deeper']['basic'] == basic_psp

def test_strings(basic_psp, adv_psp, psp_with_coupled):
    """Test whether the string generation works correctly."""
    for psp in [basic_psp, adv_psp, psp_with_coupled]:
        str(psp)
        repr(psp)
        psp.get_info_str()

def test_item_access(psp_with_coupled):
    """Assert that item access is working and safe"""
    psp = psp_with_coupled
    
    # get method - should be a deepcopy
    assert psp.get("foo") == "bar"
    assert psp.get("mutable") == [0, 0, 0]
    assert psp.get("mutable") is not psp._dict["mutable"]

    # pop method - should not work for parameter dimensions
    assert psp.pop("foo") == "bar"
    assert psp.pop("foo", "baz") == "baz"
    assert "foo" not in psp._dict

    assert psp.pop("spam") == "eggs"
    assert "spam" not in psp._dict
    
    with pytest.raises(KeyError, match="Cannot remove item with key"):
        psp.pop("a")

    with pytest.raises(KeyError, match="Cannot remove item with key"):
        psp.pop("c1")

@pytest.mark.skip("Feature is not implemented yet.")
def test_subspace():
    """Test whether the subspace retrieval is correct."""
    pass


# YAML Dumping ----------------------------------------------------------------

def test_yaml_unsafe_dump_and_load(basic_psp, adv_psp, psp_with_coupled, tmpdir):
    """Tests that YAML dumping and reloading works"""
    for i, psp_out in enumerate([basic_psp, adv_psp, psp_with_coupled]):
        psp_out = basic_psp
        path = tmpdir.join("out_{}.yml".format(i))
        
        # Dump it
        with open(path, "x") as out_file:
            yaml.dump(psp_out, stream=out_file)

        # Read it in again
        with open(path, "r") as in_file:
            psp_in = yaml.load(in_file)

        # Check that the contents are equivalent
        assert psp_in == psp_out

@pytest.mark.skip("Not yet working!")
def test_yaml_safe_dump_and_load(basic_psp, tmpdir):
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
    d_out = basic_psp

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
