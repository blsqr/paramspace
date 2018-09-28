"""Tests for the ParamSpace class"""

from functools import reduce
from collections import OrderedDict

import pytest
import numpy as np
import numpy.ma

from paramspace import ParamSpace, ParamDim, CoupledParamDim, yaml

# Setup methods ---------------------------------------------------------------

@pytest.fixture()
def small_psp():
    """Used to setup a small pspace object to be tested on."""
    return ParamSpace(dict(p0=ParamDim(default=0, values=[1, 2]),
                           p1=ParamDim(default=0, values=[1, 2, 3]),
                           p2=ParamDim(default=0, values=[1, 2, 3, 4, 5])))

@pytest.fixture()
def basic_psp():
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
def adv_psp():
    """Used to setup a more elaborate pspace object to be tested on. Includes
    name clashes, manually set names, order, ..."""
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
def psp_with_coupled():
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
def psp_nested(basic_psp):
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

def test_strings(basic_psp, adv_psp, psp_with_coupled):
    """Test whether the string generation works correctly."""
    for psp in [basic_psp, adv_psp, psp_with_coupled]:
        str(psp)
        repr(psp)
        psp.get_info_str()
        psp._dim_names

def test_eq(adv_psp):
    """Test that __eq__ works"""
    psp = adv_psp
    
    assert (psp == "foo") is False      # Type does not match
    assert (psp == psp._dict) is False  # Not equivalent to the whole object
    assert (psp == psp) is True

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

def test_dim_access(basic_psp, adv_psp):
    """Test the _dim_by_name helper"""
    psp = basic_psp
    get_dim = psp._dim_by_name

    # Get existing parameter dimensions
    assert get_dim('p1') == psp._dict['p1']
    assert get_dim(('p1',)) == psp._dict['p1']
    assert get_dim(('d', 'pp1',)) == psp._dict['d']['pp1']
    assert get_dim(('d', 'dd', 'ppp1')) == psp._dict['d']['dd']['ppp1']

    # Non-existant dim should fail
    with pytest.raises(KeyError, match="A parameter dimension with name"):
        get_dim('foo')

    # More complicated setup, e.g. with ambiguous and custom names
    psp = adv_psp
    get_dim = psp._dim_by_name

    # p1 is now ambiguous
    with pytest.raises(ValueError, match="Could not unambiguously find"):
        get_dim('p1')

    # Need to add a "/" in front
    assert get_dim(('/', 'p1',)) == psp._dict['p1']

    # Also test the others, but with subsequences
    assert get_dim(('d', 'd', 'p1')) == psp._dict['d']['d']['p1']

    # Test access via name
    assert get_dim('ppp1') == psp._dict['d']['d']['p1']


def test_volume(small_psp, basic_psp, adv_psp):
    """Asserts that the volume calculation is correct"""
    assert small_psp.volume == 2 * 3 * 5
    assert basic_psp.volume == 3**6
    assert adv_psp.volume == 3**6

    p = ParamSpace(dict(a=ParamDim(default=0, values=[1]), # 1
                        b=ParamDim(default=0, range=[0,10,2]), # 5
                        c=ParamDim(default=0, linspace=[1,2,20]), # 20
                        d=ParamDim(default=0, logspace=[1,2,12,1]) # 12
                        ))
    assert p.volume == 1*5*20*12

    # And of a paramspace without dimensions
    empty_psp = ParamSpace(dict(a=1))
    assert empty_psp.volume == 0 == empty_psp.full_volume

def test_shape(small_psp, basic_psp, adv_psp):
    """Asserts that the returned shape is correct"""
    assert small_psp.shape == (2, 3, 5)
    assert basic_psp.shape == (3, 3, 3, 3, 3, 3)
    assert adv_psp.shape ==   (3, 3, 3, 3, 3, 3)

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

    # And the state shape, which is +1 larger in each entry
    assert small_psp.states_shape == (3, 4, 6)
    assert basic_psp.states_shape == (4, 4, 4, 4, 4, 4)
    assert adv_psp.states_shape ==   (4, 4, 4, 4, 4, 4)

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

def test_state_no(small_psp, basic_psp, adv_psp, psp_with_coupled):
    """Test that state number calculation is correct"""    
    def test_state_nos(psp):
        # Check that the state number is zero outside an iteration
        assert psp.state_no is 0
        
        # Get all points, then check them
        nos = [n for _, n in psp.all_points(with_info=("state_no",))]
        print("state numbers: ", nos)

        # Interval is correct
        assert nos[0] == min(nos)  # equivalent to sum of multipliers
        assert nos[-1] == max(nos) == reduce(lambda x, y: x*y,
                                             psp.states_shape) - 1
        assert len(nos) == psp.volume

        # All ok
        return True

    # Call the test function on the given parameter spaces
    assert test_state_nos(small_psp)
    assert test_state_nos(basic_psp)
    assert test_state_nos(adv_psp)
    assert test_state_nos(psp_with_coupled)
    # TODO add a masked one

def test_state_map(small_psp, basic_psp, adv_psp):
    """Test whether the state mapping is correct."""
    psps = [small_psp, basic_psp, adv_psp]

    for psp in psps:
        assert psp._smap is None
        psp.state_map
        assert psp._smap is not None

        # Call again, which will return the cached value
        psp.state_map

    # With specific pspace, do more explicit tests
    psp = small_psp
    imap = psp.state_map
    assert imap.shape == psp.states_shape
    assert imap[0,0,0] == 0
    assert np.max(imap) == reduce(lambda x, y: x*y, psp.states_shape) - 1

    # Test that the slices of the different dimensions are correct
    assert list(imap[:,0,0]) == [0, 1, 2]                # multiplier:  1
    assert list(imap[0,:,0]) == [0, 3, 6, 9]             # multiplier:  3
    assert list(imap[0,0,:]) == [0, 12, 24, 36, 48, 60]  # multiplier:  12

def test_mapping_funcs(small_psp):
    """Tests other mapping functions"""
    psp = small_psp

    # Test the get_state_vector method
    assert psp.get_state_vector(state_no=0) == (0, 0, 0)
    assert psp.get_state_vector(state_no=16) == (1, 1, 1)

    with pytest.raises(ValueError, match="Did not find state number -1"):
        psp.get_state_vector(state_no=-1)

    # Test the get_dim_values method
    psp.state_vector = (0, 0, 0)
    assert psp.get_dim_values() == OrderedDict([(('p0',), 0),
                                                (('p1',), 0),
                                                (('p2',), 0)])
    
    assert list(psp.get_dim_values(state_no=16).values()) == [1, 1, 1]
    assert list(psp.get_dim_values(state_vector=(1,2,3)).values()) == [1, 2, 3]

    # Should not work for both arguments given
    with pytest.raises(TypeError, match="Expected only one of the arguments"):
        psp.get_dim_values(state_no=123, state_vector=(1,2,3))

def test_basic_iteration(small_psp, basic_psp, adv_psp):
    """Tests whether the iteration goes through all points"""
    # Test on the __iter__ and __next__ level
    psp = small_psp
    it = psp.all_points()  # is a generator now
    assert it.__next__() == dict(p0=1, p1=1, p2=1)
    assert psp.state_vector == (1, 1, 1)
    assert psp.state_no == 16 # == 1 + 3 + 12  # 16

    assert it.__next__() == dict(p0=2, p1=1, p2=1)
    assert psp.state_vector == (2, 1, 1)
    assert psp.state_no == 16 + 1

    assert it.__next__() == dict(p0=1, p1=2, p2=1)
    assert psp.state_vector == (1, 2, 1)
    assert psp.state_no == 16 + 3

    assert it.__next__() == dict(p0=2, p1=2, p2=1)
    assert psp.state_vector == (2, 2, 1)
    assert psp.state_no == 16 + 3 + 1
    
    # ... and so on
    psp.reset()
    assert psp.state_vector == (0, 0, 0)
    assert psp.state_no == 0

    # Test some general properties relating to iteration and state
    # Test manually setting state vector
    psp.state_vector = (1, 1, 1)
    assert psp.current_point == dict(p0=1, p1=1, p2=1)

    with pytest.raises(ValueError, match="needs to be of same length as"):
        psp.state_vector = (0, 0)
    
    with pytest.raises(ValueError, match="Could not set the state of "):
        psp.state_vector = (-1, 42, 123.45)

    # A paramspace without volume should raise an error
    empty_psp = ParamSpace(dict(foo="bar"))
    with pytest.raises(ValueError, match="Cannot iterate over ParamSpace of"):
        list(iter(empty_psp))

    # Check the dry run
    psp.reset()
    snos = list([s for s in psp.all_points(with_info='state_no',omit_pt=True)])
    assert snos[:4] == [16, 17, 19, 20]
    
    # Check that the counts match using a helper function . . . . . . . . . . .
    def check_counts(iters, counts):
        cntrs = {i:0 for i, _ in enumerate(counts)}

        for it_no, (it, count) in enumerate(zip(iters, counts)):
            for _ in it:
                cntrs[it_no] += 1
            assert cntrs[it_no] == count

    # For the explicit call
    check_counts((basic_psp.all_points(), adv_psp.all_points()),
                 (basic_psp.volume, adv_psp.volume))

    # For the call via __iter__ and __next__
    check_counts((basic_psp, adv_psp),
                 (basic_psp.volume, adv_psp.volume))

    # Also test all information tuples and the dry run
    info = ("state_no", "state_vec")
    check_counts((small_psp.all_points(with_info=info),),
                 (small_psp.volume,))

    # ... and whether invalid values lead to failure
    with pytest.raises(ValueError):
        info = ("state_no", "foo bar")
        check_counts((small_psp.all_points(with_info=info),),
                     (small_psp.volume,))


# Masking ---------------------------------------------------------------------

def test_masking(small_psp):
    """Test whether the masking feature works"""
    psp = small_psp
    assert psp.shape == (2, 3, 5)
    assert psp.volume == 2 * 3 * 5

    # First try setting binary masks
    psp.set_mask('p0', True)
    assert psp.shape == (1, 3, 5)  # i.e.: 0th dimension only returns default
    assert psp.volume == 1 * 3 * 5

    # Mask completely
    psp.set_mask('p1', True)
    psp.set_mask('p2', True)
    assert psp.shape == (1, 1, 1)  # i.e.: all dimensions masked
    assert psp.volume == 1 * 1 * 1

    # Full volume and full shape should remain the old
    assert psp.full_volume == 2 * 3 * 5
    assert psp.full_shape == (2, 3, 5)

    # Remove the masks again, sometimes partly
    psp.set_mask('p0', False)
    assert psp.shape == (2, 1, 1)

    psp.set_mask('p1', (1, 1, 0))  # length 1
    assert psp.shape == (2, 1, 1)

    psp.set_mask('p2', (1, 0, 0, 1, 0))  # length 3
    assert psp.shape == (2, 1, 3)

    # Try inverting the mask
    psp.set_mask('p2', True, invert=True)
    assert psp.shape == (2, 1, 5)

    # Try setting multiple masks at once
    psp.set_masks(('p0', False), ('p1', False), ('p2', False))
    assert psp.shape == (2, 3, 5)
    
    psp.set_masks(dict(name=('p0',), mask=True),
                  dict(name='p2',    mask=slice(2), invert=True))
    assert psp.shape == (1, 3, 2)

    # Fully masked dimension also prompts changes in info string
    assert psp.get_info_str().find("fully masked -> using default:") > -1

def test_masked_iteration(small_psp):
    """Check iteration with a masked parameter space"""
    # First test: fully masked array
    psp = small_psp
    psp.set_mask('p0', True)
    psp.set_mask('p1', True)
    psp.set_mask('p2', True)

    # This should return only one entry: the default ParamSpace
    iter_res = {state_no:d
                for d, state_no in psp.all_points(with_info='state_no')}
    print("fully masked array: ", iter_res)
    
    assert len(iter_res) == 1
    assert 0 in iter_res
    assert iter_res[0] == dict(p0=0, p1=0, p2=0) == psp.default

    # Now the same with a non-trivial mask
    psp.set_mask('p0', (1, 0))  # length 1
    assert psp.shape == (1, 1, 1)
    iter_res = {state_no:d
                for d, state_no in psp.all_points(with_info='state_no')}
    print("p0 mask (True, False): ", iter_res)
    assert len(iter_res) == 1 == psp.volume
    assert 2 in iter_res
    assert iter_res[2] == dict(p0=2, p1=0, p2=0)

    # ... and an even more complex one
    psp.set_mask('p1', (0, 1, 0))
    assert psp.shape == (1, 2, 1)
    iter_res = {state_no:d
                for d, state_no in psp.all_points(with_info='state_no')}
    print("+ p1 mask (False, True, False): ", iter_res)
    assert len(iter_res) == 1 * 2 == psp.volume
    assert (3 + 2) in iter_res
    assert iter_res[5] == dict(p0=2, p1=1, p2=0)
    assert (9 + 2) in iter_res
    assert iter_res[11] == dict(p0=2, p1=3, p2=0)

def test_masked_mapping_funcs(small_psp):
    """Test the mapping methods for masked parameter spaces"""
    psp = small_psp

    # Compare the default array to a reference masked array
    ref_ma = np.ma.MaskedArray(data=small_psp.state_map)
    ref_ma[:,0,0] = np.ma.masked
    ref_ma[0,:,0] = np.ma.masked
    ref_ma[0,0,:] = np.ma.masked
    mmap = psp.get_masked_smap()
    print("masked state map:\n", mmap)
    assert np.array_equal(mmap, ref_ma)

# Complicated content ---------------------------------------------------------

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

    # Invalid coupling targets should raise an error
    with pytest.raises(ValueError, match="Could not resolve the coupling"):
        ParamSpace(dict(a=ParamDim(default=0, range=[10]),
                        b=CoupledParamDim(target_name="foo")))
    
def test_nested(psp_nested, basic_psp):
	"""Tests whether nested ParamSpaces behave as desired"""
	default = psp_nested.default

	assert default['foo'] == "bar"
	assert default['basic'] == basic_psp
	assert default['deeper']['basic'] == basic_psp


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
