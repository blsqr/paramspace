"""Tests for the ParamSpace class"""
from collections import OrderedDict
from functools import reduce

import numpy as np
import numpy.ma
import pytest

from paramspace import CoupledParamDim, ParamDim, ParamSpace
from paramspace.paramdim import Masked
from paramspace.yaml import *

# Fixtures --------------------------------------------------------------------


@pytest.fixture()
def small_psp():
    """Used to setup a small pspace object to be tested on."""
    return ParamSpace(
        dict(
            p0=ParamDim(default=0, values=[1, 2]),
            p1=ParamDim(default=0, values=[1, 2, 3]),
            p2=ParamDim(default=0, values=[1, 2, 3, 4, 5]),
        )
    )


@pytest.fixture()
def float_valued_psp():
    """A small parameter space that has float values for parameter values"""
    return ParamSpace(
        dict(
            lin1=ParamDim(default=0.0, linspace=[-1.0, 1.0, 11]),
            lin2=ParamDim(default=0.0, linspace=[-2.0, 2.0, 11]),
            log1=ParamDim(default=0.0, logspace=[-10.0, 10.0, 6]),
        )
    )


@pytest.fixture()
def str_valued_psp():
    """A small parameter space that has string values for parameter values"""
    return ParamSpace(
        dict(
            p0=ParamDim(default="foo", values=["foo", "bar", "baz"]),
            p1=ParamDim(default="0", values=["0", "1", "2", "3", "4"]),
            lin=ParamDim(default=0.0, linspace=[0.0, 1.0, 6]),
        )
    )


@pytest.fixture()
def basic_psp():
    """Used to setup a basic pspace object to be tested on."""
    d = dict(
        a=1,
        b=2,
        foo="bar",
        spam="eggs",
        mutable=[0, 0, 0],
        p1=ParamDim(default=0, values=[1, 2, 3]),
        p2=ParamDim(default=0, values=[1, 2, 3]),
        d=dict(
            aa=1,
            bb=2,
            pp1=ParamDim(default=0, values=[1, 2, 3]),
            pp2=ParamDim(default=0, values=[1, 2, 3]),
            dd=dict(
                aaa=1,
                bbb=2,
                ppp1=ParamDim(default=0, values=[1, 2, 3]),
                ppp2=ParamDim(default=0, values=[1, 2, 3]),
            ),
        ),
    )

    return ParamSpace(d)


@pytest.fixture()
def adv_psp():
    """Used to setup a more elaborate pspace object to be tested on. Includes
    name clashes, manually set names, order, ..."""
    d = dict(
        a=1,
        b=2,
        foo="bar",
        spam="eggs",
        mutable=[0, 0, 0],
        p1=ParamDim(default=0, values=[1, 2, 3], order=0),
        p2=ParamDim(default=0, values=[1, 2, 3], order=1),
        d=dict(
            a=1,
            b=2,
            p1=ParamDim(default=0, values=[1, 2, 3], order=-1),
            p2=ParamDim(default=0, values=[1, 2, 3], order=0),
            d=dict(
                a=1,
                b=2,
                p1=ParamDim(default=0, values=[1, 2, 3], name="ppp1"),
                p2=ParamDim(default=0, values=[1, 2, 3], name="ppp2"),
            ),
        ),
    )

    return ParamSpace(d)


@pytest.fixture()
def seq_psp():
    """A parameter space with dimensions within sequences"""
    d = dict(
        a=1,
        b=2,
        foo="bar",
        spam="eggs",
        mutable=[0, 0, 0],
        s=[
            ParamDim(default=0, values=[1, 2, 3]),
            [ParamDim(default=1, values=[1, 2, 3], order=1), 2, 3],
        ],
        d=dict(
            a=1,
            b=2,
            s=[
                ParamDim(default=0, values=[1, 2, 3], order=-1),
                ParamDim(default=1, values=[1, 2, 3], order=0, name="ds1"),
                2,
                3,
            ],
            d=dict(
                a=1,
                b=2,
                p=ParamDim(default=0, values=[1, 2, 3]),
                s=[
                    0,
                    ParamDim(default=1, values=[1, 2, 3]),
                    ParamDim(default=2, values=[1, 2, 3], name="dds2"),
                    3,
                ],
            ),
        ),
    )

    return ParamSpace(d)


@pytest.fixture()
def psp_with_coupled():
    """Used to setup a pspace object with coupled param dims"""
    d = dict(
        a=ParamDim(default=0, values=[1, 2, 3], order=0),
        c1=CoupledParamDim(target_name=("a",)),
        d=dict(
            aa=ParamDim(default=0, values=[1, 2, 3], order=-1),
            cc1=CoupledParamDim(target_name=("d", "aa")),
            cc2=CoupledParamDim(target_name=("a",)),
            cc3=CoupledParamDim(target_name="aa"),
        ),
        foo="bar",
        spam="eggs",
        mutable=[0, 0, 0],
    )

    return ParamSpace(d)


@pytest.fixture()
def psp_nested(basic_psp):
    """Creates two ParamSpaces nested within another ParamSpace"""
    return ParamSpace(
        dict(foo="bar", basic=basic_psp, deeper=dict(basic=basic_psp))
    )


# Tests -----------------------------------------------------------------------


def test_init(basic_psp, adv_psp, seq_psp):
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


def test_default(small_psp, adv_psp, seq_psp):
    """Tests whether the default values can be retrieved."""
    d1 = small_psp.default

    # Test for correct values
    assert d1["p0"] == 0
    assert d1["p1"] == 0
    assert d1["p2"] == 0

    # ...and type
    assert not isinstance(d1["p0"], Masked)
    assert not isinstance(d1["p1"], Masked)
    assert not isinstance(d1["p2"], Masked)

    # Same for the deeper dict
    d2 = adv_psp.default
    assert d2["p1"] == 0
    assert d2["p2"] == 0
    assert d2["d"]["p1"] == 0
    assert d2["d"]["p2"] == 0
    assert d2["d"]["d"]["p1"] == 0
    assert d2["d"]["d"]["p2"] == 0

    # ...and type
    assert not isinstance(d2["p1"], Masked)
    assert not isinstance(d2["p2"], Masked)
    assert not isinstance(d2["d"]["p1"], Masked)
    assert not isinstance(d2["d"]["p2"], Masked)
    assert not isinstance(d2["d"]["d"]["p1"], Masked)
    assert not isinstance(d2["d"]["d"]["p2"], Masked)

    # Same for the dict with parameter dimensions inside sequences
    d3 = seq_psp.default
    assert d3["s"] == [0, [1, 2, 3]]
    assert d3["d"]["s"] == [0, 1, 2, 3]
    assert d3["d"]["d"]["p"] == 0
    assert d3["d"]["d"]["s"] == [0, 1, 2, 3]

    assert not isinstance(d3["s"][0], Masked)
    assert not isinstance(d3["s"][1], Masked)
    assert not isinstance(d3["d"]["s"][0], Masked)
    assert not isinstance(d3["d"]["s"][1], Masked)
    assert not isinstance(d3["d"]["d"]["p"], Masked)
    assert not isinstance(d3["d"]["d"]["s"][1], Masked)
    assert not isinstance(d3["d"]["d"]["s"][2], Masked)


def test_strings(basic_psp, adv_psp, psp_with_coupled, seq_psp):
    """Test whether the string generation works correctly."""
    for psp in [basic_psp, adv_psp, psp_with_coupled, seq_psp]:
        str(psp)
        repr(psp)

        psp.get_info_str()

        psp._parse_dims(mode="names")
        psp._parse_dims(mode="locs")
        psp._parse_dims(mode="both")
        with pytest.raises(ValueError, match="Invalid mode: foo"):
            psp._parse_dims(mode="foo")


def test_eq(adv_psp):
    """Test that __eq__ works"""
    psp = adv_psp

    assert (psp == "foo") is False  # Type does not match
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


def test_dim_name_creation():
    """Asserts that creation of unique dimension names works"""
    create_names = ParamSpace._unique_dim_names

    def check_names(*name_check_pdim):
        """Create the necessary input data for the create_names function, then
        perform it and assert equality to the expected values ...
        """
        kv_pairs = [
            (path, ParamDim(default=0, values=[1, 2], name=pd_name))
            if pdim is not None
            else (path, ParamDim(default=0, values=[1, 2]))
            for path, _, pd_name in name_check_pdim
        ]
        expected_names = [name_out for _, name_out, _ in name_check_pdim]
        actual_names = [k for k, _ in create_names(kv_pairs)]
        assert expected_names == actual_names

    # Start with the tests
    # Arguments for check_names: (input path, expected name, custom name)
    # Some basics
    check_names(
        (("foo", "bar"), "bar", None),
        (("foo", "baz"), "foo.baz", None),
        (("abc", "def"), "spam", "spam"),
        (("bar", "baz"), "bar.baz", None),
    )

    # Repeating pattern -> resolved up unto root
    check_names(
        (("p0",), ".p0", None),
        (("d", "p0"), ".d.p0", None),
        (("d", "d", "p0"), "d.d.p0", None),
        (("d", "d", "d", "p0"), "p1", "p1"),
    )

    # Custom names
    check_names(
        (("p0",), "p0", "p0"),
        (("d", "p0"), "p1", "p1"),
        (("d", "d", "p0"), "p2", "p2"),
        (("d", "d", "d", "p0"), "p3", "p3"),
    )

    # Single non-custom name at root level
    check_names(
        (("p0",), "p0", "p0"),
        (("d", "p0"), "p1", "p1"),
        (("d", "d", "p0"), "p2", "p2"),
        (("v0",), "v0", None),
    )

    # Custom names have priority over existing paths
    check_names(
        (("p0",), ".p0", None),
        (("d", "p0"), "p0", "p0"),
        (("d", "d", "p0"), ".d.d.p0", None),
        (("d", "d", "d", "p0"), "d.d.d.p0", None),
    )

    # Can have integer elements in there
    check_names(
        (("foo", "bar", 0), "bar.0", None),
        (("foo", "baz", 1), "baz.1", None),
        (("abc", "def", 23, "foo"), "def.23.foo", None),
        ((12, "bar", "baz", 0, "foo"), ".12.bar.baz.0.foo", None),
    )

    check_names(
        (("d", "d", "s", 1), "d.d.s.1", None),
        (("d", "d", "s", 2), "d.s.2", None),
        (("d", "d", "s", 3), "d.s.3", None),
        (("d", "s", 0), ".d.s.0", None),
        (("d", "s", 1), ".d.s.1", None),
        (("s", 0), ".s.0", None),
        (("s", 1), ".s.1", None),
        (("s", 3, 0), "s.3.0", None),
        (("s", 2, 0), "s.2.0", None),
        (("t", 1, 0), "t.1.0", None),
        (("t", 1, 1), "t.1.1", None),
    )

    # Can also be other numerical values (although not a super idea, typically)
    check_names(
        (("foo", "bar", -2), "bar.-2", None),
        (("foo", "baz", 1.5), "baz.1.5", None),
        (("abc", "def", 23.45, "foo"), "def.23.45.foo", None),
        ((12.34, "bar", "baz", -0.1, "foo"), ".12.34.bar.baz.-0.1.foo", None),
    )

    # Paths cannot include '.', require a custom name
    with pytest.raises(ValueError, match="Please select a custom name for"):
        check_names(
            (("foo.bar", "baz"), "baz", None),
            (("foo", "bar.baz"), "bar.baz", None),
        )

    # Path separator in custom names is not allowed
    with pytest.raises(ValueError, match="cannot contain the hierarchy-sep"):
        check_names(
            (("p0",), ".p0", None),
            (("d", "p0"), ".d.p0", None),
            (("d", "d", "p0"), "d.d.p0", None),
            (("d", "d", "d", "p0"), "d.d.d.p0", "d.p0"),
        )

    # Custom names need be strings
    with pytest.raises(TypeError, match="need to be strings"):
        check_names(
            (("p0",), "p0", 1.23),
        )

    # Colliding custom names -> ValueError
    with pytest.raises(ValueError, match="There were duplicates among"):
        check_names(
            (("p0",), ".p0", None),
            (("d", "p0"), ".d.p0", None),
            (("d", "d", "p0"), "d.d.p0", "p1"),
            (("d", "d", "d", "p0"), "d.d.d.p0", "p1"),
        )

    # Pathological case where no unique names can be found; should abort
    with pytest.raises(ValueError, match="Could not automatically find"):
        check_names((("d", "p0"), ".d.p0", None), (("d", "p0"), ".d.p0", None))


def test_dim_access(basic_psp, adv_psp, seq_psp):
    """Test the _get_dim helper"""
    # -- Basics
    psp = basic_psp
    get_dim = psp._get_dim

    # Get existing parameter dimensions by name
    assert get_dim("p1") == psp._dict["p1"]
    assert get_dim("pp1") == psp._dict["d"]["pp1"]
    assert get_dim("ppp1") == psp._dict["d"]["dd"]["ppp1"]

    # And by location
    assert get_dim(("p1",)) == psp._dict["p1"]
    assert (
        get_dim(
            (
                "d",
                "pp1",
            )
        )
        == psp._dict["d"]["pp1"]
    )
    assert get_dim(("d", "dd", "ppp1")) == psp._dict["d"]["dd"]["ppp1"]

    # Non-existant name or location should fail
    with pytest.raises(ValueError, match="A parameter dimension with name"):
        get_dim("foo")

    with pytest.raises(ValueError, match="parameter dimension matching locat"):
        get_dim(("foo",))

    # -- More complicated setup, e.g. with ambiguous and custom names
    psp = adv_psp
    get_dim = psp._get_dim

    # p1 is now ambiguous
    with pytest.raises(ValueError, match="parameter dimension with name 'p1'"):
        get_dim("p1")

    with pytest.raises(ValueError, match="Could not unambiguously find a"):
        get_dim(("p1",))

    # Thus, the unique name is different
    assert get_dim(".p1") == psp._dict["p1"]
    assert (
        get_dim(
            (
                "",
                "p1",
            )
        )
        == psp._dict["p1"]
    )

    # Also test the next level
    assert get_dim("d.p1") == psp._dict["d"]["p1"]
    assert get_dim(("", "d", "p1")) == psp._dict["d"]["p1"]

    # On the third level, there are custom names
    assert get_dim("ppp1") == psp._dict["d"]["d"]["p1"]
    assert get_dim("ppp2") == psp._dict["d"]["d"]["p2"]

    # Access via path should still be possible ...
    assert get_dim(("d", "d", "p1")) == psp._dict["d"]["d"]["p1"]
    assert get_dim(("d", "d", "p2")) == psp._dict["d"]["d"]["p2"]

    # -- And now with item access
    psp = seq_psp
    get_dim = psp._get_dim

    assert get_dim(".s.0") == psp._dict["s"][0]
    assert get_dim(("", "s", 0)) == psp._dict["s"][0]

    assert get_dim("s.1.0") == psp._dict["s"][1][0]
    assert get_dim(("", "s", 1, 0)) == psp._dict["s"][1][0]

    assert get_dim("ds1") == psp._dict["d"]["s"][1]
    assert get_dim(("", "d", "s", 1)) == psp._dict["d"]["s"][1]

    assert get_dim("p") == psp._dict["d"]["d"]["p"]
    assert get_dim(("", "d", "d", "p")) == psp._dict["d"]["d"]["p"]

    assert get_dim("d.s.1") == psp._dict["d"]["d"]["s"][1]
    assert get_dim(("", "d", "d", "s", 1)) == psp._dict["d"]["d"]["s"][1]


def test_volume(small_psp, basic_psp, adv_psp, seq_psp):
    """Asserts that the volume calculation is correct"""
    assert small_psp.volume == 2 * 3 * 5
    assert basic_psp.volume == 3 ** 6
    assert adv_psp.volume == 3 ** 6
    assert seq_psp.volume == 3 ** 7

    p = ParamSpace(
        dict(
            a=ParamDim(default=0, values=[1]),  # 1
            b=ParamDim(default=0, range=[0, 10, 2]),  # 5
            c=ParamDim(default=0, linspace=[1, 2, 20]),  # 20
            d=ParamDim(default=0, logspace=[1, 2, 12, 1]),  # 12
        )
    )
    assert p.volume == 1 * 5 * 20 * 12

    # And of a paramspace without dimensions
    empty_psp = ParamSpace(dict(a=1))
    assert empty_psp.volume == 0 == empty_psp.full_volume


def test_shape(small_psp, basic_psp, adv_psp, seq_psp):
    """Asserts that the returned shape is correct"""
    assert small_psp.shape == (2, 3, 5)
    assert basic_psp.shape == (3, 3, 3, 3, 3, 3)
    assert adv_psp.shape == (3, 3, 3, 3, 3, 3)
    assert seq_psp.shape == (3, 3, 3, 3, 3, 3, 3)

    p = ParamSpace(
        dict(
            a=ParamDim(default=0, values=[1]),  # 1
            b=ParamDim(default=0, range=[0, 10, 2]),  # 5
            c=ParamDim(default=0, linspace=[1, 2, 20]),  # 20
            d=ParamDim(default=0, logspace=[1, 2, 12, 1]),  # 12
        )
    )
    assert p.shape == (1, 5, 20, 12)

    # Also test the number of dimensions
    assert basic_psp.num_dims == 6
    assert adv_psp.num_dims == 6
    assert p.num_dims == 4

    # And the state shape, which is +1 larger in each entry
    assert small_psp.states_shape == (3, 4, 6)
    assert basic_psp.states_shape == (4, 4, 4, 4, 4, 4)
    assert adv_psp.states_shape == (4, 4, 4, 4, 4, 4)

    # And that the maximum state number is correct
    assert small_psp.max_state_no == 71
    assert basic_psp.max_state_no == 4095
    assert adv_psp.max_state_no == 4095


def test_coords(small_psp):
    """Tests coordinate-related properts"""
    psp = small_psp

    # Retrieve and check coordinate properties against reference arrays.
    # NOTE That Masked.__eq__ compares its _value_ to other, thus it is not
    #      necessary to include Masked objects here; is checked below
    assert psp.coords == dict(
        p0=[0, 1, 2], p1=[0, 1, 2, 3], p2=[0, 1, 2, 3, 4, 5]
    )
    assert psp.pure_coords == dict(
        p0=[0, 1, 2], p1=[0, 1, 2, 3], p2=[0, 1, 2, 3, 4, 5]
    )
    assert psp.current_coords == dict(p0=0, p1=0, p2=0)

    # Check the Masked state. Outside of iteration, index 0 should be Masked
    assert all(isinstance(v[0], Masked) for v in psp.coords.values())
    assert all(isinstance(v, Masked) for v in psp.current_coords.values())


def test_dim_order(basic_psp, adv_psp, seq_psp):
    """Tests whether the dimension order is correct."""
    basic_psp_locs = (  # alphabetically sorted
        (
            "d",
            "dd",
            "ppp1",
        ),
        (
            "d",
            "dd",
            "ppp2",
        ),
        (
            "d",
            "pp1",
        ),
        (
            "d",
            "pp2",
        ),
        ("p1",),
        ("p2",),
    )
    for name_is, name_should in zip(basic_psp.dims_by_loc, basic_psp_locs):
        assert name_is == name_should

    adv_psp_locs = (  # sorted by order parameter
        ("d", "p1"),
        ("d", "p2"),
        ("p1",),
        ("p2",),
        ("d", "d", "p1"),
        ("d", "d", "p2"),
    )
    for name_is, name_should in zip(adv_psp.dims_by_loc, adv_psp_locs):
        assert name_is == name_should

    seq_psp_locs = (  # sorting includes indices
        ("d", "s", 0),
        ("d", "s", 1),
        ("s", 1, 0),
        ("d", "d", "p"),
        ("d", "d", "s", 1),
        ("d", "d", "s", 2),
        ("s", 0),
    )
    for actual, expected in zip(seq_psp.dims_by_loc, seq_psp_locs):
        assert actual == expected


def test_state_no(small_psp, basic_psp, adv_psp, seq_psp, psp_with_coupled):
    """Test that state number calculation is correct"""

    def test_state_nos(psp):
        # Check that the state number is zero outside an iteration
        assert psp.state_no == 0

        # Get all points, then check them
        nos = [n for _, n in psp.iterator(with_info=("state_no",))]
        print("state numbers: ", nos)

        # Interval is correct
        assert nos[0] == min(nos)  # equivalent to sum of multipliers
        assert (
            nos[-1]
            == max(nos)
            == reduce(lambda x, y: x * y, psp.states_shape) - 1
        )
        assert len(nos) == psp.volume

        # Try setting the state number
        psp.state_no = 1
        assert psp.state_no == 1

        # Reset
        psp.reset()
        assert psp.state_no == 0

    # Call the test function on the given parameter spaces
    test_state_nos(small_psp)
    test_state_nos(basic_psp)
    test_state_nos(adv_psp)
    test_state_nos(seq_psp)
    test_state_nos(psp_with_coupled)
    # TODO add a masked one


def test_state_map(small_psp, basic_psp, adv_psp, seq_psp):
    """Test whether the state mapping is correct."""
    psps = [small_psp, basic_psp, adv_psp, seq_psp]

    for psp in psps:
        assert psp._smap is None
        psp.state_map
        assert psp._smap is not None

        # Call again, which will return the cached value
        psp.state_map

    # With specific pspace, do more explicit tests
    psp = small_psp
    imap = psp.state_map
    print("Got state map:\n", imap)

    assert imap.dtype == int
    assert imap.shape == psp.states_shape
    assert imap[0, 0, 0] == 0
    assert np.max(imap) == reduce(lambda x, y: x * y, psp.states_shape) - 1

    # Test that the slices of the different dimensions are correct
    assert list(imap[0, 0, :]) == [0, 1, 2, 3, 4, 5]  # multiplier:  1
    assert list(imap[0, :, 0]) == [0, 6, 12, 18]  # multiplier:  6
    assert list(imap[:, 0, 0]) == [0, 24, 48]  # multiplier:  24

    # Test the xarray features
    # Make sure all coordinate values are unmasked
    print("coords:", [v for v in imap.coords["p0"].to_series()])
    assert all(
        [not isinstance(v, Masked) for v in imap.coords["p0"].to_series()]
    )
    assert all(
        [not isinstance(v, Masked) for v in imap.coords["p1"].to_series()]
    )
    assert all(
        [not isinstance(v, Masked) for v in imap.coords["p2"].to_series()]
    )

    # Check the coordinate dtype is not object
    print("coords dtypes: ", {k: c.dtype for k, c in imap.coords.items()})
    assert imap.coords["p0"].dtype != "object"
    assert imap.coords["p1"].dtype != "object"
    assert imap.coords["p2"].dtype != "object"


def test_mapping_funcs(small_psp):
    """Tests other mapping functions"""
    psp = small_psp

    # Test the get_state_vector method
    assert psp.get_state_vector(state_no=0) == (0, 0, 0)
    assert psp.get_state_vector(state_no=16) == (0, 2, 4)
    assert psp.get_state_vector(state_no=31) == (1, 1, 1)

    with pytest.raises(ValueError, match="Did not find state number -1"):
        psp.get_state_vector(state_no=-1)

    # Test the get_dim_values method
    psp.state_vector = (0, 0, 0)
    assert psp.get_dim_values() == OrderedDict(
        [("p0", 0), ("p1", 0), ("p2", 0)]
    )

    assert list(psp.get_dim_values(state_no=31).values()) == [1, 1, 1]
    assert list(psp.get_dim_values(state_vector=(1, 2, 3)).values()) == [
        1,
        2,
        3,
    ]

    # Should not work for both arguments given
    with pytest.raises(TypeError, match="Expected only one of the arguments"):
        psp.get_dim_values(state_no=123, state_vector=(1, 2, 3))


def test_basic_iteration(small_psp, seq_psp):
    """Tests whether the iteration goes through all points"""
    # Test on the __iter__ and __next__ level
    psp = small_psp
    it = psp.iterator()  # is a generator now
    assert it.__next__() == dict(p0=1, p1=1, p2=1)
    assert psp.state_vector == (1, 1, 1)
    assert psp.state_no == 31  # == 24 + 6 + 1

    assert it.__next__() == dict(p0=1, p1=1, p2=2)
    assert psp.state_vector == (1, 1, 2)
    assert psp.state_no == 31 + 1

    assert it.__next__() == dict(p0=1, p1=1, p2=3)
    assert psp.state_vector == (1, 1, 3)
    assert psp.state_no == 31 + 2

    assert it.__next__() == dict(p0=1, p1=1, p2=4)
    assert it.__next__() == dict(p0=1, p1=1, p2=5)
    assert it.__next__() == dict(p0=1, p1=2, p2=1)
    assert psp.state_vector == (1, 2, 1)
    assert psp.state_no == 31 + 6

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

    # A paramspace without volume should still be iterable
    empty_psp = ParamSpace(dict(foo="bar"))
    assert list(iter(empty_psp)) == [dict(foo="bar")]

    # Check the dry run
    psp.reset()
    snos = list([s for s in psp.iterator(with_info="state_no", omit_pt=True)])
    assert snos[:4] == [31, 32, 33, 34]

    # Check that the counts match using a helper function . . . . . . . . . . .
    def check_counts(iters, counts):
        cntrs = {i: 0 for i, _ in enumerate(counts)}

        for it_no, (it, count) in enumerate(zip(iters, counts)):
            for _ in it:
                cntrs[it_no] += 1
            assert cntrs[it_no] == count

    # For the explicit call
    check_counts(
        (small_psp.iterator(), seq_psp.iterator()),
        (small_psp.volume, seq_psp.volume),
    )

    # For the call via __iter__ and __next__
    check_counts((small_psp, seq_psp), (small_psp.volume, seq_psp.volume))

    # Also test all information tuples and the dry run
    info = ("state_no", "state_vec", "state_no_str", "current_coords")
    check_counts(
        (
            small_psp.iterator(with_info=info),
            empty_psp.iterator(with_info=info),
        ),
        (small_psp.volume, 1),
    )
    check_counts(
        (small_psp.iterator(with_info="state_no"),), (small_psp.volume,)
    )

    # ... and whether invalid values lead to failure
    with pytest.raises(ValueError, match="No such information 'foo bar' avai"):
        info = ("state_no", "foo bar")
        check_counts(
            (small_psp.iterator(with_info=info),), (small_psp.volume,)
        )


# Masking ---------------------------------------------------------------------


def test_masking(small_psp):
    """Test whether the masking feature works"""
    psp = small_psp
    assert psp.shape == (2, 3, 5)
    assert psp.volume == 2 * 3 * 5

    # First try setting binary masks
    psp.set_mask("p0", True)
    assert psp.shape == (1, 3, 5)  # i.e.: 0th dimension only returns default
    assert psp.volume == 1 * 3 * 5

    # Mask completely
    psp.set_mask("p1", True)
    psp.set_mask("p2", True)
    assert psp.shape == (1, 1, 1)  # i.e.: all dimensions masked
    assert psp.volume == 1 * 1 * 1

    # Full volume and full shape should remain the old
    assert psp.full_volume == 2 * 3 * 5
    assert psp.full_shape == (2, 3, 5)

    # Remove the masks again, sometimes partly
    psp.set_mask("p0", False)
    assert psp.shape == (2, 1, 1)

    psp.set_mask("p1", (1, 1, 0))  # length 1
    assert psp.shape == (2, 1, 1)

    psp.set_mask("p2", (1, 0, 0, 1, 0))  # length 3
    assert psp.shape == (2, 1, 3)

    # Try inverting the mask
    psp.set_mask("p2", True, invert=True)
    assert psp.shape == (2, 1, 5)

    # Try setting multiple masks at once
    psp.set_masks(("p0", False), ("p1", False), ("p2", False))
    assert psp.shape == (2, 3, 5)

    psp.set_masks(
        dict(name=("p0",), mask=True),
        dict(name="p2", mask=slice(2), invert=True),
    )
    assert psp.shape == (1, 3, 2)

    # Fully masked dimension also prompts changes in info string
    assert psp.get_info_str().find("fully masked -> using default:") > -1


def test_masked_iteration(small_psp):
    """Check iteration with a masked parameter space"""
    # First test: fully masked array
    psp = small_psp
    psp.set_mask("p0", True)
    psp.set_mask("p1", True)
    psp.set_mask("p2", True)

    # This should return only one entry: the default ParamSpace
    iter_res = {
        state_no: d for d, state_no in psp.iterator(with_info="state_no")
    }
    print("fully masked array: ", iter_res)

    assert len(iter_res) == 1
    assert 0 in iter_res
    assert iter_res[0] == dict(p0=0, p1=0, p2=0) == psp.default

    # Now the same with a non-trivial mask
    psp.set_mask("p0", (1, 0))  # length 1
    assert psp.shape == (1, 1, 1)
    iter_res = {
        state_no: d for d, state_no in psp.iterator(with_info="state_no")
    }
    print("p0 mask (True, False): ", iter_res)
    assert len(iter_res) == 1 == psp.volume
    assert 48 in iter_res
    assert iter_res[48] == dict(p0=2, p1=0, p2=0)

    # ... and an even more complex one
    psp.set_mask("p1", (0, 1, 0))
    assert psp.shape == (1, 2, 1)
    iter_res = {
        state_no: d for d, state_no in psp.iterator(with_info="state_no")
    }
    print("+ p1 mask (False, True, False): ", iter_res)
    assert len(iter_res) == 1 * 2 == psp.volume
    assert iter_res[54] == dict(p0=2, p1=1, p2=0)
    assert iter_res[66] == dict(p0=2, p1=3, p2=0)


def test_active_state_map(small_psp):
    """Test the state map method for masked parameter spaces"""
    psp = small_psp

    # Get the unmasked version
    amap = psp.active_state_map
    print("\nactive state map (of fully unmasked pspace):\n", amap)

    # ... which should be of same shape as the parameter space
    assert amap.shape == psp.shape

    # Assert that it does not have any Masked objects left on the dimensions
    print("coords:", [v for v in amap.coords["p0"].to_series()])
    assert all(
        [not isinstance(v, Masked) for v in amap.coords["p0"].to_series()]
    )
    assert all(
        [not isinstance(v, Masked) for v in amap.coords["p1"].to_series()]
    )
    assert all(
        [not isinstance(v, Masked) for v in amap.coords["p2"].to_series()]
    )

    # Now set a mask and check again
    psp.set_mask("p1", (0, 1, 0))

    amap = psp.active_state_map
    print("\nactive state map (of partly masked pspace):\n", amap)
    assert amap.shape == psp.shape

    # Check the coordinate dtype is not object
    print("coords dtypes: ", {k: c.dtype for k, c in amap.coords.items()})
    assert amap.coords["p0"].dtype != "object"
    assert amap.coords["p1"].dtype != "object"
    assert amap.coords["p2"].dtype != "object"

    # Now mask fully, which should still work
    psp.set_mask("p0", True)
    psp.set_mask("p1", True)
    psp.set_mask("p2", True)

    amap = psp.active_state_map
    print("\nactive state map (of fully masked pspace):\n", amap)
    assert amap.shape == psp.shape


def test_subspace(small_psp, basic_psp):
    """Test the many ways a subspace can be selected."""
    psp = small_psp

    # Test different invocation signatures
    # Via index only
    psp.activate_subspace(
        p0=dict(idx=1), p1=dict(idx=[2, 3]), p2=dict(idx=[3, 4, 5])
    )
    assert psp.volume == 1 * 2 * 3
    assert list(psp.active_state_map.coords["p0"]) == [1]
    assert list(psp.active_state_map.coords["p1"]) == [2, 3]
    assert list(psp.active_state_map.coords["p2"]) == [3, 4, 5]

    # By default, all others masks should have been reset, such that another
    # application would _not_ result in a fully masked parameter space
    psp.activate_subspace(p0=dict(idx=2))
    assert psp.volume == 1 * 3 * 5
    assert list(psp.active_state_map.coords["p0"]) == [2]

    # Via location. Looks the same here due to small_psp pdim values
    # Can use the shorthand here ...
    psp.activate_subspace(p0=1, p1=[2, 3], p2=[3, 4, 5])
    assert psp.volume == 1 * 2 * 3
    assert list(psp.active_state_map.coords["p0"]) == [1]
    assert list(psp.active_state_map.coords["p1"]) == [2, 3]
    assert list(psp.active_state_map.coords["p2"]) == [3, 4, 5]

    # Test slicing
    psp.activate_subspace(
        p0=slice(2),  # -> 1
        p1=slice(1.5, 3.1),  # -> 2, 3
        p2=slice(None, None, 2),
    )  # -> 1, 3, 5
    assert psp.volume == 1 * 2 * 3
    assert list(psp.active_state_map.coords["p0"]) == [1]
    assert list(psp.active_state_map.coords["p1"]) == [2, 3]
    assert list(psp.active_state_map.coords["p2"]) == [1, 3, 5]

    psp.activate_subspace(
        p0=dict(idx=slice(2)),  # -> 1, 2
        p1=dict(idx=slice(1, 3)),  # -> 2, 3
        p2=dict(idx=slice(None, None, 2)),
    )  # -> 1, 3, 5
    assert psp.volume == 2 * 2 * 3
    assert list(psp.active_state_map.coords["p0"]) == [1, 2]
    assert list(psp.active_state_map.coords["p1"]) == [2, 3]
    assert list(psp.active_state_map.coords["p2"]) == [1, 3, 5]

    # Empty arguments
    psp.activate_subspace()
    assert psp.volume == 2 * 3 * 5

    basic_psp.activate_subspace()
    assert basic_psp.volume == 3 ** 6

    # Test the error messages
    # Bad argument combination
    with pytest.raises(ValueError, match="accepting _either_ of the argument"):
        psp.activate_subspace(p0=dict(idx=1, loc=1))

    with pytest.raises(ValueError, match="Missing one of the required"):
        psp.activate_subspace(p0=dict())

    # Bad paramdim name
    with pytest.raises(ValueError, match="'foo' was not found in this"):
        psp.activate_subspace(foo=1)

    # Bad index values
    with pytest.raises(IndexError, match="Encountered index 0 in list of"):
        psp.activate_subspace(p0=dict(idx=0))

    with pytest.raises(IndexError, match="exceeds the highest index, 2."):
        psp.activate_subspace(p0=dict(idx=10))

    with pytest.raises(ValueError, match="at least one duplicate element"):
        psp.activate_subspace(p0=dict(idx=[1, 1, 2]))

    # Bad loc values
    with pytest.raises(KeyError, match="not available as coordinate of this"):
        psp.activate_subspace(p0=[3.14])

    with pytest.raises(ValueError, match="at least one duplicate item!"):
        psp.activate_subspace(p0=[1, 1, 2])

    # allow_default
    with pytest.raises(ValueError, match="'p0' would be totally masked, thus"):
        psp.activate_subspace(p0=[])


def test_subspace_float_locs(float_valued_psp):
    """Test that parameter spaces with float-valued parameter values can also
    be reliably selected.
    """
    psp = float_valued_psp
    assert psp.volume == 11 * 11 * 6

    # Select subspace via location
    psp.activate_subspace(
        lin1=0.2, lin2=[-2.0, -1.6, 0.4], log1=[1.0e-10, 1e10]
    )
    assert psp.volume == 1 * 3 * 2
    assert np.isclose(psp.active_state_map.coords["lin1"], [0.2]).all()
    assert np.isclose(
        psp.active_state_map.coords["lin2"], [-2.0, -1.6, +0.4]
    ).all()
    assert np.isclose(
        psp.active_state_map.coords["log1"], [1.0e-10, 1.0e10]
    ).all()

    # Can also pass a custom tolerance
    psp.activate_subspace()
    assert psp.volume == 11 * 11 * 6

    psp.activate_subspace(
        lin1=dict(loc=0.2 + 1e-6, atol=1e-5),
        lin2=dict(loc=[-2.0 + 1e-16, 0.4 + 1e-16], atol=1e-10),
        log1=dict(
            loc=[1.0e-10 + 1e-16, 1e10 - 1e-16], rtol=1e-10, atol=1e-100
        ),
    )
    assert psp.volume == 1 * 2 * 2
    assert np.isclose(psp.active_state_map.coords["lin1"], [0.2]).all()
    assert np.isclose(psp.active_state_map.coords["lin2"], [-2.0, +0.4]).all()
    assert np.isclose(
        psp.active_state_map.coords["log1"], [1.0e-10, 1.0e10]
    ).all()


def test_subspace_str_locs(str_valued_psp):
    """Test that parameter spaces with str-valued parameter values can also
    be reliably selected.
    """
    psp = str_valued_psp
    assert psp.volume == 3 * 5 * 6

    # Select subspace via location
    psp.activate_subspace(p0="foo", p1=["1", "3", "4"])
    assert psp.volume == 1 * 3 * 6
    assert (psp.active_state_map.coords["p0"] == ["foo"]).all()
    assert (psp.active_state_map.coords["p1"] == ["1", "3", "4"]).all()


def test_subspace_idx_paths(seq_psp):
    """Test that parameter spaces with paramter dimensions that include indices
    can also be reliably selected
    """
    psp = seq_psp
    assert psp.volume == 3 ** 7

    # Select subspace via location
    psp.activate_subspace(**{".s.0": 1}, dds2=[1, 2])
    assert psp.volume == 1 * 2 * 3 ** (7 - 2)
    assert (psp.active_state_map.coords[".s.0"] == [1]).all()
    assert (psp.active_state_map.coords["dds2"] == [1, 2]).all()


def test_subspace_mixed_values():
    """Test that parameter spaces with parameter dimensions that have mixed
    values raise the expected error message.
    """
    psp = ParamSpace(
        dict(
            all_str=ParamDim(default="foo", values=["foo", "bar", "baz"]),
            mixed=ParamDim(default="0", values=[0.0, "1", "2", 3, "4"]),
        )
    )
    assert psp.volume == 3 * 5

    # These should fail due to a mixed-type dimension is tried to be masked
    with pytest.raises(TypeError, match="Could not ascertain whether"):
        psp.activate_subspace(mixed=0.0)

    with pytest.raises(TypeError, match="Could not ascertain whether"):
        psp.activate_subspace(all_str="foo", mixed=[0.0, "1"])

    with pytest.raises(TypeError, match="Could not ascertain whether"):
        psp.activate_subspace(all_str="foo", mixed=["1", "2"])

    # ... but this should work, as the mixed-type dimension is not touched
    psp.activate_subspace(all_str="foo")
    assert psp.volume == 1 * 5


# Complicated content ---------------------------------------------------------


def test_coupled(psp_with_coupled):
    """Test parameter spaces with CoupledParamDims in them"""
    psp = psp_with_coupled
    print("ParamSpace with CoupledParamDim:\n", psp)

    def assert_coupling(src: tuple, target: tuple):
        """Asserts that the CoupledParamDim at keyseq src is coupled to the target ParamDim at keyseq target."""
        assert (
            psp.coupled_dims_by_loc[src].target_pdim == psp.dims_by_loc[target]
        )

    # Assert correct coupling
    assert_coupling(("c1",), ("a",))
    assert_coupling(("d", "cc1"), ("d", "aa"))
    assert_coupling(("d", "cc2"), ("a",))
    assert_coupling(("d", "cc3"), ("d", "aa"))

    # Check default is correct
    default = psp.default

    assert default["c1"] == default["a"]
    assert default["d"]["cc1"] == default["d"]["aa"]
    assert default["d"]["cc2"] == default["a"]
    assert default["d"]["cc3"] == default["d"]["aa"]

    # Iterate over the paramspace and check correctness
    for pt in psp:
        print("Point: ", pt)

        assert pt["c1"] == pt["a"]
        assert pt["d"]["cc1"] == pt["d"]["aa"]
        assert pt["d"]["cc2"] == pt["a"]
        assert pt["d"]["cc3"] == pt["d"]["aa"]

    # Invalid coupling targets should raise an error
    with pytest.raises(ValueError, match="Could not resolve the coupling"):
        ParamSpace(
            dict(
                a=ParamDim(default=0, range=[10]),
                b=CoupledParamDim(target_name="foo"),
            )
        )


def test_nested(psp_nested, basic_psp):
    """Tests whether nested ParamSpaces behave as desired"""
    default = psp_nested.default

    assert default["foo"] == "bar"
    assert default["basic"] == basic_psp
    assert default["deeper"]["basic"] == basic_psp


# YAML Dumping ----------------------------------------------------------------


def test_yaml_unsafe_dump_and_load(
    tmpdir, small_psp, adv_psp, seq_psp, psp_with_coupled
):
    """Tests that YAML dumping and reloading works"""
    yaml = yaml_unsafe

    d_out = dict(
        small=small_psp, adv=adv_psp, seq=seq_psp, coupled=psp_with_coupled
    )
    path = tmpdir.join("out.yml")

    # Dump it
    with open(path, "x") as out_file:
        yaml.dump(d_out, stream=out_file)

    # Read what was written
    with open(path) as in_file:
        print("Content of written file:\n")
        print("".join(in_file.readlines()))
        print("--- end of file ---")

    # Load it again
    with open(path) as in_file:
        d_in = yaml.load(in_file)

    # Check that the contents are equivalent
    for k_out, v_out in d_out.items():
        assert k_out in d_in
        psp = d_in[k_out]

        print(
            k_out,
            ":",
        )
        print(
            "  dims:   \n   ",
            "\n    ".join([f"{k}: {v}" for k, v in psp.dims.items()]),
        )

        print(
            "  coupled:\n   ",
            "\n    ".join([f"{k}: {v}" for k, v in psp.coupled_dims.items()]),
        )

        assert v_out == psp


def test_yaml_safe_dump_and_load(
    tmpdir, small_psp, adv_psp, seq_psp, psp_with_coupled
):
    """Tests that YAML dumping and reloading works with both default dump and
    load methods as well as with the safe versions.
    """

    def dump_load_assert_equal(d_out: dict, *, path, dump_func, load_func):
        """Helper method for dumping, loading, and asserting equality"""
        # Dump it
        with open(path, "x") as out_file:
            dump_func(d_out, stream=out_file)

        # Read it in again
        with open(path) as in_file:
            d_in = load_func(in_file)

        # Check that the contents are equivalent
        assert d_out == d_in

    # Use the dict of ParamDim objects for testing
    d_out = dict(
        small=small_psp, adv=adv_psp, seq=seq_psp, coupled=psp_with_coupled
    )

    # Test all possible combinations of dump and load methods
    methods = [
        ("unsafe-unsafe", yaml_unsafe.dump, yaml_unsafe.load),
        ("unsafe-safe", yaml_unsafe.dump, yaml_safe.load),
        ("safe-unsafe", yaml_safe.dump, yaml_unsafe.load),
        ("safe-safe", yaml_safe.dump, yaml_safe.load),
    ]

    for prefix, dump_func, load_func in methods:
        # Generate file name and some output to know what went wrong ...
        fname = prefix + ".yml"
        path = tmpdir.join(fname)

        print(f"Now testing combination:  {prefix}  ... ", end="")

        # Call the test function
        dump_load_assert_equal(
            d_out, path=path, dump_func=dump_func, load_func=load_func
        )

        print("Works!")
