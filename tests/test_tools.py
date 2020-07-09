"""Tests for the tool functions of the paramspace package"""
import copy

import pytest

from paramspace import tools as t


# Dummy objects to find using the tools
class Dummy:
    def __init__(self, i: int):
        self.i = i


# -----------------------------------------------------------------------------


def test_create_indices():
    """Test the create_indices function"""
    ci = t.create_indices
    assert ci() == []
    assert ci(from_range=[0, 10, 2]) == list(range(0, 10, 2))
    assert ci(append=[1, 1, 1], unique=True) == [1]
    assert ci(append=[1, 1, 1, 2], remove=[1]) == [2]
    assert ci(append=[3, 2, 1], remove=[2], sort=True) == [1, 3]
    assert ci(append=[3, 2, 1], remove=[2], sort=False) == [3, 1]


def test_contains():
    """Tests the recursive_contains function"""
    d = dict(
        a=dict(b=dict(c=dict(d=["e"], _="foo"), _="foo"), _="foo"), _="foo"
    )

    # Test both possible cases
    assert t.recursive_contains(d, keys=("a", "b", "c", "d", "e")) == True
    assert t.recursive_contains(d, keys=("a", "b", "bar", "d")) == False

    # There should be a TypeError if the last element was a list
    with pytest.raises(TypeError):
        assert t.recursive_contains(d, keys=("a", "b", "c", "d", "e", "f"))


def test_getitem():
    """Tests the recursive_getitem function"""
    d = dict(a=dict(b=dict(c=dict(d=[0])), l=[dict(l0l=0)]))

    # Should pass
    assert 0 == t.recursive_getitem(d, keys=("a", "b", "c", "d", 0))
    assert 0 == t.recursive_getitem(d, keys=("a", "l", 0, "l0l"))

    # Should fail
    with pytest.raises(KeyError):
        t.recursive_getitem(d, keys=("a", "l", 0, "l"))
    with pytest.raises(KeyError):
        t.recursive_getitem(d, keys=("a", "x", "c", "d"))
    with pytest.raises(IndexError):
        t.recursive_getitem(d, keys=("a", "b", "c", "d", 1))
    with pytest.raises(IndexError):
        t.recursive_getitem(d, keys=("a", "l", 1, "l0l"))


def test_collect():
    """Tests the recursive_collect function"""
    collect = t.recursive_collect

    # Generate a few objects to find
    find_me = [Dummy(i) for i in range(5)]

    d = dict(
        a=1,
        b=tuple([1, 2, find_me[0], find_me[1]]),
        c=find_me[2],
        d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])),
    )
    # The keys at which the dummies sit
    find_keys = [("b", 2), ("b", 3), ("c",), ("d", "aa"), ("d", "cc", "aa")]
    find_dvals = [d.i for d in find_me]

    # A selection function
    sel_dummies = lambda x: isinstance(x, Dummy)

    # Test if all dummies are found
    assert find_me == collect(d, select_func=sel_dummies)

    # Test if they return the correct prepended info key
    for k, ctup in zip(
        find_keys, collect(d, select_func=sel_dummies, prepend_info=("keys",))
    ):
        assert k == ctup[0]

    # Test if info func works
    for i, ctup in zip(
        find_dvals,
        collect(
            d,
            select_func=sel_dummies,
            prepend_info=("info_func",),
            info_func=lambda d: d.i,
        ),
    ):
        assert i == ctup[0]

    # Test if error is raised for wrong prepend info keys
    with pytest.raises(ValueError):
        collect(d, select_func=sel_dummies, prepend_info=("invalid_entry",))


def test_replace():
    """Tests the recursive_replace function"""
    replace = t.recursive_replace
    collect = t.recursive_collect

    # Generate a few objects to replace
    find_me = [Dummy(float(i)) for i in range(5)]

    # Try if replacement works (using collect function)
    d = dict(
        a=1,
        b=[1, 2, find_me[0], find_me[1]],
        c=find_me[2],
        d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])),
    )
    d = replace(
        d,
        select_func=lambda x: isinstance(x, Dummy),
        replace_func=lambda dummy: dummy.i,
    )
    assert [d.i for d in find_me] == collect(
        d, select_func=lambda x: isinstance(x, float)
    )

    # See if it fails with immutable containers
    with pytest.raises(TypeError):
        d = dict(
            a=1,
            b=tuple([1, 2, find_me[0], find_me[1]]),
            c=find_me[2],
            d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])),
        )
        d = replace(
            d,
            select_func=lambda x: isinstance(x, Dummy),
            replace_func=lambda dummy: dummy.i,
        )

    # Should not work with an immutable object
    with pytest.raises(TypeError):
        replace(
            tuple(range(3)),
            select_func=lambda v: isinstance(v, int),
            replace_func=lambda *args: 0,
        )


def test_setitem():
    """Tests the recursive_setitem function"""
    setitem = t.recursive_setitem

    # Create a dict to fill
    d = dict(
        a=1,
        b=[1, 2, 3],
        c=("foo",),
        d=dict(aa=1.23, bb=2.34, cc=dict(aa=4.56)),
    )

    # top level
    setitem(d, keys=("a",), val=2)
    assert d["a"] == 2

    # lower level
    setitem(d, keys=("d", "aa"), val=-1.23)
    assert d["d"]["aa"] == -1.23

    setitem(d, keys=("d", "cc", "aa"), val=-4.56)
    assert d["d"]["cc"]["aa"] == -4.56

    # non-existing key on the way
    with pytest.raises(KeyError, match="No key 'dd' found in dict"):
        setitem(d, keys=("d", "dd", "dd"), val=3.45, create_key=False)

    setitem(d, keys=("d", "dd", "ddd"), val=3.45, create_key=True)
    assert d["d"]["dd"]["ddd"] == 3.45


def test_update():
    """Tests the recursive_update function"""
    d = dict(
        a=1,
        b=2,
        d=dict(a=1, b=2, d=dict(a=1, b=2, d="not_a_dict")),
        l1=[1, 2, 3],
        l2="foobar",
        l3=[1, 2, 3],
        l4=(1, 2, 3, 4),
        l5=[1, 2, 3, dict(a=1)],
        l6=[1, 2, 3],
        fail_conversion=None,
    )

    u = dict(
        a=2,
        b=3,
        c=4,
        d=dict(a=2, b=3, c=4, d=dict(a=2, b=3, c=4, d=dict(a=2))),
        l1=[2, 3, 4, 5],  # longer here
        l2=[2, 3, 4],  # not a list in obj
        l3=[2, 3],  # shorter here
        l4=[2, 3],  # tuple in obj,
        l5=[2, 3, 4, dict(a=2, b=3,)],  # tuple in obj,
        l6=[t.SKIP, t.SKIP, 4],  # skip updating
        fail_conversion=["none"],
    )

    du = t.recursive_update(copy.deepcopy(d), u)
    # Check the dict values
    assert du["a"] == 2
    assert du["b"] == 3
    assert du["c"] == 4
    assert du["d"]["a"] == 2
    assert du["d"]["b"] == 3
    assert du["d"]["c"] == 4

    # Check the lists
    assert du["l1"] == [2, 3, 4, 5]
    assert du["l2"] == [2, 3, 4]
    assert du["l3"] == [2, 3, 3]

    # With trying to convert the lists
    with pytest.warns(UserWarning):
        du = t.recursive_update(copy.deepcopy(d), u, try_list_conversion=True)

    assert du["l1"] == [2, 3, 4, 5]
    assert du["l2"] == [2, 3, 4]
    assert du["l3"] == [2, 3, 3]
    assert du["l4"] == [2, 3, 3, 4]
    assert du["l5"] == [2, 3, 4, dict(a=2, b=3)]
    assert du["l6"] == [1, 2, 4]
