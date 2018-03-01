"""Tests for the tool functions of the paramspace package"""

import pytest
import paramspace.tools as t

# Dummy objects to find using the tools
class Dummy:
    def __init__(self, i: int):
        self.i = i

def test_contains():
    """Tests the recursive_contains function"""
    d = dict(a=dict(b=dict(c=dict(d=['e'], _="foo"), _="foo"), _="foo"), _="foo")

    # Test both possible cases
    assert t.recursive_contains(d, keys=('a', 'b', 'c', 'd', 'e')) == True
    assert t.recursive_contains(d, keys=('a', 'b', 'bar', 'd')) == False

    # There should be a TypeError if the last element was a list
    with pytest.raises(TypeError):
        assert t.recursive_contains(d, keys=('a', 'b', 'c', 'd', 'e', 'f'))

def test_getitem():
    """Tests the recursive_getitem function"""
    d = dict(a=dict(b=dict(c=dict(d=[0])), l=[dict(l0l=0)]))

    # Should pass
    assert 0 == t.recursive_getitem(d, keys=('a', 'b', 'c', 'd', 0))
    assert 0 == t.recursive_getitem(d, keys=('a', 'l', 0, 'l0l'))

    # Should fail
    with pytest.raises(KeyError):
        t.recursive_getitem(d, keys=('a', 'l', 0, 'l'))
    with pytest.raises(KeyError):
        t.recursive_getitem(d, keys=('a', 'x', 'c', 'd'))
    with pytest.raises(IndexError):
        t.recursive_getitem(d, keys=('a', 'b', 'c', 'd', 1))
    with pytest.raises(IndexError):
        t.recursive_getitem(d, keys=('a', 'l', 1, 'l0l'))

def test_collect():
    """Tests the recursive_collect function"""
    collect = t.recursive_collect

    # Generate a few objects to find
    find_me = [Dummy(i) for i in range(5)]

    d = dict(a=1,
             b=tuple([1,2,find_me[0], find_me[1]]),
             c=find_me[2],
             d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])))
    # The keys at which the dummies sit
    find_keys = [('b', 2), ('b', 3), ('c',), ('d', 'aa'), ('d', 'cc', 'aa')]
    find_dvals = [d.i for d in find_me]

    # A selection function
    sel_dummies = lambda x: isinstance(x, Dummy)

    # Test if all dummies are found
    assert find_me == collect(d, select_func=sel_dummies)

    # Test if they return the correct prepended info key
    for k, ctup in zip(find_keys, collect(d, select_func=sel_dummies,
                                          prepend_info=('keys',))):
        assert k == ctup[0]

    # Test if info func works
    for i, ctup in zip(find_dvals, collect(d, select_func=sel_dummies,
                                           prepend_info=('info_func',),
                                           info_func=lambda d: d.i)):
        assert i == ctup[0]

    # Test if error is raised for wrong prepend info keys
    with pytest.raises(ValueError):
        collect(d, select_func=sel_dummies, prepend_info=('invalid_entry',))


def test_replace():
    """Tests the recursive_replace function"""
    replace = t.recursive_replace
    collect = t.recursive_collect

    # Generate a few objects to replace
    find_me = [Dummy(float(i)) for i in range(5)]

    # Try if replacement works (using collect function)
    d = dict(a=1,
             b=[1,2,find_me[0], find_me[1]],
             c=find_me[2],
             d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])))
    d = replace(d, select_func=lambda x: isinstance(x, Dummy),
                replace_func=lambda dummy: dummy.i)
    assert [d.i for d in find_me] == collect(d, select_func=lambda x: isinstance(x, float))

    # See if it fails with immutable containers
    with pytest.raises(TypeError):
        d = dict(a=1,
                 b=tuple([1,2,find_me[0], find_me[1]]),
                 c=find_me[2],
                 d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])))
        d = replace(d, select_func=lambda x: isinstance(x, Dummy),
                    replace_func=lambda dummy: dummy.i)

    # Should not work with an immutable object
    with pytest.raises(TypeError):
        replace(tuple(range(3)),
                select_func=lambda v: isinstance(v, int),
                replace_func=lambda *args: 0)