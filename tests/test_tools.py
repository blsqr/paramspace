"""Tests for the tool functions of the paramspace package"""

import pytest
import paramspace.tools as t

# Dummy objects to find using the tools
class Dummy:
    def __init__(self, i: int):
        self.i = i


def test_collect():
    """Tests the recursive_collect function"""
    collect = t.recursive_collect

    # Generate a few objects to find
    find_me = [Dummy(i) for i in range(5)]

    d = dict(a=1,
             b=tuple([1,2,find_me[0], find_me[1]]),
             c=find_me[2],
             d=dict(aa=find_me[3], bb=2, cc=dict(aa=find_me[4])))

    assert find_me == collect(d, select_func=lambda x: isinstance(x, Dummy))


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