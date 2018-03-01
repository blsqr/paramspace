"""Tests for the tool functions of the paramspace package"""

import pytest
import paramspace.tools as t

def test_replace():
    """Tests the recursive_replace function"""
    func = t.recursive_replace

    with pytest.raises(TypeError):
        func(tuple(range(3)),
             select_func=lambda v: isinstance(v, int),
             replace_func=lambda *args: 0)