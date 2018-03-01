"""The tools module provides methods that both the ParamSpan and ParamSpace classes use or depend upon."""

import copy
import logging
import pprint
import collections
from collections import OrderedDict, Mapping
from typing import Union, Sequence, Mapping, Callable, Iterator, MutableSequence, MutableMapping, Collection

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

def recursive_contains(d: Collection, *, keys: Sequence) -> bool:
    """Checks on the Mapping d, whether a certain key sequence is reachable.
    
    Args:
        d (Collection): The collection to go through
        keys (Sequence): The sequence of keys to check for
    
    Returns:
        bool: Whether the key sequence is reachable
    """
    if len(keys) > 1:
        # Check and continue recursion
        if keys[0] in d:
            return recursive_contains(d[keys[0]], keys=keys[1:])
        else:
            return False
    else:
        # reached the end of the recursion
        return keys[0] in d

def recursive_getitem(d: dict, *, keys: Sequence):
    """Recursively goes through dict-like d along the keys in tuple keys and returns the reference to the at the end."""
    keyerr_fstr = "No such key '{}' of key sequence {} is available in {}."
    idxerr_fstr = "No such index '{}' of key sequence {} is available in {}."

    if len(keys) > 1:
        # Check and continue recursion
        try:
            return recursive_getitem(d[keys[0]], keys=keys[1:])
        except KeyError as err:
            raise KeyError(keyerr_fstr.format(keys[0], keys, d)) from err
        except IndexError as err:
            raise IndexError(idxerr_fstr.format(keys[0], keys, d)) from err
    else:
        # reached the end of the recursion
        try:
            return d[keys[0]]
        except KeyError as err:
            raise KeyError(keyerr_fstr.format(keys[0], keys, d)) from err
        except IndexError as err:
            raise IndexError(idxerr_fstr.format(keys[0], keys, d)) from err

def recursive_update(d: dict, u: dict):
    """Update Mapping d with values from Mapping u"""
    for k, v in u.items():
        if isinstance(d, Mapping):
            # Already a Mapping
            if isinstance(v, Mapping):
                # Already a Mapping, continue recursion
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                # Not a mapping -> at leaf -> update value
                d[k] = v    # ... which is just u[k]
        else:
            # Not a mapping -> create one
            d = {k: u[k]}
    return d

def recursive_setitem(d: dict, *, keys: tuple, val, create_key: bool=False):
    """Recursively goes through dict-like d along the keys in tuple keys and sets the value to the child entry."""
    if len(keys) > 1:
        # Check and continue recursion
        if keys[0] in d:
            recursive_setitem(d=d[keys[0]], keys=keys[1:],
                               val=val, create_key=create_key)
        else:
            if create_key:
                d[keys[0]]  = {}
                recursive_setitem(d=d[keys[0]], keys=keys[1:],
                                   val=val, create_key=create_key)
            else:
                raise KeyError("No key '{}' found in dict {}; if it should be created, set create_key argument to True.".format(keys[0], d))
    else:
        # reached the end of the recursion
        d[keys[0]]  = val

def recursive_collect(obj: Union[Mapping, Sequence], *, select_func: Callable, prepend_info: Sequence=None, info_func: Callable=None, _parent_keys: tuple=None) -> list:
    """Go recursively through a mapping or sequence and collect selected elements.
    
    The `select_func` is called on each values. If the return value is True, that value will be collected to a list, which is returned at the end.
    
    Additionally, some information can be gathered about these elements, controlled by `prepend_info`
    
    With `prepend_info`, information can be prepended to the return value. Then, not only the values but also these additional items can be gathered:
        `keys`      : prepends the key
        `info_func` : prepends the return value of `info_func(val)`
    The resulting return value is then a list of tuples (in that order).
    
    Args:
        cont (Union[Mapping, Sequence]): The mapping or sequence to
            recursively search through
        select_func (Callable): Each element is passed to this function; if
            True is returned, the element is collected and search ends here.
        prepend_info (Sequence, optional): If given, additional info about the
            selected elements can be gathered. 1) By passing `keys`, the
            sequence of keys to get to this element is appended; 2) by passing
            `info_func`, the `info_func` function is called on the argument
            and that value is added to the tuple.
        info_func (Callable, optional): The function used to prepend info
        _parent_keys (tuple, optional): Used to track the keys; not public!
    
    Returns:
        list: the collected elements, as selected by select_func(val) or -- if 
            `prepend_info` was set -- tuples of (info, element), where the 
            requested information is in the first entries of the tuple
    
    Raises:
        ValueError: Raised if invalid `prepend_info` entries were set
    """
    log.debug("recursive_collect called")

    # Return value list
    coll = []

    # Now go through all values
    for key, val in get_key_val_iter(obj):
        # Generate the tuple of parent keys... for this iterator of the loop
        if _parent_keys is None:
            these_keys = (key,)
        else:
            these_keys = _parent_keys + (key,)

        # Apply the select_func and, depending on return, continue recursion or not
        if select_func(val):
            # found the desired element
            # Distinguish cases where information is prepended and where not
            if not prepend_info:
                entry = val
            else:
                entry = (val,)
                # Loop over the keys to prepend in reversed order (such that the order of the given tuple is not inverted)
                for info in reversed(prepend_info):
                    if info in ['key', 'keys', 'keyseq', 'keysequence']:
                        entry = (these_keys,) + entry
                    elif info in ['info_func']:
                        entry = (info_func(val),) + entry
                    else:
                        raise ValueError("No such `prepend_info` entry implemented: "+str(info))

            # Add it to the return list
            coll.append(entry)

        elif is_iterable(val):
            # Not the desired element, but recursion possible ...
            coll += recursive_collect(val,
                                      select_func=select_func,
                                      prepend_info=prepend_info,
                                      info_func=info_func,
                                      _parent_keys=these_keys)

        # else: is something that cannot be selected and cannot be further recursed ...

    return coll

def recursive_replace(obj: Union[Mapping, Sequence], *, select_func: Callable, replace_func: Callable) -> Union[Mapping, Sequence]:
    """Go recursively through a mapping or sequence and call a replace function on each element that the select function returned true on.
    
    For passing arguments to any of the two, use lambda functions.    
    
    Args:
        cont (Union[Mapping, Sequence]): The mapping or sequence to go through
            recursively
        select_func (Callable): The function that each value is passed to
        replace_func (Callable): The replacement function, called if the
            selection function returned True on an element of the mapping
    
    Returns:
        Union[Mapping, Sequence]: The updated mapping where each element that was selected was replaced by the return value of the replacement function.
    """

    log.debug("recursive_replace called")

    def replace(ms, *, key, replace_by):
        """Try to replace the entry; catch exception"""
        try:
            ms[key] = replace_by
        except TypeError as err:
            raise TypeError("Failed to replace element via item assignment; probably because given container type ({}) was not mutable for key '{}'.".format(type(ms), key)) from err

    # Go through all items
    for key, val in get_key_val_iter(obj):
        if select_func(val):
            # found the desired element -> replace by the value returned from the replace_func
            replace(obj, key=key,
                    replace_by=replace_func(val))

        elif is_iterable(val):
            # Not the desired element, but recursion possible ...
            replace(obj, key=key,
                    replace_by=recursive_replace(val,
                                                 select_func=select_func,
                                                 replace_func=replace_func))

        # else: was not selected and cannot be further recursed, thus: stays the same

    return obj

# Helpers ---------------------------------------------------------------------
def is_iterable(obj) -> bool:
    """Tests if the given object is iterable or not
    
    Args:
        obj: The object to test
    
    Returns:
        bool: True if iterable, False else
    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True

def get_key_val_iter(obj: Union[Mapping, Sequence]) -> Iterator:
    """Given an object -- assumed dict- or sequence-like -- returns a (key, value) iterator.
    
    Args:
        obj (Union[Mapping, Sequence]): The obj to generate the key-value iter from
    
    Returns:
        Iterator: An iterator that emits (key, value) tuples
    """
    # Distinguish different ways of iterating over it
    if hasattr(obj, 'items') and callable(obj.items):
        # assume it is dict-like
        return obj.items()
    else:
        # assume sequence-like
        return enumerate(obj)
