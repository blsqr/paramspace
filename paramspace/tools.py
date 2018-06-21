"""This module provides general methods needed by the ParamSpan and ParamSpace classes."""

import warnings
import logging
import collections
from typing import Union, Callable, Iterator, Sequence, Mapping, List

# Get logger
log = logging.getLogger(__name__)

# Special class for indicating that a value is to be skipped
# Used in recursive_update
class Skip:
    """A skip object can be used to indiciate that no action should be taken."""
    pass

# Also initialise such an object, simplifying the calls
SKIP = Skip()

# -----------------------------------------------------------------------------

def recursive_contains(obj: Union[Mapping, Sequence], *, keys: Sequence) -> bool:
    """Checks whether the given keysequence is reachable in the `obj`.
    
    Args:
        obj (Union[Mapping, Sequence]): The object to check recursively
        keys (Sequence): The sequence of keys to check for
    
    Returns:
        bool: Whether the key sequence is reachable
    """
    if len(keys) > 1:
        # Check and continue recursion
        if keys[0] in obj:
            return recursive_contains(obj[keys[0]], keys=keys[1:])
        # else: not available
        return False
    # else: reached the end of the recursion
    return keys[0] in obj

def recursive_getitem(obj: Union[Mapping, Sequence], *, keys: Sequence):
    """Go along the sequence of `keys` through `obj` and return the target item.
    
    Args:
        obj (Union[Mapping, Sequence]): The object to get the item from
        keys (Sequence): The sequence of keys to follow
    
    Returns:
        The target item from `obj`, specified by `keys`
    
    Raises:
        IndexError: If any index in the key sequence was not available
        KeyError: If any key in the key sequence was not available
    """
    # Define some error format strings
    keyerr_fstr = "No such key '{}' of key sequence {} is available in {}."
    idxerr_fstr = "No such index '{}' of key sequence {} is available in {}."

    if len(keys) > 1:
        # Check and continue recursion
        try:
            return recursive_getitem(obj[keys[0]], keys=keys[1:])
        except KeyError as err:
            raise KeyError(keyerr_fstr.format(keys[0], keys, obj)) from err
        except IndexError as err:
            raise IndexError(idxerr_fstr.format(keys[0], keys, obj)) from err
    else:
        # reached the end of the recursion
        try:
            return obj[keys[0]]
        except KeyError as err:
            raise KeyError(keyerr_fstr.format(keys[0], keys, obj)) from err
        except IndexError as err:
            raise IndexError(idxerr_fstr.format(keys[0], keys, obj)) from err

def recursive_update(obj: Union[Mapping, List], upd: Union[Mapping, List], *, try_list_conversion: bool=False, no_convert: tuple=(str,)) -> Union[Mapping, List]:
    """Recursively update items in `obj` with the values from `upd`.
    
    Be aware that objects are not copied from `upd` to `obj`, but only
    assigned. This means:
        * the given `obj` will be changed in place
        * changing mutable elements in `obj` will also change them in `upd`
    
    After the update, `obj` holds all entries of `upd` plus those that it did
    not have in common with `upd`.
    
    If recursion is possible is determined by type; it is only done for types
    mappings (dicts) or lists.

    To indicate that a value in a list should not be updated, an instance of
    the tools.Skip class, e.g. the tools.SKIP object, can be passed instead.
    
    Args:
        obj (Union[Mapping, List]): The object to update.
        upd (Union[Mapping, List]): The object to use for updating.
        try_list_conversion (bool, optional): If true, it is tried to convert
            an entry in `obj` to a list if it is a list in `upd`
        no_convert (tuple, optional): For these types conversion is skipped 
            and an empty list is generated instead
    
    Returns:
        Union[Mapping, List]: The updated `obj`
    """
    # Distinguish the cases where `upd` is a mapping and a list
    if isinstance(upd, collections.abc.Mapping):
        # Check if the target object is of the correct type
        if not isinstance(obj, collections.abc.Mapping):
            # Discard the old object and use a dict instead
            obj = dict()

        # Go over the items of `upd` and ensure that they will be set.
        for key, val in upd.items():
            # The target object is already a mapping.
            # Can now either recurse or set the value, depending on val
            if isinstance(val, (collections.abc.Mapping, list)):
                obj[key] = recursive_update(obj.get(key, {}), val,
                                            try_list_conversion=try_list_conversion,
                                            no_convert=no_convert)
                # NOTE the .get also creates an empty mapping, if needed
            else:
                obj[key] = val
        return obj

    elif isinstance(upd, list):
        # Ensure that the target object is also a list
        if not isinstance(obj, list):
            if not try_list_conversion or isinstance(obj, no_convert):
                # Discard the whole list and start with an empty one
                obj = list()

            else:
                # Try conversion, falling back to list if this fails
                try:
                    obj = list(obj)
                except Exception as err:
                    warnings.warn("Could not convert object of type {} "
                                  "to a list, got {}:{}.\nUsing empty list "
                                  "instead.".format(type(obj),
                                                    err.__class__.__name__,
                                                    str(err)),
                                  UserWarning)
                    # Ditch the list
                    obj = list()

        # It is now ensured that `obj` is a list
        # Need to check that there are enough elements in the `obj` list
        if len(obj) < len(upd):
            obj += [None for _ in range(len(upd) - len(obj))]

        # Go over the items of `upd` and ensure that they will be set.
        for idx, val in enumerate(upd):
            # Option to skip this value
            if isinstance(val, Skip):
                continue

            # Determine whether recursion needs to start/continue
            if isinstance(val, (collections.abc.Mapping, list)):
                # Continue recursion
                obj[idx] = recursive_update(obj[idx], val,
                                            try_list_conversion=try_list_conversion,
                                            no_convert=no_convert)
                # NOTE it was ensured above that this element is available
            else:
                obj[idx] = val
        
        return obj
    # else: this case is logically impossible

def recursive_setitem(d: dict, *, keys: tuple, val, create_key: bool=False):
    """Recursively goes through dict-like d along the keys in tuple keys and sets the value to the child entry."""
    if len(keys) > 1:
        # Check and continue recursion
        if keys[0] in d:
            recursive_setitem(d=d[keys[0]], keys=keys[1:],
                               val=val, create_key=create_key)
        else:
            if create_key:
                d[keys[0]] = {}
                recursive_setitem(d=d[keys[0]], keys=keys[1:],
                                   val=val, create_key=create_key)
            else:
                raise KeyError("No key '{}' found in dict {}; if it should be created, set create_key argument to True.".format(keys[0], d))
    else:
        # reached the end of the recursion
        d[keys[0]] = val

def recursive_collect(obj: Union[Mapping, Sequence], *, select_func: Callable, prepend_info: Sequence=None, info_func: Callable=None, stop_recursion_types: tuple=None, _parent_keys: tuple=None) -> list:
    """Go recursively through a mapping or sequence and collect selected elements.
    
    The `select_func` is called on each values. If the return value is True, that value will be collected to a list, which is returned at the end.
    
    Additionally, some information can be gathered about these elements, controlled by `prepend_info`
    
    With `prepend_info`, information can be prepended to the return value. Then, not only the values but also these additional items can be gathered:
        `keys`      : prepends the key
        `info_func` : prepends the return value of `info_func(val)`
    The resulting return value is then a list of tuples (in that order).
    
    Args:
        obj (Union[Mapping, Sequence]): The object to recursively search
        select_func (Callable): Each element is passed to this function; if
            True is returned, the element is collected and search ends here.
        prepend_info (Sequence, optional): If given, additional info about the
            selected elements can be gathered. 1) By passing `keys`, the
            sequence of keys to get to this element is appended; 2) by passing
            `info_func`, the `info_func` function is called on the argument
            and that value is added to the tuple.
        info_func (Callable, optional): The function used to prepend info
        stop_recursion_types (tuple, optional): Can specify types here that
            will not be further searched through.
            NOTE: strings are never iterated through further
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

    # Compile the list of types that should not be recursed further
    if not stop_recursion_types:
        # Specify the types that should never be further recursed
        stop_recursion_types = (str, )
    else:
        # Assure that strings are never recursed further
        if str not in stop_recursion_types:
            stop_recursion_types += (str,)

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

        elif is_iterable(val) and not isinstance(val, stop_recursion_types):
            # Not the desired element, but recursion possible ...
            coll += recursive_collect(val,
                                      select_func=select_func,
                                      prepend_info=prepend_info,
                                      info_func=info_func,
                                      stop_recursion_types=stop_recursion_types,
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

        elif is_iterable(val) and not isinstance(val, str):
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
    # assume sequence-like
    return enumerate(obj)