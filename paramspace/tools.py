"""The tools module provides methods that both the ParamSpan and ParamSpace classes use or depend upon."""

import copy
import logging
import pprint
from collections import OrderedDict, Mapping

import numpy as np

# TODO clean these up
# TODO let them not be named private

# Get logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

def _recursive_update(d: dict, u: dict):
	"""Update Mapping d with values from Mapping u"""
	for k, v in u.items():
		if isinstance(d, Mapping):
			# Already a Mapping
			if isinstance(v, Mapping):
				# Already a Mapping, continue recursion
				d[k] = _recursive_update(d.get(k, {}), v)
			else:
				# Not a mapping -> at leaf -> update value
				d[k] = v 	# ... which is just u[k]
		else:
			# Not a mapping -> create one
			d = {k: u[k]}
	return d

def _recursive_contains(d: dict, keys: tuple):
	"""Checks on the dict-like d, whether a key is present. If the key is a tuple with more than one key, it recursively continues checking."""
	if len(keys) > 1:
		# Check and continue recursion
		if keys[0] in d:
			return _recursive_contains(d[keys[0]], keys[1:])
		else:
			return False
	else:
		# reached the end of the recursion
		return keys[0] in d

def _recursive_getitem(d: dict, keys: tuple):
	"""Recursively goes through dict-like d along the keys in tuple keys and returns the reference to the at the end."""
	if len(keys) > 1:
		# Check and continue recursion
		if keys[0] in d:
			return _recursive_getitem(d[keys[0]], keys[1:])
		else:
			raise KeyError("No key '{}' found in dict {}.".format(keys[0], d))
	else:
		# reached the end of the recursion
		return d[keys[0]]

def _recursive_setitem(d: dict, keys: tuple, val, create_key: bool=False):
	"""Recursively goes through dict-like d along the keys in tuple keys and sets the value to the child entry."""
	if len(keys) > 1:
		# Check and continue recursion
		if keys[0] in d:
			_recursive_setitem(d=d[keys[0]], keys=keys[1:],
			                   val=val, create_key=create_key)
		else:
			if create_key:
				d[keys[0]] 	= {}
				_recursive_setitem(d=d[keys[0]], keys=keys[1:],
				                   val=val, create_key=create_key)
			else:
				raise KeyError("No key '{}' found in dict {}; if it should be created, set create_key argument to True.".format(keys[0], d))
	else:
		# reached the end of the recursion
		d[keys[0]] 	= val

def _recursive_collect(itr, select_func, *select_args, prepend_info: tuple=None, parent_keys: tuple=None, info_func=None, info_func_kwargs: dict=None, **select_kwargs) -> list:
	"""Go recursively through the dict- or sequence-like (iterable) itr and call select_func(val, *select_args, **select_kwargs) on the values. If the return value is True, that value will be collected to a list, which is returned at the end.

	With `prepend_info`, information can be prepended to the return value. Then, not only the values but also these additional items can be gathered:
		`keys`  	: prepends the key
		`info_func` : prepends the return value of `info_func(val)`
	The resulting return value is then a list of tuples

	The argument parent_keys is used to pass on the key sequence of parent keys. (Necessary for the `items` mode.)
	"""

	# Return value list
	coll 	= []

	# Default values
	info_func_kwargs 	= info_func_kwargs if info_func_kwargs else {}

	# TODO check more generally for iterables?!
	if isinstance(itr, dict):
		iterator 	= itr.items()
	elif isinstance(itr, (list, tuple)):
		iterator 	= enumerate(itr)
	else:
		raise TypeError("Cannot iterate through argument itr of type {}".format(type(itr)))

	for key, val in iterator:
		# Generate the tuple of parent keys... for this iterator of the loop
		if parent_keys is None:
			these_keys 	= (key,)
		else:
			these_keys 	= parent_keys + (key,)

		# Apply the select_func and, depending on return, continue recursion or not
		if select_func(val, *select_args, **select_kwargs):
			# found the desired element
			# Distinguish cases where information is prepended and where not
			if not prepend_info:
				entry 	= val
			else:
				entry 	= (val,)
				# Loop over the keys to prepend in reversed order (such that the order of the given tuple is not inverted)
				for info in reversed(prepend_info):
					if info in ['key', 'keys']:
						entry 	= (these_keys,) + entry
					elif info in ['info_func']:
						entry 	= (info_func(val, **info_func_kwargs),) + entry
					else:
						raise ValueError("No such `prepend_info` entry implemented: "+str(info))

			# Add it to the return list
			coll.append(entry)

		elif isinstance(val, (dict, list, tuple)):
			# Not the desired element, but recursion possible ...
			coll 	+= _recursive_collect(val, select_func, *select_args,
			                              prepend_info=prepend_info,
			                              info_func=info_func,
			                              info_func_kwargs=info_func_kwargs,
			                              parent_keys=these_keys,
			                              **select_kwargs)

		else:
			# is something that cannot be selected and cannot be further recursed ...
			pass

	return coll

def _recursive_replace(itr, replace_func, select_func, *select_args, replace_kwargs=None, **select_kwargs) -> list:
	"""Go recursively through the dict- or sequence-like (iterable) itr and call select_func(val, *select_args, **select_kwargs) on the values. If the return value is True, that value will be collected to a list, which is returned at the end."""

	replace_kwargs 	= replace_kwargs if replace_kwargs else {}

	# TODO further types possible?!
	if isinstance(itr, dict):
		iterator 	= itr.items()
	elif isinstance(itr, (list, tuple)):
		iterator 	= enumerate(itr)
	else:
		raise TypeError("Cannot iterate through given iterable of type {}".format(type(itr)))

	for key, val in iterator:
		if select_func(val, *select_args, **select_kwargs):
			# found the desired element -> replace by the value returned from the replace_func
			itr[key] 	= replace_func(val, **replace_kwargs)

		elif isinstance(val, (dict, list, tuple)):
			# Not the desired element, but recursion possible ...
			itr[key] 	=  _recursive_replace(val, replace_func,
			                                  select_func, *select_args,
			                                  replace_kwargs=replace_kwargs,
			                                  **select_kwargs)

		else:
			# was not selected and cannot be further recursed, thus: stays the same
			pass

	return itr
