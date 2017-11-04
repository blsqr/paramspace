# -*- coding: utf-8 -*-
''' The ParamSpace class as an extension of a dict, which can be used to span over a paramter space.'''

import copy
import logging
from collections import OrderedDict, Mapping

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# TODO
# * add ordering of spans in ParamSpace
# * add yaml constructors
# -----------------------------------------------------------------------------

class ParamSpanBase:
	''' The ParamSpan base class. This is used so that the actual ParamSpan class can be a child class of this one and thus be distinguished from CoupledParamSpan'''

	def __init__(self, arg): # TODO more readable argument passing
		''' Initialise the ParamSpan from an argument that is a list, tuple or a dict.'''

		# Set attributes
		self.state 		= None # State of the span (idx of the current value or None, if default state)
		self.enabled 	= True
		self.name 		= None

		if isinstance(arg, (list, tuple)):
			# Initialise from sequence: first value is default, all are span
			self.default 	= arg[0]
			self.span 		= arg[:]
			# NOTE the default value is only inside the span, if the pspan is defined via sequence!

			log.debug("Initialised ParamSpan object from sequence.")

		elif isinstance(arg, dict):
			# Get default value
			self.default 	= arg['default']

			# Get either of the span constructors
			if 'span' in arg:
				self.span 	= list(arg['span'])

			elif 'range' in arg:
				self.span 	= list(range(*arg['range']))

			elif 'linspace' in arg:
				# explicit float casting, because else numpy objects are somehow retained
				self.span 	= [float(x) for x in np.linspace(*arg['linspace'])]

			elif 'logspace' in arg:
				# explicit float casting, because else numpy objects are somehow retained
				self.span 	= [float(x) for x in np.logspace(*arg['logspace'])]

			else:
				raise ValueError("No valid span key (span, range, linspace, logspace) found in init argument, got {}.".format(arg.keys()))

			# Add additional values to the span
			if 'add' in arg:
				add = arg['add']
				if isinstance(add, (list, tuple)):
					self.span += list(add)
				else:
					self.span.append(add)

			# Optionally, cast to int or float
			if arg.get('as_int'):
				self.span 	= [int(v) for v in self.span]

			elif arg.get('as_float'):
				self.span 	= [float(v) for v in self.span]

			elif arg.get('as_str'):
				self.span 	= [str(v) for v in self.span]

			# If the state idx was given, also save this
			if isinstance(arg.get('state'), int):
				log.warning("Setting state of ParamSpan during initialisation. This might lead to unexpected behaviour if iterating over points in ParamSpace.")
				self._state 	= arg.get('state')

			# A span can also be not enabled
			if not arg.get('enabled', True):
				self.enabled 	= False

			# And it can have a name
			if 'name' in arg:
				self.name 		= arg.get('name')

			log.debug("Initialised ParamSpan object from mapping.")

		else:
			raise TypeError("ParamSpan init argument needs to be of type list, tuple or dict, was {}.".format(type(arg)))

		return

	def __str__(self):
		return repr(self)

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__,
		                       repr(dict(default=self.default,
                                         span=self.span,
                                         state=self.state,
                                         enabled=self.enabled,
                                         name=self.name)))

	def __len__(self):
		''' Return how many span values there are, if the span is enabled.'''
		if self.enabled:
			return len(self.span)
		else:
			return 1

	def __getitem__(self, idx):
		if not self.enabled:
			log.warning("ParamSpan is not enabled. Still returning item ...")
		try:
			return self.span[idx]
		except IndexError:
			# reached end of span
			# raise error - is caught by iterators to know that its finished
			raise

	# Properties

	@property
	def value_list(self):
		return self.span

	# Public methods

	def get_val_by_state(self):
		''' Returns the current ParamSpan value according to the state. This is the main method used by the ParamSpace to resolve the dictionary to its correct state.'''
		if self.state is None:
			return self.default
		else:
			return self.span[self.state]

	def next_state(self) -> bool:
		''' Increments the state by one, if the state is enabled.

		If None, sets the state to 0.

		If reaching the last possible state, it will restart at zero and return False, signalising that all states were looped through. In all other cases it will return True.
		'''
		log.debug("ParamSpan.next_state called ...")

		if not self.enabled:
			return False

		if self.state is None:
			self.state 	= 0
		else:
			self.state 	= (self.state+1)%len(self)

			if self.state == 0:
				# It is 0 after increment, thus it must have hit the wall.
				return False

		return True

	def set_state_to_zero(self) -> bool:
		''' Sets the state to zero (necessary before the beginning of an iteration), if the span is enabled.'''
		if self.enabled:
			self.state = 0
			return True
		return False

# .............................................................................

class ParamSpan(ParamSpanBase):
	''' The ParamSpan class.'''

# .............................................................................

class CoupledParamSpan(ParamSpanBase):
	''' A CoupledParamSpan object is recognized by the ParamSpace and its state moves alongside with another ParamSpan's state.'''

	def __init__(self, arg):

		# Check if default and/or span were not given; in those cases, the values from the coupled span are to be used upon request
		self.use_coupled_default 	= bool('default' not in arg)
		self.use_coupled_span 		= bool('span' not in arg)

		# Make sure they are set, so parent init does not get confused
		arg['default'] 	= arg.get('default')
		if arg.get('span') is None:
			arg['span'] 	= []

		super().__init__(arg)

		self.coupled_to 	= arg['coupled_to']
		self.coupled_pspan 	= None # ParamSpace sets this after initialisation

		if not isinstance(self.coupled_to, str):
			# ensure it is a tuple; important for span name lookup
			self.coupled_to 	= tuple(self.coupled_to)

		log.debug("CoupledParamSpan initialised.")

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__,
		                       repr(dict(default=self.default,
                                         span=self.span,
                                         state=self.state,
                                         enabled=self.enabled,
                                         name=self.name,
                                         coupled_to=self.coupled_to)))

	# Overwrite the properties that need to relay the coupling to the other ParamSpan

	@property
	def default(self):
		''' If the CoupledParamSpan was initialised with a default value on its own, returns that. If not, returns the default value of the coupled ParamSpan object. If not yet coupled to that, returns None.'''
		if not self.use_coupled_default:
			return self._default
		elif self.coupled_pspan:
			return self.coupled_pspan.default
		else:
			return None

	@default.setter
	def default(self, val):
		self._default = val

	@property
	def span(self):
		''' If the CoupledParamSpan was initialised with a span on its own, returns that span. If not, returns the span of the coupled ParamSpan object. If not yet coupled to that, returns None.'''
		if not self.use_coupled_span:
			return self._span
		elif self.coupled_pspan:
			return self.coupled_pspan.span
		else:
			return None

	@span.setter
	def span(self, val):
		self._span = val

	@property
	def state(self):
		if self.coupled_pspan:
			return self.coupled_pspan.state
		else:
			return None

	@state.setter
	def state(self, val):
		self._state = val

	@property
	def enabled(self):
		if self.coupled_pspan:
			return self.coupled_pspan.enabled
		else:
			return self._enabled

	@enabled.setter
	def enabled(self, val):
		self._enabled 	= val

	# Methods

	def get_val_by_state(self):
		''' Adds a try-except clause to the parent method, to give an understandable error message in case of an index error (due to a coupled span with unequal length).'''
		try:
			return super().get_val_by_state()
		except IndexError as err:
			if not self.use_coupled_span and len(self) != len(self.coupled_pspan):
				raise IndexError("The span provided to CoupledParamSpan has not the same length as that of the ParamSpan it couples to. Lengths: {} and {}.".format(len(self), len(self.coupled_pspan))) from err

			raise

# -----------------------------------------------------------------------------

class ParamSpace:

	def __init__(self, d, return_class=dict):
		''' Initialise the ParamSpace object from a dictionary.

		Upon init, the dictionary is traversed; when meeting a ParamSpan object, it will be collected and then added to the spans.
		'''

		log.debug("Initialising ParamSpace ...")

		self._init(d, return_class=return_class)

		# Done.
		log.info("Initialised ParamSpace object. (%d dimensions, volume %d)", len(self._spans), self._max_state)

	def _init(self, d, return_class=dict):
		''' Initialisation helper, which is called from __init__ and from update. It Initialises the base dictionaries (_init_dict and _dict) as well as the spans and some variables regarding state number.
		'''

		# Keep the initial dictionary. This will never be messed with (only exception being an update, where this _init method is called again).
		self._init_dict = copy.deepcopy(d) 	# includes the ParamSpan objects

		# The current dictionary (in default state as copy from initial dict)
		# This dictionary is what is returned on request and what is worked on.
		self._dict 		= copy.deepcopy(self._init_dict)

		# Initialise the self._spans attribute
		self._spans 	= None 				# ...is defined in _init_spans
		self._init_spans(self._dict)
		# self._spans is an OrderedDict, which includes as keys the name of the span keys (or a tuple with the traversal path to an entry), and as values the ParamSpan objects
		# The current state of the parameter space is saved in the ParamSpan objects and can be incremented there as well.

		# Additionally, the state_id counts the number of the point the current dictionary is in. It is incremented upon self._next_state() until the number self._max_state is reached.
		self._state_no 	= None	 		 	# None means: default vals
		self._max_state = self.volume

		# The requested ParamSpace points can be cast to a certain class:
		self._return_class = return_class

		# The inverse mapping can be cached
		self._imap 		= None

		return

	def _init_spans(self, d):
		''' Looks for instances of ParamSpan in the dictionary d, extracts spans from there, and carries them over
		- Their default value stays in the init_dict
		- Their spans get saved in the spans dictionary
		'''
		log.debug("Initialising spans ...")

		# Traverse the dict and look for ParamSpan objects; collect them and their key tuples
		pspans	= _recursive_collect(d, isinstance, ParamSpan, _kv_mode=True)

		# Sort by first key (very important for consistency!)
		pspans.sort(key=lambda tup: tup[0])

		# Cast to an OrderedDict (pspans is a list of tuples -> same data structure as OrderedDict)
		self._spans 	= OrderedDict(pspans)

		# Also collect the coupled ParamSpans and continue with the same procedure
		coupled = _recursive_collect(d, isinstance, CoupledParamSpan,
		                             _kv_mode=True)
		coupled.sort(key=lambda tup: tup[0])
		self._cpspans 	= OrderedDict(coupled)

		# Now resolve the coupling targets and add them to CoupledParamSpan instances ...and create a dict where the coupled span can be accessed via the name of the span it couples to
		self._cpspan_targets 	= {}

		for cpspan in self._cpspans.values():
			c_target 				= cpspan.coupled_to
			cpspan.coupled_pspan 	= self.get_span_by_name(c_target)
			self._cpspan_targets[c_target] 	= cpspan

		# NOTE: it is not necessary to have the coupled pspans collected here, because their replacement happens completely in the get_default and get_point methods

		log.debug("Initialised %d spans and %d coupled spans.", len(self._spans), len(self._cpspans))

	# Formatting ..............................................................

	def __str__(self):
		log.debug("__str__ called. Returning current state dict.")
		return str(self._dict)

	def __repr__(self):
		return "ParamSpace("+repr(self._dict)+")"

	def __format__(self, spec: str):
		''' Returns a formatted string

		The spec argument is the part right of the colon in the '{foo:bar}' of a format string.
		'''

		ALLOWED_JOIN_STRS 	=  ["_", "__"]

		# Special behaviour
		if len(spec) == 0:
			return ""

		elif spec == 'span_names':
			# Compile output for span names
			return "  (showing max. last 4 keys)\n  " + "\n  ".join([("" if len(s)<=4 else "."*(len(s)-4)+" -> ") + " -> ".join(s[-min(len(s),4):]) for s in self.get_span_names()])
			# ...a bit messy, but well ...


		# Creating span strings
		parts 		= []
		spst_fstr 	= "" # span state format string
		join_char	= ""

		# First: build the format string that will be used to handle each param space
		for part in spec.split(","):
			part 	= part.split("=")

			# Catch changes of the join character
			if len(part) == 1 and part[0] in ALLOWED_JOIN_STRS:
				join_char 	= part[0]
				continue

			# Catch span state format
			if len(part) == 2 and part[0] == "states":
				spst_fstr 	= part[1].replace("[", "{").replace("]", "}")
				continue

			# Pass all other parsing to the helper
			try:
				parsed 	= self._parse_format_spec(part)
			except ValueError:
				print("Invalid format string '{}'.".format(spec))
				raise
			else:
				if parsed:
					parts.append(parsed)

		if spst_fstr:
			# Evaluate the current values of the ParamSpace
			names 	= [key[-1] for key in self.get_span_names()]
			states 	= [span.state for span in self.get_spans()]
			vals 	= [span.get_val_by_state() for span in self.get_spans()]
			digits 	= [len(str(len(span))) for span in self.get_spans()]

			spst_parts = [spst_fstr.format(name=n, state=s, digits=d, val=v)
						  for n, s, d, v in zip(names, states, digits, vals)]

			parts.append("_".join(spst_parts))

		return join_char.join(parts)

	def _parse_format_spec(self, part: list): # TODO

		return None # currently not implementedd

		# if len(part) == 2:
		# 	# is a key, value pair
		# 	key, val 	= part

		# 	if key in ["bla"]:
		# 		pass # TODO
		# 	else:
		# 		raise ValueError("Invalid key value pair '{}: {}'.".format(key, val))

		# elif len(part) == 1:
		# 	key 	= part[0]

		# 	if key in ["bla"]:
		# 		pass # TODO
		# 	else:
		# 		raise ValueError("Invalid key '{}'.".format(key))

		# else:
		# 	raise ValueError("Part '{}' had more than one '=' as separator.".format("=".join(part)))

		# return None

	def get_info_str(self) -> str:
		'''Returns an information string about the ParamSpace'''
		l = ["ParamSpace Information"]

		# General information about the Parameter Space
		l.append("  Dimensions:  {}".format(self.num_dimensions))
		l.append("  Volume:      {}".format(self.volume))

		# Span information
		l += ["", "Parameter Spans"]
		l += ["  (First spans are iterated over first.)", ""]

		for name, span in self.spans.items():
			l.append("  * {}".format(" -> ".join(name)))
			l.append("      {}".format(span.value_list))
			l.append("")

		# Coupled Span information
		if len(self._cpspans):
			l += ["", "Coupled Parameter Spans"]
			l += ["  (Move alongside the state of Parameter Spans)", ""]

			for name, cspan in self._cpspans.items():
				l.append("  * {}".format(" -> ".join(name)))
				l.append("      Coupled to:  {}".format(cspan.coupled_to))
				l.append("      Span:        {}".format(cspan.value_list))
				l.append("")

		return "\n".join(l)

	# Retrieving states of the ParamSpace .....................................

	@property
	def num_dimensions(self) -> int:
		''' Returns the number of dimensions, i.e. the number of spans.'''
		return len(self._spans)

	@property
	def volume(self) -> int:
		''' Returns the volume of the parameter space.'''
		vol 	= 1
		for pspan in self.get_spans():
			vol *= len(pspan)
		return vol

	@property
	def spans(self):
		''' Return the OrderedDict that holds the spans.'''
		return self._spans

	@property
	def span_names(self) -> list:
		''' Get a list of the span names (tuples of strings). If the span was itself named, that name is used rather than the one created from the dictionary key.'''
		names 	= []

		for name, span in self._spans.items():
			if span.name:
				names.append((span.name,))
			else:
				names.append(name)

		return names

	# TODO migrate the following to properties

	def get_default(self):
		''' Returns the default state of the ParamSpace'''
		_dd = _recursive_replace_by_attr(copy.deepcopy(self._init_dict),
		                                 'default',
		                                 isinstance, ParamSpanBase)
		return self._return_class(_dd)

	def get_point(self):
		''' Return the current point in Parameter Space (i.e. corresponding to the current state).'''
		_pd = _recursive_replace_by_attr(copy.deepcopy(self._dict),
		                                 'get_val_by_state',
		                                 isinstance, ParamSpanBase)
		return self._return_class(_pd)

	def get_state_no(self) -> int:
		''' Returns the state number'''
		return self._state_no

	def get_span(self, dim_no: int) -> ParamSpan:
		try:
			return list(self.get_spans())[dim_no]
		except IndexError:
			log.error("No span corresponding to argument dim_no {}".format(dim_no))
			raise

	def get_spans(self):
		''' Return the spans'''
		return self._spans.values()

	def get_span_keys(self):
		''' Get the iterator over the span keys (tuples of strings).'''
		return self._spans.keys()

	def get_span_names(self):
		''' Get a list of the span names (tuples of strings). If the span was itself named, that name is used rather than the one created from the dictionary key.'''
		return self.span_names

	def get_span_states(self):
		''' Returns a tuple of the current span states'''
		return tuple([span.state for span in self.get_spans()])

	def get_span_dim_no(self, name: str) -> int:
		''' Returns the dimension number of a span, i.e. the index of the ParamSpan object in the list of spans of this ParamSpace. As the spans are held in an ordered data structure, the dimension number can be used to identify the span. This number also corresponds to the index in the inverse mapping of the ParamSpace.

		Args:
			name (tuple, str) : the name of the span, which can be a tuple of strings or a string. If name is a tuple of strings, the exact tuple is required to find the span by its span_name. If name is a string, only the last element of the span_name is considered.

		Returns:
			int 	: the number of the dimension
			None 	: a span by this name was not found

		Raises:
			ValueError: If argument name was only a string, there can be duplicates. In the case of duplicate entries, a ValueError is raised.
		'''
		dim_no 	= None

		if isinstance(name, str):
			for n, span_name in enumerate(self.get_span_names()):
				if span_name[-1] == name:
					if dim_no is not None:
						# Was already set -> there was a duplicate
						raise ValueError("Duplicate span name {} encountered during access via the last key of the span name. To not get an ambiguous result, pass the full span name as a tuple.".format(name))
					dim_no 	= n

		else:
			for n, span_name in enumerate(self.get_span_names()):
				if span_name[-len(name):] == name:
					# The last part of the sequence matches the given name
					if dim_no is not None:
						# Was already set -> there was a duplicate
						raise ValueError("Access via '{}' was ambiguous. Give the full sequence of strings as a span name to be sure to access the right element.".format(name))
					dim_no 	= n

		return dim_no

	def get_span_by_name(self, name: str) -> ParamSpan:
		''' Returns the ParamSpan corresponding to this name.

		Args:
			name (tuple, str) : the name of the span, which can be a tuple of strings or a string. If name is a tuple of strings, the exact tuple is required to find the span by its span_name. If name is a string, only the last element of the span_name is considered.

		Returns:
			int 	: the number of the dimension
			None 	: a span by this name was not found

		Raises:
			ValueError: If argument name was only a string, there can be duplicates. In the case of duplicate entries, a ValueError is raised.
		'''

		return self.get_span(self.get_span_dim_no(name))

	def get_inverse_mapping(self) -> np.ndarray:
		''' Creates a mapping of the state tuple to a state number and the corresponding span parameters.

		Returns:
			np.ndarray with the shape of the spans and the state number as value
		'''

		if hasattr(self, '_imap') and self._imap is not None:
			# Return the cached result
			# NOTE hasattr is needed for legacy reasons: old objects that are loaded from pickles and do not have the attribute ...
			log.debug("Using previously created inverse mapping ...")
			return self._imap
		# else: calculate the inverse mapping

		# Create empty n-dimensional array
		shape 	= tuple([len(_span) for _span in self.get_spans()])
		imap 	= np.ndarray(shape, dtype=int)
		imap.fill(-1) 	# -> Not set yet

		# Iterate over all points and save the state number to the map
		for state_no, _ in self.get_points():
			# Get the span states and convert all Nones to zeros, as these dimensions have no entry
			s = [Ellipsis if i is None else i for i in self.get_span_states()]

			# Save the state number to the mapping
			try:
				imap[tuple(s)]	= state_no
			except IndexError:
				log.error("Getting inverse mapping failed.")
				print("s: ", s)
				print("imap shape: ", imap.shape)
				raise

		# Save the result to attributes
		self._imap 	= imap

		return imap

	# Iterating over the ParamSpace ...........................................

	def get_points(self, fstr: str=None) -> tuple:
		''' Returns a generator of all states in state space. If with_state_no is True, a tuple of point number and the point will be returned instead.'''
		if fstr is not None and not isinstance(fstr, str):
			raise TypeError("Argument fstr needs to be a string or None, was "+str(type(fstr)))
		elif fstr is None:
			# No additional return value
			_add_ret 	= ()
		# else: will use format string, even if it is empty

		if self.num_dimensions == 0:
			log.warning("No dimensions in ParamSpace. Returning defaults.")

			if fstr:
				_add_ret	= ('',)

			yield (None, self.get_default()) + _add_ret
			return # not executed further

		# else: there is a volume to iterate over:

		# Prepare pspans: set them to state 0, else they start with the default
		for pspan in self.get_spans():
			pspan.set_state_to_zero()

		# This is the initial state with state number 0
		self._state_no 	= 0

		# Determine state string
		_add_ret 	= () if not fstr else (fstr.format(self),)

		# Yield the initial state
		yield (self.get_state_no(), self.get_point()) + _add_ret

		# Now yield all the other states
		while self.next_state():
			_add_ret 	= () if not fstr else (fstr.format(self),)
			yield (self.get_state_no(), self.get_point()) + _add_ret

		else:
			log.info("Visited every point in ParamSpace.")
			log.info("Resetting to initial state ...")
			self.reset()
			return

	def next_state(self) -> bool:
		''' Increments the state variable'''
		log.debug("ParamSpace.next_state called ...")

		for pspan in self.get_spans():
			if pspan.next_state():
				# Iterated to next step without reaching the last span item
				break
			else:
				# Went through all states of this span -> carry one over (as with addition) and increment the next spans in the ParamSpace.
				continue
		else:
			# Loop went through -> all states visited
			self.reset()			# reset back to default
			return False			# i.e. reached the end

		# Broke out of loop -> Got the next state and not at the end yet

		# Increment state number
		if self._state_no is None:
			self._state_no = 0
		else:
			self._state_no += 1

		return True				# i.e. not reached the end

	# Misc ....................................................................

	def reset(self):
		''' Resets all state variables, the state id, and the current dictionary back to the initial dictionary (i.e. with the default values).'''

		for pspan in self.get_spans():
			# Reset the pspan state
			pspan.state 	= None

		self._state_no 	= None

		log.debug("ParamSpace resetted.")

	def add_span(self): # TODO
		''' Add a span to the ParamSpace manually, e.g. after initialisation with a regular dict.'''
		raise NotImplementedError("Manually adding a span is not implemented yet. Please initialise the ParamSpace object with the ParamSpan objects already in place.")

	def update(self, u, recessively: bool=True):
		''' Update the dictionaries of the ParamSpace with the values from u.

		If recessively is True, the update dictionary u will be updated with the values from self._dict.
		For False, the regular dictionary update will be performed, where self._dict is updated with u.
		'''

		if self.get_state_no() is not None:
			log.warning("ParamSpace object can only be updated in the default state, but was in state %s. Call .reset() on the ParamSpace to return to the default state.", self.get_state_no())
			return

		if recessively:
			# Recessive behaviour: Old values have priority
			new_d 	= _recursive_update(u, self._dict)
		else:
			# Normal update: New values overwrite old ones
			log.info("Performing non-recessive update. Note that the new values have priority over any previous values, possibly overwriting ParamSpan objects with the same keys.")
			new_d 	= _recursive_update(self._dict, u)

		# In order to make the changes apply to _dict and _init_dict, the _init method is called again. This makes sure, the ParamSpace object is in a consistent state after the update.
		self._init(new_d, return_class=self._return_class)

		log.info("Updated ParamSpace object.")

		return

# -----------------------------------------------------------------------------

def _recursive_update(d: dict, u: dict):
	''' Update Mapping d with values from Mapping u'''
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
	''' Checks on the dict-like d, whether a key is present. If the key is a tuple with more than one key, it recursively continues checking.'''
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
	''' Recursively goes through dict-like d along the keys in tuple keys and returns the reference to the at the end.'''
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
	''' Recursively goes through dict-like d along the keys in tuple keys and sets the value to the child entry.'''
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

def _recursive_collect(itr, select_func, *select_args, _kv_mode: bool=False, _keys: tuple=None, **select_kwargs) -> list:
	''' Go recursively through the dict- or sequence-like (iterable) itr and call select_func(val, *select_args, **select_kwargs) on the values. If the return value is True, that value will be collected to a list, which is returned at the end.

	If _kv_mode is True, this will not collect the values, but also their keys. The argument _keys is used to pass on the tuple list of parent keys.
	'''

	coll 	= []

	# TODO check more generally for iterables?!
	if isinstance(itr, dict):
		iterator 	= itr.items()
	elif isinstance(itr, (list, tuple)):
		iterator 	= enumerate(itr)
	else:
		raise TypeError("Cannot iterate through argument itr of type {}".format(type(itr)))

	for key, val in iterator:
		# Generate the tuple of parent keys... for this iterator of the loop
		if _keys is None:
			these_keys 	= (key,)
		else:
			these_keys 	= _keys + (key,)

		# Apply the select_func and, depending on return, continue recursion or not
		if select_func(val, *select_args, **select_kwargs):
			# found the desired element
			if _kv_mode:
				coll.append((these_keys, val))
			else:
				coll.append(val)

		elif isinstance(val, (dict, list, tuple)):
			# Not the desired element, but recursion possible ...
			coll 	+= _recursive_collect(val, select_func,
			                              *select_args,
			                              _kv_mode=_kv_mode, _keys=these_keys,
			                              **select_kwargs)

		else:
			# is something that cannot be selected and cannot be further recursed ...
			pass

	return coll

def _recursive_replace_by_attr(itr, attr_name: str, select_func, *select_args, **select_kwargs) -> list:
	''' Go recursively through the dict- or sequence-like (iterable) itr and call select_func(val, *select_args, **select_kwargs) on the values. If the return value is True, that value will be collected to a list, which is returned at the end.'''

	# TODO further types possible?!
	if isinstance(itr, dict):
		iterator 	= itr.items()
	elif isinstance(itr, (list, tuple)):
		iterator 	= enumerate(itr)
	else:
		raise TypeError("Cannot iterate through argument itr of type {}".format(type(itr)))

	for key, val in iterator:
		if select_func(val, *select_args, **select_kwargs):
			# found the desired element -> replace by its attribute (call, if callable)
			if callable(getattr(val, attr_name)):
				itr[key] 	= getattr(val, attr_name).__call__()
			else:
				itr[key] 	= getattr(val, attr_name)


		elif isinstance(val, (dict, list, tuple)):
			# Not the desired element, but recursion possible ...
			itr[key] 	=  _recursive_replace_by_attr(val, attr_name,
			                                          select_func,
			                                          *select_args,
			                                          **select_kwargs)

		else:
			# was not selected and cannot be further recursed, thus: stays the same
			pass

	return itr
