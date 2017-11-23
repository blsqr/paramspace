# -*- coding: utf-8 -*-
''' The ParamSpace class as an extension of a dict, which can be used to span over a paramter space.'''

import copy
import logging
import pprint
from collections import OrderedDict, Mapping

import numpy as np

# Get logger
log = logging.getLogger(__name__)

# TODO
# 	- add yaml constructors here?
# 	- more readable argument passing to ParamSpan
# 	- more Duck Typing, less type checking
# 	- clean up ParamSpace: more private names, properties, iterators; cleaner API; ...
# 	- features: better ParamSpace slicing, e.g. with __getitem__ and returning subspaces etc.
# -----------------------------------------------------------------------------

class ParamSpanBase:
	''' The ParamSpan base class. This is used so that the actual ParamSpan class can be a child class of this one and thus be distinguished from CoupledParamSpan'''

	def __init__(self, arg):
		''' Initialise the ParamSpan from an argument that is a list, tuple or a dict.'''

		# Set attributes to default values
		self.enabled 	= True
		self.state 		= None 	# State of the span (idx of the current value or None, if default state)
		self.span 		= None 	# Filled with values below
		self.name 		= None
		self.order 		= None

		# Parse the argument and fill the span
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
			if arg.get('add'):
				add = arg['add']
				if isinstance(add, (list, tuple)):
					self.span += list(add)
				else:
					self.span.append(add)

			# Optionally, cast to int or float
			# TODO Make this a key-value argument, not three key-bool pairs
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
			self.enabled 	= arg.get('enabled', True)

			# And it can have a name
			self.name 		= arg.get('name', None)

			# Also set the order
			self.order 		= arg.get('order', np.inf) # default value (i.e.: last) if no order is supplied

			log.debug("Initialised ParamSpan object from mapping.")

		else:
			# TODO not strictly necessary; just let it fail...
			raise TypeError("ParamSpan init argument needs to be of type list, tuple or dict, was {}.".format(type(arg)))

		return

	def __str__(self):
		return repr(self)

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__,
		                       repr(dict(default=self.default,
		                                 order=self.order,
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

		if isinstance(idx, int):
			try:
				return self.span[idx]

			except IndexError:
				# reached end of span
				# raise error - is caught by iterators to know that its finished
				# TODO should better be done using iteration and StopIteration error
				raise

		else:
			# Possbile that this is a slice. Try slicing and return as new ParamSpan object
			pspan 		= copy.deepcopy(self)
			pspan.span 	= self.span[idx]
			return pspan

	def __setitem__(self, idx, val):
		raise NotImplementedError("ParamSpan values are read-only.")

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

	def apply_slice(self, slc):
		'''Applies a slice to the span list.'''
		if not self.enabled:
			# Not enabled -> no span -> nothing to do
			return

		new_span 	= self.span[slc]
		if len(new_span) > 0:
			self.span 	= new_span
		else:
			raise ValueError("Application of slice {} to {}'s span {} resulted in zero-length span, which is illegal.".format(slc, self.__class__.__name__, self.span))

	def squeeze(self):
		'''If of length one, returns the remaining value in the span; if not, returns itself.'''
		if self.enabled and len(self) == 1:
			# Enabled and have span -> return the first element
			return self.span[0]

		elif not self.enabled:
			# Not enabled -> work on default and return that
			return self.default

		else:
			# Nothing to squeeze
			return self

# .............................................................................

class ParamSpan(ParamSpanBase):
	''' The ParamSpan class.'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.target_of 	= []

	def apply_slice(self, slc):
		super().apply_slice(slc)

		# Also apply to all coupled ParamSpans
		for cps in self.target_of:
			cps.apply_slice(slc)

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

		self.coupled_pspan 	= None # ParamSpace sets this after initialisation
		self.coupled_to 	= arg['coupled_to']

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

		# Traverse the dict and look for ParamSpan objects; collect them as (order, key, value) tuples
		pspans	= _recursive_collect(d, isinstance, ParamSpan,
		                             prepend_info=('info_func', 'keys'),
		                             info_func=lambda ps: ps.order)

		# Sort them. This looks at the info first, which is the order entry, and then at the keys. If a ParamSpan does not provide an order, it has entry np.inf there, such that those without order get sorted by the key.
		pspans.sort()
		# NOTE very important for consistency

		# Now need to reduce the list items to 2-tuples, ditching the order, to allow to initialise the OrderedDict
		pspans 	= [tpl[1:] for tpl in pspans]

		# Cast to an OrderedDict (pspans is a list of tuples -> same data structure as OrderedDict)
		self._spans 	= OrderedDict(pspans)

		# Also collect the coupled ParamSpans and continue with the same procedure
		coupled = _recursive_collect(d, isinstance, CoupledParamSpan,
		                             prepend_info=('info_func', 'keys'),
		                             info_func=lambda ps: ps.order)
		coupled.sort() # same sorting rules as above, but not as crucial here because they do not change the iteration order through state space
		self._cpspans 	= OrderedDict([tpl[1:] for tpl in coupled])

		# Now resolve the coupling targets and add them to CoupledParamSpan instances ... also let the target ParamSpan objects know which CoupledParamSpan couples to them
		for cpspan in self._cpspans.values():
			c_target_name		= cpspan.coupled_to

			# Try to get it by name
			c_target 			= self.get_span_by_name(c_target_name)

			# Set attribute of the coupled ParamSpan
			cpspan.coupled_pspan= c_target

			# And inform the target ParamSpan about it being the target of the coupled param span, if it is not already included there
			if cpspan not in c_target.target_of:
				c_target.target_of.append(cpspan)

		log.debug("Initialised %d spans and %d coupled spans.", len(self._spans), len(self._cpspans))

	# Formatting ..............................................................

	def __str__(self):
		log.debug("__str__ called. Returning current state dict.")
		return pprint.pformat(self._dict)

	def __repr__(self):
		'''To reconstruct the ParamSpace object ...'''
		return "ParamSpace("+str(self)+")"

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
			l.append("  * {}".format(" -> ".join([str(e) for e in name])))
			l.append("      {}".format(span.value_list))
			l.append("")

		# Coupled Span information
		if len(self._cpspans):
			l += ["", "Coupled Parameter Spans"]
			l += ["  (Move alongside the state of Parameter Spans)", ""]

			for name, cspan in self._cpspans.items():
				l.append("  * {}".format(" -> ".join([str(e) for e in name])))
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
	def shape(self) -> tuple:
		'''The shape of the parameter space'''
		return tuple([len(s) for s in self.get_spans()])

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
	def coupled_spans(self):
		''' Return the OrderedDict that holds the coupled spans.'''
		return self._cpspans

	@property
	def span_names(self) -> list:
		''' Get a list of the span names (tuples of strings). If the span was itself named, that name is used rather than the one created from the dictionary key.

		NOTE: CoupledParamSpans are not included here, same as in the other methods.'''
		names 	= []

		for name, span in self.spans.items():
			if span.name:
				names.append((span.name,))
			else:
				names.append(name)

		return names

	# TODO migrate the following to properties

	def get_default(self):
		''' Returns the default state of the ParamSpace'''
		_dd = _recursive_replace(copy.deepcopy(self._init_dict),
		                         lambda pspan: pspan.default,
		                         isinstance, ParamSpanBase)
		return self._return_class(_dd)

	def get_point(self):
		''' Return the current point in Parameter Space (i.e. corresponding to the current state).'''
		_pd = _recursive_replace(copy.deepcopy(self._dict),
		                         lambda pspan: pspan.get_val_by_state(),
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

	def get_coupled_spans(self):
		''' Return the coupled spans'''
		return self.coupled_spans.values()

	def get_span_keys(self):
		''' Get the iterator over the span keys (tuples of strings).'''
		return self._spans.keys()

	def get_span_names(self) -> list:
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

	def get_points(self, fstr: str=None, with_span_states: bool=False) -> tuple:
		''' Returns a generator of all states in state space, returning (state_no, point in state space).

		If `with_span_states` is True, the span states tuple is returned instead of the state number'''
		if fstr is not None and not isinstance(fstr, str):
			raise TypeError("Argument fstr needs to be a string or None, was "+str(type(fstr)))
		elif fstr is None:
			# No additional return value
			_add_ret 	= ()
		# else: will use format string, even if it is empty

		if with_span_states:
			first_tuple_element 	= self.get_span_states
		else:
			first_tuple_element 	= self.get_state_no

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
		yield (first_tuple_element(), self.get_point()) + _add_ret

		# Now yield all the other states
		while self.next_state():
			_add_ret 	= () if not fstr else (fstr.format(self),)
			yield (first_tuple_element(), self.get_point()) + _add_ret

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

	# Getting a subspace ......................................................

	def get_subspace(self, *slices, squeeze: bool=True, as_dict_if_0d: bool=False):
		'''Returns a copy of this ParamSpace with the slices applied to the corresponding ParamSpans.

		If `squeeze`, the size one spans are removed.

		 (Not the nicest implementation overall...)'''

		def apply_slice(pspace, *, slc, name: str):
			'''Destructively (!) applies a slice to the span with the given name.'''
			pspan 	= pspace.get_span_by_name(name)
			pspan.apply_slice(slc)


		# Work on a copy of this ParamSpace
		subspace 	= copy.deepcopy(self)

		# Check if the length of the provided slices matches the number of dimensions that could possibly be sliced
		if len(slices) <= subspace.num_dimensions:
			# See if there are Ellipses in the slices that indicate where to expand the list
			num_ellipses = sum([s is Ellipsis for s in slices])
			if num_ellipses == 0:
				# No.
				if len(slices) < subspace.num_dimensions:
					# Add one in the end so that it is clear where to expand.
					slices.append(Ellipsis)
				# else: there was one slice defined for each dimension; can and need not add an Ellipsis

			elif num_ellipses > 1:
				raise ValueError("More than one Ellipsis object given!")

			# Now expand them so that the slices list has the same length as the source parameter space has dimensions
			_slices 	= []
			for slc in slices:
				if slc is Ellipsis:
					# Put a number of slice(None) in place of the Ellipsis
					fill_num 	= subspace.num_dimensions - len(slices) + 1
					_slices 	+= [slice(None) for _ in range(fill_num)]
				else:
					_slices.append(slc)

			# Use the new slices list
			slices 		= _slices

		else:
			raise ValueError("More slices than dimensions that could potentially be sliced given.")

		# Get the list of names
		names 		= subspace.get_span_names()

		# For each name, apply the slice
		for name, slc in zip(names, slices):
			apply_slice(subspace, slc=slc, name=name)
			# NOTE this works directly on the ParamSpan objects

		# Have the option to squeeze away the size-1 ParamSpans
		if squeeze:
			subspace 	= _recursive_replace(subspace._dict,
			                                 lambda pspan: pspan.squeeze(),
			                                 isinstance, ParamSpanBase)
		else:
			# Just extract the subspace dictionary
			subspace 	= subspace._dict
		# The previous subspace ParamSpace object will go out of scope here. The changes however were applied to the ParamSpan objects that are stored in the dictionaries.

		# Now, a new ParamSpace object should be initialised, because the old one was messed with too much.
		subspace 	= ParamSpace(subspace)

		# Only now is it clear how many dimensions the target space will have. If it is 0-dimensional (i.e. no ParamSpans inside) flatten it (if argument is set to do so)
		if as_dict_if_0d and subspace.num_dimensions == 0:
			# Overwrite with the default, which is the same as the current dict. There is no difference, because there are no ParamSpans defined anyway ...
			subspace 	= subspace.get_default()

		return subspace

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

def _recursive_collect(itr, select_func, *select_args, prepend_info: tuple=None, parent_keys: tuple=None, info_func=None, info_func_kwargs: dict=None, **select_kwargs) -> list:
	''' Go recursively through the dict- or sequence-like (iterable) itr and call select_func(val, *select_args, **select_kwargs) on the values. If the return value is True, that value will be collected to a list, which is returned at the end.

	With `prepend_info`, information can be prepended to the return value. Then, not only the values but also these additional items can be gathered:
		`keys`  	: prepends the key
		`info_func` : prepends the return value of `info_func(val)`
	The resulting return value is then a list of tuples

	The argument parent_keys is used to pass on the key sequence of parent keys. (Necessary for the `items` mode.)
	'''

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
	''' Go recursively through the dict- or sequence-like (iterable) itr and call select_func(val, *select_args, **select_kwargs) on the values. If the return value is True, that value will be collected to a list, which is returned at the end.'''

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
